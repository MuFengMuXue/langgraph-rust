//! Probe bench: upper bound of the `gather_input` deep-clone elimination.
//!
//! Background: `gather_input` (`pregel/algo.rs`) hands each task its input by
//! calling `channel.get()` per input channel. `get()` returns an **owned**
//! `serde_json::Value` (`channels/base.rs:24`), so every task pays a full deep
//! clone of every channel it reads. On a linear chain with a growing `messages`
//! history that is one full-array clone per super-step; in a fan-out that is
//! N clones of the shared input per super-step.
//!
//! The candidate fix is an architectural `Channel::get` change (borrow or
//! `Arc<JsonValue>`) so shared input is read once instead of cloned per task.
//! This bench measures the **upper bound** of that win *without touching the
//! engine*: it times `get()` on the real `messages` channel type
//! (`BinaryOperatorAggregate` + `add_messages_ref`) at the exact history sizes
//! a 4-super-step-per-turn ReAct agent sees at T turns (4T messages), then
//! cross-checks against the engine's real per-super-step time for the same
//! topology. The ratio `get()_cost / engine_per_superstep` is the maximum
//! fraction of engine time the clone elimination could reclaim.
//!
//! `checkpoint()` (also a full clone) is measured alongside because it is the
//! *parallel* O(history) term in the same super-step — the clone elimination
//! only touches `get()`, not checkpoint serialization.
//!
//! Run:
//! ```text
//! cargo test --release --test bench_gather_input -- --ignored --nocapture
//! ```

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use langgraph::channels::{BinaryOperatorAggregate, Channel};
use langgraph::checkpoint::InMemorySaver;
use langgraph::prebuilt::{
    add_messages_ref, tools_condition, BaseTool, Message, ToolCall, ToolNode,
};
use langgraph::prelude::*;
use langgraph::tool;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::StreamExt;

#[tool("fake_memory", "桩工具：模拟记忆搜索/保存的即时返回")]
async fn fake_memory(query: String) -> Result<String, String> {
    Ok(format!("桩结果：{query}"))
}

/// Build a realistic `messages` array of `n` messages, matching the shapes the
/// real ReAct bench produces (human / ai+tool_call / tool / final-ai cycles) so
/// the isolated clone cost mirrors the in-engine value byte-for-byte in shape.
fn build_messages(n: usize) -> Vec<JsonValue> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while out.len() < n {
        out.push(json!({"type": "human", "content": format!("第{i}条用户消息，你记得吗")}));
        if out.len() >= n {
            break;
        }
        out.push(
            serde_json::to_value(Message::ai_with_tool_calls(
                "",
                vec![ToolCall {
                    name: "fake_memory".to_string(),
                    args: json!({"query": "x"}),
                    id: Some(format!("call_{i}")),
                }],
            ))
            .unwrap(),
        );
        if out.len() >= n {
            break;
        }
        out.push(json!({
            "type": "tool",
            "content": "桩结果：x",
            "tool_call_id": format!("call_{i}"),
        }));
        if out.len() >= n {
            break;
        }
        out.push(serde_json::to_value(Message::ai("这是最终回答。")).unwrap());
        i += 1;
    }
    out
}

/// Time `channel.get()` (the deep clone `gather_input` pays per task) and
/// `channel.checkpoint()` (the parallel O(history) term) for a `messages`
/// channel holding `n_messages`. Returns per-call averages in microseconds.
fn measure_channel_ops(n_messages: usize, iters: usize) -> (f64, f64) {
    let ch: Box<dyn Channel> = Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref));
    ch.update(&[JsonValue::Array(build_messages(n_messages))])
        .unwrap();

    // get() — the operation gather_input's deep clone consists of.
    let mut sink = 0usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let v = ch.get().unwrap();
        sink = sink.wrapping_add(black_box(v).as_array().map(|a| a.len()).unwrap_or(0));
    }
    let get_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // checkpoint() — the parallel term (not touched by the get() change).
    let mut sink2 = 0usize;
    let t1 = Instant::now();
    for _ in 0..iters {
        let v = ch.checkpoint().unwrap();
        sink2 = sink2.wrapping_add(black_box(v).as_array().map(|a| a.len()).unwrap_or(0));
    }
    let cp_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

    assert!(sink > 0 && sink2 > 0, "black_box sinks must consume values");
    (get_us, cp_us)
}

fn build_react_agent(checkpointer: Arc<InMemorySaver>) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref)) as Box<dyn Channel>,
    );
    channels.insert(
        "search_context".to_string(),
        Box::new(langgraph::channels::LastValue::new("search_context")) as Box<dyn Channel>,
    );

    let mut graph = StateGraph::new(channels);

    graph
        .add_node(
            "search_memories",
            |input: JsonValue, _config: RunnableConfig| async move {
                let last_user = input
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|arr| arr.last())
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                let context = format!(
                    "相关记忆：{} {}",
                    last_user, "记忆内容记忆内容记忆内容记忆内容记忆内容"
                );
                Ok(json!({"search_context": context}))
            },
        )
        .unwrap();

    graph
        .add_node(
            "llm_call",
            |input: JsonValue, _config: RunnableConfig| async move {
                let msgs = input
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .cloned()
                    .unwrap_or_default();
                let last_type = msgs
                    .last()
                    .and_then(|m| m.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                if last_type == "tool" {
                    Ok(json!({"messages": [Message::ai("这是最终回答。")]}))
                } else {
                    Ok(json!({"messages": [Message::ai_with_tool_calls(
                        "",
                        vec![ToolCall {
                            name: "fake_memory".to_string(),
                            args: json!({"query": "x"}),
                            id: Some("call_1".to_string()),
                        }],
                    )]}))
                }
            },
        )
        .unwrap();

    let tool_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(vec![
        Arc::new(FakeMemory::new()) as Arc<dyn BaseTool>
    ]));
    graph.add_node("tool_node", tool_node).unwrap();

    graph.add_edge(START, "search_memories").unwrap();
    graph.add_edge("search_memories", "llm_call").unwrap();
    conditional_edges!(graph, "llm_call", tools_condition, "tools" => "tool_node", END => END)
        .unwrap();
    graph.add_edge("tool_node", "llm_call").unwrap();

    graph
        .compile_builder()
        .checkpointer(checkpointer)
        .build()
        .unwrap()
}

/// Each turn appends 4 messages (human, ai+tool_call, tool, final-ai) across 4
/// super-steps. `history = 4 * turns` messages, so `measure_channel_ops(4*turns)`
/// is the per-super-step `gather_input` get() cost the engine is paying at that
/// point, and `get() / engine_per_superstep` is the elimination's upper bound.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_gather_input_upper_bound() {
    println!("== isolated channel-op cost (per call, mimalloc/release) ==");
    println!("  msgs |  get() µs | checkpoint() µs");
    for n in [200usize, 400, 800, 1600] {
        let (get, cp) = measure_channel_ops(n, 200);
        println!("  {n:>5} | {:>8.2} | {:>13.2}", get, cp);
    }

    println!();
    println!("== engine cross-check: real ReAct graph, per-super-step ==");
    println!("  turns (hist msgs) | engine µs/superstep | gather_input get() µs | upper-bound %");
    for turns in [50usize, 100, 200] {
        let saver = Arc::new(InMemorySaver::new());
        let app = build_react_agent(saver.clone());
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-react"}),
        );

        let start = Instant::now();
        for t in 0..turns {
            let input = json!({"messages": [json!({
                "type": "human",
                "content": format!("第{t}条用户消息，你记得吗"),
            })]});
            let mut stream = app.astream(
                &input,
                &config,
                vec![StreamMode::Custom, StreamMode::Updates],
            );
            while let Some(_ev) = stream.next().await {}
        }
        let elapsed = start.elapsed();

        // 完整性断言：每轮 4 条消息，静默截断会在这里失败。
        let snapshot = app.get_state(&config).unwrap();
        let got = snapshot
            .values
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len());
        assert_eq!(
            got,
            Some(turns * 4),
            "turn loop was truncated: expected {} messages, got {got:?}",
            turns * 4
        );

        let per_ss = elapsed.as_secs_f64() * 1e6 / (turns as f64 * 4.0);
        let (get_us, _) = measure_channel_ops(turns * 4, 200);
        let pct = get_us / per_ss * 100.0;
        println!(
            "  {:>3} (hist {:>4}) | {:>18.1} | {:>20.1} | {:>12.1}%",
            turns,
            turns * 4,
            per_ss,
            get_us,
            pct
        );
    }
}
