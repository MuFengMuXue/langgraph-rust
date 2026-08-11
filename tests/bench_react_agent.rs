//! Benchmark: sakura-rust-style ReAct agent latency — framework overhead only.
//!
//! Replicates the exact graph topology of `F:/sakura-rust/src/main.rs`: a
//! ReAct agent with a memory-search prelude node. State is `messages`
//! (add_messages) + `search_context`; nodes are `search_memories` →
//! `llm_call` → (`tools_condition` routes to `tool_node` or END) →
//! `llm_call`; checkpointer is `InMemorySaver`; the turn is streamed via
//! `astream([Custom, Updates])` exactly like the real app. The LLM and the
//! HTTP memory service are stubbed with instant fakes.
//!
//! Real-world latency is dominated by the LLM/HTTP calls (hundreds of ms+);
//! this bench isolates the framework's own per-turn cost for the same shape.
//! Each turn runs 4 super-steps and appends 4 messages (human, ai+tool_call,
//! tool result, final ai), so per-turn cost also shows how an InMemorySaver
//! conversation degrades as history grows.
//!
//! Run:
//! ```text
//! cargo test --release --test bench_react_agent -- --ignored --nocapture
//! ```

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use langgraph::channels::{BinaryOperatorAggregate, Channel, LastValue};
use langgraph::checkpoint::InMemorySaver;
use langgraph::prebuilt::{
    add_messages_ref, tools_condition, BaseTool, Message, ToolCall, ToolNode,
};
use langgraph::prelude::*;
use langgraph::tool;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::StreamExt;

#[tool("fake_memory", "桩工具：模拟记忆搜索/保存的即时返回")]
async fn fake_memory(query: String) -> Result<String, String> {
    Ok(format!("桩结果：{query}"))
}

fn build_react_agent(checkpointer: Arc<InMemorySaver>) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref)) as Box<dyn Channel>,
    );
    channels.insert(
        "search_context".to_string(),
        Box::new(LastValue::new("search_context")) as Box<dyn Channel>,
    );

    let mut graph = StateGraph::new(channels);

    // 记忆预检：桩，即时返回一段固定大小的记忆上下文（真实版是 HTTP 调用）。
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

    // LLM：桩。工具结果回来后给最终回答，否则发一个 tool_call。
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

/// Each turn: human msg → search_memories → llm_call (tool_call) → tool_node
/// → llm_call (final answer) = 4 super-steps, 4 messages appended.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_react_agent_turn() {
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

        println!(
            "react-agent turn: {turns:>3} turns, history->{turns} => {elapsed:?}  ({:?}/turn, {:?}/superstep)",
            elapsed / turns as u32,
            elapsed / (turns as u32 * 4)
        );
    }
}
