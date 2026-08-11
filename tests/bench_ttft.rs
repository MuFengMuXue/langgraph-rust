//! Probe: framework TTFT (time-to-first-token) floor for an AI VTuber brain.
//!
//! The brain = user input → langgraph-rust `astream` → LLM node (`stream_llm`)
//! → first token streamed to the consumer. Real TTFT is dominated by the
//! model's own first-token latency (network + server, hundreds of ms to
//! seconds). This bench measures the FRAMEWORK's contribution with an INSTANT
//! stub model (zero model latency), so the number is the pure engine floor.
//! Real TTFT ≈ this floor + the model's actual first-token latency.
//!
//! Measured from just before `astream()` to the first yielded stream item
//! (StreamMode::Custom — the stub node emits one token via StreamWriter before
//! its super-step completes, so the first item is the token). Two configs:
//! with InMemorySaver (realistic session memory) and without (pure engine).
//! 0/50/200 turns of preloaded history show how the floor grows with the
//! O(history) state-handling (gather_input clone + checkpoint load).
//!
//! Run:
//! ```text
//! cargo test --release --test bench_ttft -- --ignored --nocapture
//! ```

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use langgraph::channels::{BinaryOperatorAggregate, Channel};
use langgraph::checkpoint::InMemorySaver;
use langgraph::prebuilt::{
    add_messages_ref, stream_llm, BaseChatModel, Message, MessageStream, ModelError, ToolDef,
};
use langgraph::prelude::*;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_stream::StreamExt;

/// Instant stub model: yields two tokens with zero latency.
#[derive(Clone)]
struct StubModel;

impl BaseChatModel for StubModel {
    fn name(&self) -> &str {
        "stub"
    }

    fn invoke(
        &self,
        _messages: &[Message],
        _config: &RunnableConfig,
    ) -> Result<Message, ModelError> {
        Ok(Message::ai("你好"))
    }

    fn astream<'a>(
        &'a self,
        _messages: &'a [Message],
        _config: &'a RunnableConfig,
    ) -> MessageStream<'a> {
        Box::pin(tokio_stream::iter([
            Ok(Message::ai("你")),
            Ok(Message::ai("你好")),
        ]))
    }

    fn bind_tools(&self, _tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        Box::new(self.clone())
    }
}

fn build_vtuber_brain(checkpointer: Option<Arc<InMemorySaver>>) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref)) as Box<dyn Channel>,
    );

    let model = Arc::new(StubModel);
    let mut graph = StateGraph::new(channels);
    graph
        .add_node("llm", move |input: JsonValue, _config: RunnableConfig| {
            let model = model.clone();
            async move { stream_llm(model.as_ref(), &input, "你是虚拟主播。").await }
        })
        .unwrap();
    graph.add_edge(START, "llm").unwrap();
    graph.add_edge("llm", END).unwrap();

    let builder = graph.compile_builder();
    match checkpointer {
        Some(saver) => builder.checkpointer(saver).build().unwrap(),
        None => builder.build().unwrap(),
    }
}

/// Preload `turns` rounds of history, then measure time to the first yielded
/// item on the next turn. Returns (ttft, total, items).
async fn measure_ttft(
    app: &CompiledStateGraph,
    config: &RunnableConfig,
    turns: usize,
) -> (Duration, Duration, usize) {
    if turns > 0 {
        for t in 0..turns {
            let input =
                json!({"messages": [json!({"type": "human", "content": format!("第{t}条")})]});
            let mut s = app.astream(
                &input,
                config,
                vec![StreamMode::Custom, StreamMode::Updates],
            );
            while let Some(_ev) = s.next().await {}
        }
    }

    let input = json!({"messages": [json!({"type": "human", "content": "你好，我是观众"})]});
    let t0 = Instant::now();
    let mut s = app.astream(
        &input,
        config,
        vec![StreamMode::Custom, StreamMode::Updates],
    );
    let mut ttft = None;
    let mut items = 0usize;
    while let Some(_ev) = s.next().await {
        items += 1;
        if ttft.is_none() {
            ttft = Some(t0.elapsed());
        }
    }
    let total = t0.elapsed();
    (ttft.unwrap(), total, items)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_ttft_floor() {
    println!("== framework TTFT floor (instant stub model, mimalloc/release) ==");
    for turns in [0usize, 50, 200] {
        for (label, saver) in [
            ("with checkpointer", Some(Arc::new(InMemorySaver::new()))),
            ("pure engine", None),
        ] {
            let app = build_vtuber_brain(saver);
            let mut config = RunnableConfig::new();
            config.insert("configurable".to_string(), json!({"thread_id": "vtuber"}));
            let (ttft, total, items) = measure_ttft(&app, &config, turns).await;
            println!(
                "{:>15} @ {:>3} turns: TTFT(first item) = {:>8.1}µs, total = {:>8.1}µs, {} items",
                label,
                turns,
                ttft.as_secs_f64() * 1e6,
                total.as_secs_f64() * 1e6,
                items
            );
        }
    }
}
