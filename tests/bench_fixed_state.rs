//! Probe bench: per-step FIXED checkpoint cost with constant-size state.
//!
//! The growth benches in `bench_pregel.rs` dominate their per-step cost with
//! the inherent serialization of a *growing* history, so they cannot isolate
//! the fixed per-step overhead (checkpoint load + save + loop). This bench
//! keeps state size constant — `payload` is overwritten each step with a
//! same-size value — so every step pays the same fixed cost. Running at a few
//! payload sizes splits the fixed overhead from the state-proportional
//! serialization component.
//!
//! Run explicitly (it is `#[ignore]`d):
//!
//! ```text
//! cargo test --release --test bench_fixed_state -- --ignored --nocapture
//! ```

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use langgraph::channels::{Channel, LastValue};
use langgraph::checkpoint::{BaseCheckpointSaver, InMemorySaver};
use langgraph::prelude::*;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Fixed-size state graph. `payload` is a constant-size LastValue overwritten
/// with the same-size value each invoke; `n` is a counter so every step's
/// checkpoint actually differs. The node only bumps `n` — it never touches
/// `payload`, so the serialized state stays `state_size` bytes every step.
fn build_fixed_state_graph(
    checkpointer: Option<Arc<dyn BaseCheckpointSaver>>,
) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "payload".to_string(),
        Box::new(LastValue::new("payload")) as Box<dyn Channel>,
    );
    channels.insert(
        "n".to_string(),
        Box::new(LastValue::new("n")) as Box<dyn Channel>,
    );

    let mut graph = StateGraph::new(channels);
    graph
        .add_node(
            "echo",
            |input: JsonValue, _config: RunnableConfig| async move {
                let n = input.get("n").and_then(|c| c.as_i64()).unwrap_or(0);
                Ok(json!({"n": n + 1}))
            },
        )
        .unwrap();
    graph.add_edge(START, "echo").unwrap();
    graph.add_edge("echo", END).unwrap();

    match checkpointer {
        Some(cp) => graph.compile_builder().checkpointer(cp).build().unwrap(),
        None => graph.compile().unwrap(),
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_fixed_state_checkpoint() {
    for state_size in [1024usize, 16_384, 131_072] {
        for steps in [100usize, 200] {
            let saver = Arc::new(InMemorySaver::new());
            let app = build_fixed_state_graph(Some(saver.clone()));
            let mut config = RunnableConfig::new();
            config.insert(
                "configurable".to_string(),
                json!({"thread_id": "bench-fixed"}),
            );

            let mut input = json!({"payload": "x".repeat(state_size), "n": 0});
            let start = Instant::now();
            for i in 0..steps {
                input["n"] = json!(i);
                app.ainvoke(&input, &config).await.unwrap();
            }
            let elapsed = start.elapsed();
            println!(
                "fixed-state checkpoint: state={state_size:>7}B, {steps:>3} steps => {elapsed:?}  ({:?}/step)",
                elapsed / steps as u32
            );
        }
    }

    // Reference: same fixed-state graph WITHOUT a checkpointer, so the
    // engine-only per-invoke overhead is visible and the checkpoint delta
    // can be read off directly.
    for state_size in [16_384usize] {
        let app = build_fixed_state_graph(None);
        let config = RunnableConfig::new();
        let mut input = json!({"payload": "x".repeat(state_size), "n": 0});
        let steps = 200usize;
        let start = Instant::now();
        for i in 0..steps {
            input["n"] = json!(i);
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "fixed-state no-checkpoint: state={state_size:>7}B, {steps:>3} steps => {elapsed:?}  ({:?}/step)",
            elapsed / steps as u32
        );
    }
}
