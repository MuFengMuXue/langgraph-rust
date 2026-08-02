//! Benchmark: end-to-end Pregel loop hot paths.
//!
//! Run explicitly (it is `#[ignore]`d so normal `cargo test` runs stay fast):
//!
//! ```text
//! cargo test --release --test bench_pregel -- --ignored --nocapture
//! ```

use langgraph::channels::{BinaryOperatorAggregate, Channel, LastValue};
use langgraph::checkpoint::{BaseCheckpointSaver, InMemorySaver};
use langgraph::prelude::*;
use langgraph_checkpoint_sqlite::SqliteSaver;
use langgraph_prebuilt::add_messages_ref;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn make_message(i: usize) -> JsonValue {
    serde_json::json!({
        "type": "ai",
        "content": format!("Assistant reply number {i}: {}", "x".repeat(300)),
        "tool_calls": [{
            "name": "search_tool",
            "args": {
                "query": format!("query-{i}"),
                "filters": {"category": "test", "limit": 10, "extra": "data"}
            },
            "id": format!("call_{i}")
        }],
        "id": format!("msg_{i}")
    })
}

/// Single-node graph: `messages` accumulates with the `add_messages` reducer.
/// Every invoke appends one message and saves a fresh checkpoint, so the
/// checkpointed state grows by one message per super-step.
fn build_linear_graph(checkpointer: Arc<dyn BaseCheckpointSaver>) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref))
            as Box<dyn Channel>,
    );

    let mut graph = StateGraph::new(channels);
    graph
        .add_node(
            "append",
            |input: JsonValue, _config: RunnableConfig| async move {
                let n = input
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                Ok(json!({"messages": [make_message(n)]}))
            },
        )
        .unwrap();
    graph.add_edge(START, "append").unwrap();
    graph.add_edge("append", END).unwrap();

    graph
        .compile_builder()
        .checkpointer(checkpointer)
        .build()
        .unwrap()
}

/// Per-step cost of a checkpointed run as the message history grows.
///
/// If checkpointing re-serialized the full state every step this shows clear
/// super-linear growth (total work ~ O(steps^2)).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_linear_checkpoint_growth() {
    for steps in [100usize, 200, 400, 800] {
        let app = build_linear_graph(Arc::new(InMemorySaver::new()));
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-linear"}),
        );

        let start = Instant::now();
        for i in 0..steps {
            let input = json!({"messages": [make_message(i)]});
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "linear checkpointed (InMemory): {steps:>4} steps, history->{steps} => {elapsed:?}  ({:?}/step)",
            elapsed / steps as u32
        );
    }
}

/// Same growth benchmark against the SQLite saver — the persistent path where
/// incremental blob writes matter most.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_linear_checkpoint_sqlite() {
    for steps in [100usize, 200, 400, 800] {
        let saver = SqliteSaver::from_conn_string("sqlite::memory:")
            .await
            .unwrap();
        saver.setup().await.unwrap();
        let app = build_linear_graph(Arc::new(saver));
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-sqlite"}),
        );

        let start = Instant::now();
        for i in 0..steps {
            let input = json!({"messages": [make_message(i)]});
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "linear checkpointed (SQLite): {steps:>4} steps, history->{steps} => {elapsed:?}  ({:?}/step)",
            elapsed / steps as u32
        );
    }
}

/// Wall-clock time to run `branches` parallel nodes in a single super-step.
///
/// Every branch sleeps for a fixed time. A serial runner takes
/// `branches * sleep`; a parallel runner takes ~`sleep`.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn bench_parallel_fanout() {
    const BRANCH_SLEEP_MS: u64 = 40;

    for branches in [2usize, 4, 8] {
        let channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        let mut graph = StateGraph::new(channels);
        for i in 0..branches {
            graph
                .add_node(
                    format!("branch{i}"),
                    |_input: JsonValue, _config: RunnableConfig| async move {
                        tokio::time::sleep(Duration::from_millis(BRANCH_SLEEP_MS)).await;
                        Ok(json!({}))
                    },
                )
                .unwrap();
            graph.add_edge(START, format!("branch{i}")).unwrap();
        }
        let app = graph.compile().unwrap();

        let start = Instant::now();
        app.ainvoke(&json!({}), &RunnableConfig::new())
            .await
            .unwrap();
        let elapsed = start.elapsed();

        println!(
            "parallel fan-out: {branches} branches x {BRANCH_SLEEP_MS}ms => {elapsed:?}  (serial {BRANCH_SLEEP_MS}ms*n, parallel ~{BRANCH_SLEEP_MS}ms)"
        );
    }
}

/// The win case for incremental writes: a large static channel (e.g. embedded
/// knowledge base) written once at thread start, then untouched while
/// `messages` grows every step. Without delta writes the static channel's
/// blob gets re-encoded and re-inserted every single step.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_sqlite_static_context() {
    const CONTEXT_SIZE: usize = 200 * 1024;

    for steps in [200usize, 400, 800] {
        let saver = SqliteSaver::from_conn_string("sqlite::memory:")
            .await
            .unwrap();
        saver.setup().await.unwrap();

        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert(
            "messages".to_string(),
            Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref))
                as Box<dyn Channel>,
        );
        channels.insert(
            "context".to_string(),
            Box::new(LastValue::new("context")) as Box<dyn Channel>,
        );

        let mut graph = StateGraph::new(channels);
        graph
            .add_node(
                "append",
                |input: JsonValue, _config: RunnableConfig| async move {
                    let n = input
                        .get("messages")
                        .and_then(|m| m.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    Ok(json!({"messages": [make_message(n)]}))
                },
            )
            .unwrap();
        graph.add_edge(START, "append").unwrap();
        graph.add_edge("append", END).unwrap();
        let app = graph
            .compile_builder()
            .checkpointer(Arc::new(saver))
            .build()
            .unwrap();

        let context = "x".repeat(CONTEXT_SIZE);
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-ctx"}),
        );

        // Seed the static channel once, then grow messages every step.
        app.ainvoke(&json!({"messages": [make_message(0)], "context": context}), &config)
            .await
            .unwrap();

        let start = Instant::now();
        for i in 1..steps {
            let input = json!({"messages": [make_message(i)]});
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "sqlite static context: {steps:>4} steps, {CONTEXT_SIZE}-byte static channel => {elapsed:?}  ({:?}/step)",
            elapsed / (steps - 1) as u32
        );
    }
}

/// Guards against the benchmark graphs silently becoming no-ops.
#[tokio::test]
async fn sanity_bench_graphs_work() {
    let app = build_linear_graph(Arc::new(InMemorySaver::new()));
    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        json!({"thread_id": "bench-sanity"}),
    );
    app.ainvoke(&json!({"messages": [make_message(0)]}), &config)
        .await
        .unwrap();
    app.ainvoke(&json!({"messages": [make_message(1)]}), &config)
        .await
        .unwrap();
    let snapshot = app.get_state(&config).unwrap();
    assert_eq!(
        snapshot.values.get("messages").and_then(|m| m.as_array()).map(|a| a.len()),
        Some(4)
    );
}
