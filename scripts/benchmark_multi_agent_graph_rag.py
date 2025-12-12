#!/usr/bin/env python3
"""Simulate multi-agent & graph-aware RAG performance trade-offs.

This script generates synthetic latency/token curves for different numbers of
agents and knowledge-graph scales so that we can reason about routing and
resource allocation *before* the full multi-agent orchestration is wired into
production.  The output can be consumed by notebooks or dashboards to guide
design choices.

Example
-------
python scripts/benchmark_multi_agent_graph_rag.py \
    --agent-counts 1 2 3 4 \
    --graph-sizes 200 500 1000 2000 \
    --iterations 10 \
    --output output/rag_multi_agent_simulation.json

The simulation is intentionally lightweight: it does not require a running
ChromaDB instance or LLM API access.  Instead, it applies empirically-inspired
scaling laws (logarithmic retrieval latency, quasi-linear graph traversal
costs, etc.) and injects a small amount of noise to emulate production
variance.  These formulas can be refined as we collect real telemetry once the
multi-agent pipeline is live.
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


# Rough 2025 token pricing for reference (USD per 1K tokens)
PRICING_TABLE = {
    "gpt-5": {"prompt": 0.003, "completion": 0.015},
    "gpt-4.1": {"prompt": 0.0025, "completion": 0.010},
    "llama-4": {"prompt": 0.0006, "completion": 0.0006},
}


@dataclass
class SimulationConfig:
    """Configuration for a single sweep run."""

    agent_count: int
    graph_nodes: int
    iterations: int
    parallel_agents: int
    base_retrieval_ms: float
    base_graph_ms: float
    base_synthesis_ms: float
    base_prompt_tokens: int
    base_completion_tokens: int
    target_model: str


@dataclass
class IterationResult:
    """Raw measurements from one synthetic inference run."""

    latency_ms: float
    prompt_tokens: float
    completion_tokens: float
    agent_cycles: int
    graph_edges_traversed: float


@dataclass
class AggregateResult:
    """Aggregated metrics for plotting."""

    agent_count: int
    graph_nodes: int
    iterations: int
    avg_latency_ms: float
    p95_latency_ms: float
    std_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    estimated_cost_usd: float
    avg_agent_cycles: float
    avg_graph_edges_traversed: float


def _sample_latency(base_ms: float, variability: float = 0.08) -> float:
    """Return a positive latency sample with multiplicative noise."""

    noisy_ms = base_ms * (1.0 + variability * (2.0 * (math.sin(time.time()) % 1) - 0.5))
    return max(noisy_ms, 0.0)


def run_iteration(config: SimulationConfig) -> IterationResult:
    """Simulate a single multi-agent, graph-aware RAG pass."""

    agents = config.agent_count
    graph_nodes = max(config.graph_nodes, 1)

    retrieval_ms = (
        config.base_retrieval_ms
        * math.log2(graph_nodes + 1)
        * (1 + 0.07 * (agents - 1))
        / max(config.parallel_agents, 1)
    )

    traversal_edges = graph_nodes * min(agents, 4) * 1.8
    graph_ms = config.base_graph_ms * (traversal_edges / 300.0)

    synthesis_ms = config.base_synthesis_ms * (1 + 0.12 * (agents - 1))

    total_latency_ms = (
        _sample_latency(retrieval_ms)
        + _sample_latency(graph_ms)
        + _sample_latency(synthesis_ms)
    )

    prompt_tokens = (
        config.base_prompt_tokens
        * (1 + graph_nodes / 1200.0)
        * (1 + 0.05 * (agents - 1))
    )
    completion_tokens = config.base_completion_tokens * (1 + 0.04 * (agents - 1))

    return IterationResult(
        latency_ms=total_latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        agent_cycles=agents,
        graph_edges_traversed=traversal_edges,
    )


def aggregate_results(
    agent_count: int,
    graph_nodes: int,
    results: List[IterationResult],
    model: str,
) -> AggregateResult:
    latencies = [r.latency_ms for r in results]
    prompt_tokens = [r.prompt_tokens for r in results]
    completion_tokens = [r.completion_tokens for r in results]
    agent_cycles = [r.agent_cycles for r in results]
    graph_edges = [r.graph_edges_traversed for r in results]

    prompt_mean = statistics.mean(prompt_tokens)
    completion_mean = statistics.mean(completion_tokens)

    pricing = PRICING_TABLE.get(model, PRICING_TABLE["gpt-5"])
    cost = (
        (prompt_mean / 1000.0) * pricing["prompt"]
        + (completion_mean / 1000.0) * pricing["completion"]
    )

    p95_latency = (
        statistics.quantiles(latencies, n=20)[-1]
        if len(latencies) > 1
        else latencies[0]
    )

    return AggregateResult(
        agent_count=agent_count,
        graph_nodes=graph_nodes,
        iterations=len(results),
        avg_latency_ms=statistics.mean(latencies),
        p95_latency_ms=p95_latency,
        std_latency_ms=statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        avg_prompt_tokens=prompt_mean,
        avg_completion_tokens=completion_mean,
        estimated_cost_usd=cost,
        avg_agent_cycles=statistics.mean(agent_cycles),
        avg_graph_edges_traversed=statistics.mean(graph_edges),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate multi-agent graph-aware RAG performance curves"
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of agent counts to simulate",
    )
    parser.add_argument(
        "--graph-sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000],
        help="Approximate number of graph nodes explored per query",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of Monte Carlo iterations per configuration",
    )
    parser.add_argument(
        "--parallel-agents",
        type=int,
        default=2,
        help="Effective number of agents that can run concurrently",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        choices=list(PRICING_TABLE.keys()),
        help="Primary model assumed for cost estimation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/rag_multi_agent_simulation.json"),
        help="Path to write aggregated JSON results",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write CSV results",
    )
    return parser.parse_args()


def build_config(agent_count: int, graph_nodes: int, args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        agent_count=agent_count,
        graph_nodes=graph_nodes,
        iterations=args.iterations,
        parallel_agents=args.parallel_agents,
        base_retrieval_ms=120.0,
        base_graph_ms=45.0,
        base_synthesis_ms=180.0,
        base_prompt_tokens=1400,
        base_completion_tokens=380,
        target_model=args.model,
    )


def write_csv(results: List[AggregateResult], path: Path) -> None:
    header = (
        "agent_count,graph_nodes,iterations,avg_latency_ms,p95_latency_ms,std_latency_ms,"
        "avg_prompt_tokens,avg_completion_tokens,estimated_cost_usd,avg_agent_cycles,avg_graph_edges_traversed\n"
    )
    lines = [header]
    for result in results:
        lines.append(
            f"{result.agent_count},{result.graph_nodes},{result.iterations},"
            f"{result.avg_latency_ms:.2f},{result.p95_latency_ms:.2f},{result.std_latency_ms:.2f},"
            f"{result.avg_prompt_tokens:.1f},{result.avg_completion_tokens:.1f},"
            f"{result.estimated_cost_usd:.4f},{result.avg_agent_cycles:.2f},{result.avg_graph_edges_traversed:.1f}\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    aggregated: List[AggregateResult] = []

    for agent_count in args.agent_counts:
        for graph_nodes in args.graph_sizes:
            config = build_config(agent_count, graph_nodes, args)
            iteration_results: List[IterationResult] = []

            for _ in range(config.iterations):
                iteration_results.append(run_iteration(config))

            aggregate = aggregate_results(
                agent_count,
                graph_nodes,
                iteration_results,
                config.target_model,
            )
            aggregated.append(aggregate)

            print(
                f"Agents={agent_count:>2} | GraphNodes={graph_nodes:>5} | "
                f"Latency≈{aggregate.avg_latency_ms:7.2f} ms | "
                f"Tokens≈{aggregate.avg_prompt_tokens + aggregate.avg_completion_tokens:8.1f} | "
                f"Cost≈${aggregate.estimated_cost_usd:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, List[Dict[str, float]]] = {
        "metadata": {
            "agent_counts": args.agent_counts,
            "graph_sizes": args.graph_sizes,
            "iterations": args.iterations,
            "model": args.model,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": [asdict(res) for res in aggregated],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.csv is not None:
        write_csv(aggregated, args.csv)


if __name__ == "__main__":
    main()

