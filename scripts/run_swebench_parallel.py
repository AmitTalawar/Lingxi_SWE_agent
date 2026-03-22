#!/usr/bin/env python3
"""Run multiple SWE-bench instances in parallel against the local graph directly.

This runner intentionally uses process-level parallelism to isolate each run's
runtime state and reduce cross-example interference at the application layer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.constant import RUNTIME_DIR


@dataclass
class RunResult:
    instance_id: str
    status: str
    elapsed_seconds: float
    output_file: str
    error: str | None = None


def _load_instances_from_dataset(max_examples: int) -> list[str]:
    from datasets import load_dataset

    if max_examples < 1:
        raise ValueError("--max-examples must be >= 1")

    swe_instances = load_dataset(
        "princeton-nlp/SWE-bench_Verified", split="test", cache_dir=RUNTIME_DIR
    )

    instances: list[str] = []
    for row in swe_instances:
        instance_id = str(row["instance_id"])
        instances.append(instance_id)
        if len(instances) >= max_examples:
            break

    if not instances:
        raise ValueError("SWE-bench dataset returned no instances.")

    return instances


def _worker_run_instance(instance_id: str) -> dict[str, Any]:
    """Process worker: runs one SWE-bench instance end-to-end."""
    start = time.time()

    try:
        from langchain_core.messages import HumanMessage
        from src.workflow.swebench_workflow import swebench_resolve_graph

        thread_id = f"bench-{instance_id}-{uuid.uuid4().hex[:8]}"
        run_id = str(uuid.uuid4())

        initial_input = {
            "messages": [HumanMessage(content=instance_id)],
            "preset": instance_id,
            "human_in_the_loop": False,
        }
        run_config = {
            "recursion_limit": 100,
            "run_id": uuid.UUID(run_id),
            "tags": ["swebench", "parallel-benchmark"],
            "configurable": {"thread_id": thread_id},
        }

        events: list[dict[str, Any]] = []
        final_message: str | None = None

        # stream() is synchronous here, and that is fine because each run is process-isolated.
        for chunk in swebench_resolve_graph.stream(
            initial_input, config=run_config, stream_mode="values"
        ):
            msg = None
            if isinstance(chunk, dict) and chunk.get("messages"):
                msg = chunk["messages"][-1]

            if msg is not None:
                content = getattr(msg, "content", None)
                events.append(
                    {
                        "name": getattr(msg, "name", None),
                        "type": getattr(msg, "type", None),
                        "content": content,
                    }
                )
                final_message = str(content)

        elapsed = time.time() - start
        return {
            "instance_id": instance_id,
            "status": "success",
            "elapsed_seconds": elapsed,
            "run_id": run_id,
            "thread_id": thread_id,
            "final_message": final_message,
            "events": events,
            "error": None,
        }
    except Exception:
        elapsed = time.time() - start
        return {
            "instance_id": instance_id,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "run_id": None,
            "thread_id": None,
            "final_message": None,
            "events": [],
            "error": traceback.format_exc(),
        }


def _write_result(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_file = output_dir / f"{payload['instance_id']}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple SWE-bench instances in parallel via swebench_resolve_graph."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of worker processes to run in parallel.",
    )
    parser.add_argument("--max-examples", type=int, default=10, help="Number of SWE-bench examples to run from the dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    instances = _load_instances_from_dataset(args.max_examples)

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path("runs/swebench_parallel") / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(instances)} instances with {args.max_workers} workers")
    print(f"Output dir: {output_dir}")

    summary: list[RunResult] = []

    # Use spawn to avoid inheriting parent process state (singleton/runtime side effects).
    mp_ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers, mp_context=mp_ctx
    ) as executor:
        future_map = {
            executor.submit(_worker_run_instance, instance): instance
            for instance in instances
        }

        for future in concurrent.futures.as_completed(future_map):
            instance_id = future_map[future]
            try:
                payload = future.result()
            except Exception:
                payload = {
                    "instance_id": instance_id,
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "run_id": None,
                    "thread_id": None,
                    "final_message": None,
                    "events": [],
                    "error": traceback.format_exc(),
                }

            out_file = _write_result(output_dir, payload)
            summary.append(
                RunResult(
                    instance_id=payload["instance_id"],
                    status=payload["status"],
                    elapsed_seconds=float(payload["elapsed_seconds"]),
                    output_file=str(out_file),
                    error=payload.get("error"),
                )
            )

            print(
                f"[{payload['status']}] {payload['instance_id']} "
                f"({payload['elapsed_seconds']:.1f}s) -> {out_file}"
            )

    total = len(summary)
    succeeded = sum(1 for item in summary if item.status == "success")
    failed = total - succeeded

    summary_payload = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "results": [item.__dict__ for item in summary],
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=True)

    print(f"Done. success={succeeded}, failed={failed}")
    print(f"Summary: {summary_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
