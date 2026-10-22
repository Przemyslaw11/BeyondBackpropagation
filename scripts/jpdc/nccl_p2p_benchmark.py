#!/usr/bin/env python3
"""Short NCCL point-to-point microbenchmark for the W2 latency floor.

Two ranks on two GPUs exchange one 1 MB tensor in a ping-pong loop.  This is a
diagnostic, not a W2 confirmatory endpoint: it measures the interconnect's
message latency and achieved bandwidth for the tensor size used by one cut
activation, so the link-rate model can separate "bytes dominate" from "latency
floor dominates" at 10 Gbit/s.

Run with torchrun, not directly:

    torchrun --standalone --nproc_per_node=2 scripts/jpdc/nccl_p2p_benchmark.py \
        --output jpdc_data/g1/w2_nccl/nccl_p2p_1mb.json

The job is deliberately short (a few seconds of GPU time) and is prepared in
``scripts/slurm_scripts/w2_nccl_p2p.slurm``; it is not submitted by the W1
submission helper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=Path("jpdc_data/g1/w2_nccl/nccl_p2p_1mb.json")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the NCCL microbenchmark.")
    if "LOCAL_RANK" not in os.environ:
        raise SystemExit("Run with torchrun so LOCAL_RANK/RANK/WORLD_SIZE are set.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise SystemExit(f"This benchmark expects exactly 2 ranks, got {world_size}.")

    dtype_bytes = 4  # float32
    if args.tensor_bytes % dtype_bytes != 0:
        raise SystemExit("--tensor-bytes must be divisible by 4.")
    n_elements = args.tensor_bytes // dtype_bytes
    tensor = torch.ones(n_elements, dtype=torch.float32, device="cuda")

    # Warm up both directions and the NCCL communicator.
    for _ in range(args.warmup):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        else:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=1)
    torch.cuda.synchronize()
    dist.barrier()

    # Host wall clock around explicit device synchronisation is intentionally
    # used instead of CUDA events: NCCL send/recv may run on an internal stream,
    # so events recorded on the default stream need not bracket the transfer.
    dist.barrier()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(args.iterations):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        else:
            dist.recv(tensor, src=0)
            dist.send(tensor, dst=1)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    dist.barrier()

    # Every iteration moves one tensor in each direction; rank 0's timing is
    # the ping-pong round-trip.  Report both one-way message latency and the
    # bidirectional achieved rate.
    round_trip_ms = elapsed_ms / args.iterations
    one_way_ms = round_trip_ms / 2.0
    one_way_gbps = (args.tensor_bytes * args.iterations) / (elapsed_ms / 1000.0) / 1e9
    round_trip_gbps = (2.0 * args.tensor_bytes * args.iterations) / (
        elapsed_ms / 1000.0
    ) / 1e9

    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": "nccl",
        "world_size": world_size,
        "device": torch.cuda.get_device_name(local_rank),
        "tensor_bytes": args.tensor_bytes,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "timing_method": "host wall clock around torch.cuda.synchronize()",
        "elapsed_ms": elapsed_ms,
        "round_trip_ms": round_trip_ms,
        "one_way_latency_us": one_way_ms * 1000.0,
        "one_way_bandwidth_gbps": one_way_gbps,
        "bidirectional_bandwidth_gbps": round_trip_gbps,
        "rank": rank,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"NCCL P2P {args.tensor_bytes} B: "
            f"one-way {record['one_way_latency_us']:.1f} us, "
            f"{one_way_gbps:.2f} Gbit/s one-way, "
            f"{round_trip_gbps:.2f} Gbit/s bidirectional"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
