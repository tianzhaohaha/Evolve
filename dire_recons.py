#!/usr/bin/env python3
"""Occupy specified idle NVIDIA GPUs with memory allocation and compute load."""

from __future__ import annotations

import argparse
import getpass
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

SCRIPT_NAME = os.path.basename(os.path.abspath(__file__))


@dataclass(frozen=True)
class GpuState:
    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int


def parse_gpu_list(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be a comma-separated list of GPU indices") from exc
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU index")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus contains duplicate GPU indices")
    return gpus


def parse_pattern_list(value: str) -> list[str]:
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    if not patterns:
        raise argparse.ArgumentTypeError("pattern list must contain at least one non-empty entry")
    return patterns


def run_nvidia_smi_query() -> dict[int, GpuState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"nvidia-smi query failed: {message}") from exc

    states: dict[int, GpuState] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, used, total, utilization = (int(part) for part in parts)
        states[index] = GpuState(
            index=index,
            memory_used_mib=used,
            memory_total_mib=total,
            utilization_percent=utilization,
        )
    return states


def get_other_user_scripts(patterns: list[str], user: str | None = None) -> list[str]:
    """Return command lines of the given user's processes matching any pattern.

    Excludes this script itself (and its child processes) so it never blocks itself.
    """
    if not user:
        user = getpass.getuser()
    try:
        result = subprocess.run(
            ["ps", "-u", user, "-o", "pid=,ppid=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    own_pid = os.getpid()
    own_pgid = os.getpgid(own_pid)
    matches: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, _ppid_str, cmdline = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        # Skip this script and its process group (child GPU workers).
        try:
            if pid == own_pid or os.getpgid(pid) == own_pgid:
                continue
        except (ProcessLookupError, PermissionError):
            continue
        if SCRIPT_NAME in cmdline:
            continue
        lowered = cmdline.lower()
        if any(pattern.lower() in lowered for pattern in patterns):
            matches.append(f"pid={pid} cmd={cmdline[:120]}")
    return matches


def is_gpu_idle(state: GpuState, utilization_threshold: int, used_memory_threshold_mib: int) -> bool:
    return (
        state.utilization_percent <= utilization_threshold
        and state.memory_used_mib <= used_memory_threshold_mib
    )


def wait_for_idle_gpu(
    gpu_index: int,
    utilization_threshold: int,
    used_memory_threshold_mib: int,
    check_interval_seconds: float,
    stop_event: mp.Event,
    user_process_patterns: list[str] | None = None,
    user: str | None = None,
) -> GpuState | None:
    while not stop_event.is_set():
        states = run_nvidia_smi_query()
        state = states.get(gpu_index)
        if state is None:
            raise RuntimeError(f"GPU {gpu_index} was not reported by nvidia-smi")
        if not is_gpu_idle(state, utilization_threshold, used_memory_threshold_mib):
            print(
                f"[gpu {gpu_index}] busy: util={state.utilization_percent}% "
                f"memory={state.memory_used_mib}/{state.memory_total_mib} MiB; waiting",
                flush=True,
            )
            stop_event.wait(check_interval_seconds)
            continue
        if user_process_patterns:
            running = get_other_user_scripts(user_process_patterns, user=user)
            if running:
                preview = "; ".join(running[:3])
                print(
                    f"[gpu {gpu_index}] idle, but user has {len(running)} other script(s) running "
                    f"({preview}); waiting",
                    flush=True,
                )
                stop_event.wait(check_interval_seconds)
                continue
        return state
    return None


def bytes_from_mib(value: int) -> int:
    return value * 1024 * 1024


def build_gpu_rng(args: argparse.Namespace, gpu_index: int) -> random.Random:
    seed = args.seed if args.seed is not None else int(time.time_ns() & 0xFFFFFFFF)
    return random.Random(seed + gpu_index * 1009)


def jittered_int(rng: random.Random, base_value: int, jitter_fraction: float, minimum: int = 1) -> int:
    if jitter_fraction <= 0:
        return max(base_value, minimum)
    low = 1.0 - jitter_fraction
    high = 1.0 + jitter_fraction
    return max(int(base_value * rng.uniform(low, high)), minimum)


def hold_gpu(args: argparse.Namespace, gpu_index: int, stop_event: mp.Event) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rng = build_gpu_rng(args, gpu_index)

    if args.startup_jitter > 0:
        stop_event.wait(rng.uniform(0, args.startup_jitter))
        if stop_event.is_set():
            return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to allocate and compute on GPU") from exc

    state = wait_for_idle_gpu(
        gpu_index=gpu_index,
        utilization_threshold=args.util_threshold,
        used_memory_threshold_mib=args.used_mem_threshold_mib,
        check_interval_seconds=args.check_interval,
        stop_event=stop_event,
        user_process_patterns=None if args.no_user_check else args.user_process_patterns,
        user=args.user,
    )
    if state is None:
        return

    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.empty_cache()

    free_mib = max(state.memory_total_mib - state.memory_used_mib - args.reserve_mib, 0)
    memory_fraction = args.memory_fraction
    if args.memory_jitter > 0:
        memory_fraction *= rng.uniform(1.0 - args.memory_jitter, 1.0 + args.memory_jitter)
        memory_fraction = min(max(memory_fraction, 0.01), 1.0)
    target_mib = int(free_mib * memory_fraction)
    if args.memory_mib is not None:
        target_mib = min(target_mib, args.memory_mib)
    if target_mib <= 0:
        print(f"[gpu {gpu_index}] no memory to allocate after reserve; skipping", flush=True)
        return

    print(
        f"[gpu {gpu_index}] idle; allocating about {target_mib} MiB "
        f"({memory_fraction:.3f} of free after reserve) and starting compute load",
        flush=True,
    )
    holder = torch.empty(bytes_from_mib(target_mib) // 2, dtype=torch.float16, device=device)
    holder.fill_(1.0)

    matrix_size = jittered_int(rng, args.matrix_size, args.matrix_jitter)
    matrix_size = max(128, matrix_size - matrix_size % 128)
    left = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
    right = torch.randn((matrix_size, matrix_size), dtype=torch.float16, device=device)
    deadline = None if args.duration <= 0 else time.monotonic() + args.duration

    iterations = 0
    while not stop_event.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            break
        left = torch.matmul(left, right)
        left = left / left.norm().clamp_min(1.0)
        iterations += 1
        sleep_seconds = args.sleep
        if args.sleep_jitter > 0:
            sleep_seconds += rng.uniform(0, args.sleep_jitter)
        if sleep_seconds > 0:
            torch.cuda.synchronize(device)
            stop_event.wait(sleep_seconds)
        if args.log_interval > 0 and iterations % args.log_interval == 0:
            print(
                f"[gpu {gpu_index}] still occupying; iterations={iterations}; "
                f"matrix_size={matrix_size}; allocated_mib={target_mib}",
                flush=True,
            )

    del holder, left, right
    torch.cuda.empty_cache()
    print(f"[gpu {gpu_index}] released", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait for specified GPUs to become idle, then occupy them with GPU memory and compute load.",
    )
    parser.add_argument("--gpus", required=True, type=parse_gpu_list, help="Comma-separated GPU indices, e.g. 0,1,3")
    parser.add_argument("--memory-fraction", type=float, default=0.88, help="Fraction of currently free memory to allocate")
    parser.add_argument("--memory-jitter", type=float, default=0.08, help="Per-GPU random variation applied to --memory-fraction")
    parser.add_argument("--memory-mib", type=int, default=None, help="Optional upper bound for allocated memory per GPU")
    parser.add_argument("--reserve-mib", type=int, default=512, help="Memory to leave free before applying --memory-fraction")
    parser.add_argument("--util-threshold", type=int, default=5, help="GPU utilization percent considered idle")
    parser.add_argument("--used-mem-threshold-mib", type=int, default=1024, help="Used memory threshold considered idle")
    parser.add_argument("--check-interval", type=float, default=120.0, help="Seconds between idle checks")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to occupy after allocation; 0 means until stopped")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix size used for compute load")
    parser.add_argument("--matrix-jitter", type=float, default=0.12, help="Per-GPU random variation applied to --matrix-size")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between compute iterations to reduce load")
    parser.add_argument("--sleep-jitter", type=float, default=0.0, help="Additional random per-iteration sleep upper bound in seconds")
    parser.add_argument("--startup-jitter", type=float, default=1.0, help="Random per-GPU startup delay upper bound in seconds")
    parser.add_argument("--log-interval", type=int, default=0, help="Log every N compute iterations; 0 disables progress logs")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible per-GPU variation")
    parser.add_argument(
        "--user-process-patterns",
        type=parse_pattern_list,
        default=["torchrun", "deepspeed", "main_ppo"],
        help="Comma-separated substrings; if any matches another process of the current user, occupation is postponed",
    )
    parser.add_argument(
        "--no-user-check",
        action="store_true",
        help="Disable checking whether the current user has other scripts running",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="jcgu",
        help="Username whose running scripts are checked before occupying GPUs",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("--memory-fraction must be in the range (0, 1]")
    if not 0 <= args.memory_jitter < 1:
        raise ValueError("--memory-jitter must be in the range [0, 1)")
    if args.memory_mib is not None and args.memory_mib <= 0:
        raise ValueError("--memory-mib must be positive")
    if args.reserve_mib < 0:
        raise ValueError("--reserve-mib must be non-negative")
    if args.util_threshold < 0:
        raise ValueError("--util-threshold must be non-negative")
    if args.used_mem_threshold_mib < 0:
        raise ValueError("--used-mem-threshold-mib must be non-negative")
    if args.check_interval <= 0:
        raise ValueError("--check-interval must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be non-negative")
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be positive")
    if args.matrix_jitter < 0:
        raise ValueError("--matrix-jitter must be non-negative")
    if args.sleep < 0:
        raise ValueError("--sleep must be non-negative")
    if args.sleep_jitter < 0:
        raise ValueError("--sleep-jitter must be non-negative")
    if args.startup_jitter < 0:
        raise ValueError("--startup-jitter must be non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        states = run_nvidia_smi_query()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    missing = [gpu for gpu in args.gpus if gpu not in states]
    if missing:
        print(f"error: GPU index not found: {missing}", file=sys.stderr)
        return 2

    stop_event = mp.Event()

    def request_stop(signum: int, _frame: object) -> None:
        print(f"received signal {signum}; releasing GPUs", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    processes = [mp.Process(target=hold_gpu, args=(args, gpu, stop_event), daemon=False) for gpu in args.gpus]
    for process in processes:
        process.start()

    exit_code = 0
    try:
        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                exit_code = process.exitcode or 1
    finally:
        stop_event.set()
        for process in processes:
            if process.is_alive():
                process.join(timeout=5)
            if process.is_alive():
                process.terminate()
    return exit_code


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())



# nohup python dire_recons.py --gpus 5,6,7 > dire.log 2>&1 &