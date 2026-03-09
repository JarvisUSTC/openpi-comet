"""
Run BEHAVIOR-1K (OmniGibson) evaluation for the 50 challenge tasks × 10 eval instances,
first with WM injected into the prompt, then without WM-in-prompt.

This script orchestrates two processes:
  1) OpenPi websocket policy server (`scripts/serve_b1k.py`) which can toggle `wm_in_prompt`.
  2) OmniGibson evaluator (`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py`) which runs instances.

It is intentionally "thin": it does not import OmniGibson; it just launches subprocesses.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def _now_run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_task_list(task_mapping_path: Path) -> list[str]:
    mapping = json.loads(task_mapping_path.read_text())
    # Prefer ordering by task_index when present.
    items = []
    for name, entry in mapping.items():
        idx = entry.get("task_index")
        items.append((idx if isinstance(idx, int) else 10**9, name))
    items.sort(key=lambda x: (x[0], x[1]))
    return [name for _, name in items]


def _healthz_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return resp.status == 200 and "OK" in body
    except Exception:
        return False


def _wait_for_server(port: int, timeout_s: float, proc: subprocess.Popen | None = None) -> None:
    start = time.time()
    while True:
        # If the process already exited, fail fast instead of waiting for timeout.
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Policy server exited early (exit={proc.returncode}); port={port}")
        if _healthz_ok(port):
            return
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Policy server not healthy on port {port} after {timeout_s:.1f}s")
        time.sleep(0.5)


def _terminate_process(proc: subprocess.Popen, name: str, timeout_s: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    start = time.time()
    while proc.poll() is None and (time.time() - start) < timeout_s:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def _format_hydra_list(ints: list[int]) -> str:
    return "[" + ",".join(str(i) for i in ints) + "]"


@dataclasses.dataclass
class Args:
    # Paths
    behavior1k_root: Path = Path("/home/ruben/BEHAVIOR-1K")
    openpi_comet_root: Path = Path("/home/ruben/openpi-comet")
    output_root: Path = Path("./b1k_ablation_runs")
    run_id: str = dataclasses.field(default_factory=_now_run_id)

    # Which python to use for each process (useful if you have separate envs)
    serve_python: str = sys.executable
    eval_python: str = sys.executable
    # How to launch the policy server.
    # - "python": run `serve_python scripts/serve_b1k.py ...`
    # - "uv": run `uv run scripts/serve_b1k.py ...` (matches your existing workflow)
    serve_runner: str = "python"
    # If set, export XLA_PYTHON_CLIENT_MEM_FRACTION for the policy server process.
    xla_mem_fraction: float | None = None
    # Extra env vars for the policy server, like KEY=VAL (can repeat).
    serve_env: list[str] = dataclasses.field(default_factory=list)
    # If true, sanitize env vars for the policy server so it doesn't inherit Isaac/Omni PYTHONPATH, etc.
    serve_clean_env: bool = True

    # Ports / timeouts
    port: int = 8000
    server_start_timeout_s: float = 300.0

    # Task/instance selection
    tasks: list[str] | None = None  # if None, use all 50 from task_mapping.json
    instances: list[int] = dataclasses.field(default_factory=lambda: list(range(10)))

    # Eval config knobs (passed to Hydra overrides)
    headless: bool = True
    write_video: bool = True
    partial_scene_load: bool = False
    max_steps: int | None = None
    enable_wm_monitor: bool = False

    # Policy server args passthrough (in addition to task_name/port/wm_in_prompt)
    #
    # Example (checkpoint):
    #   --serve_args --policy Checkpoint --policy.config pi0_aloha_sim --policy.dir /path/to/ckpt
    #
    # Example (default):
    #   --serve_args --policy Default
    serve_args: list[str] = dataclasses.field(default_factory=list)

    # If True, run WM-in-prompt first, then no-WM-in-prompt. Otherwise reverse.
    wm_first: bool = True

    # If True, skip a task if all requested metrics files already exist.
    skip_if_done: bool = True


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(
        description=(
            "Run BEHAVIOR-1K evaluation for 50 tasks × 10 instances, "
            "first with WM appended to prompt, then without."
        )
    )

    # Paths
    p.add_argument("--behavior1k_root", type=Path, default=Args.behavior1k_root)
    p.add_argument("--openpi_comet_root", type=Path, default=Args.openpi_comet_root)
    p.add_argument("--output_root", type=Path, default=Args.output_root)
    p.add_argument("--run_id", type=str, default=_now_run_id())

    # Python executables (can point to different venvs/conda envs)
    p.add_argument("--serve_python", type=str, default=sys.executable)
    p.add_argument("--eval_python", type=str, default=sys.executable)
    p.add_argument("--serve_runner", type=str, choices=["python", "uv"], default="python")
    p.add_argument(
        "--xla_mem_fraction",
        type=float,
        default=None,
        help="If set, export XLA_PYTHON_CLIENT_MEM_FRACTION for the policy server (e.g. 0.5).",
    )
    p.add_argument(
        "--serve_env",
        action="append",
        default=[],
        help="Extra env var for policy server (KEY=VAL). Can be repeated.",
    )
    p.add_argument(
        "--serve_clean_env",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, remove/clean env vars (e.g. PYTHONPATH) for the policy server process.",
    )

    # Ports / timeouts
    p.add_argument("--port", type=int, default=Args.port)
    p.add_argument("--server_start_timeout_s", type=float, default=Args.server_start_timeout_s)

    # Selection
    p.add_argument("--tasks", nargs="*", default=None, help="If omitted, use all 50 tasks from task_mapping.json")
    p.add_argument(
        "--instances",
        nargs="*",
        type=int,
        default=list(range(10)),
        help="Eval instance indices (0-9 for challenge self-eval).",
    )

    # Eval knobs
    p.add_argument("--headless", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--write_video", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--partial_scene_load", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument(
        "--enable_wm_monitor",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, pass +enable_wm_monitor=true to OmniGibson eval (for wm_monitor.py on port 9999).",
    )

    # Control flow
    p.add_argument("--wm_first", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--skip_if_done", default=True, action=argparse.BooleanOptionalAction)

    # Passthrough args for serve_b1k.py.
    # NOTE: This MUST be the last flag in your command.
    p.add_argument(
        "--serve_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Pass-through args to serve_b1k.py (must be last). Example: --serve_args policy:checkpoint --policy.config ...",
    )

    ns = p.parse_args(argv)
    return Args(
        behavior1k_root=ns.behavior1k_root,
        openpi_comet_root=ns.openpi_comet_root,
        output_root=ns.output_root,
        run_id=ns.run_id,
        serve_python=ns.serve_python,
        eval_python=ns.eval_python,
        serve_runner=ns.serve_runner,
        xla_mem_fraction=ns.xla_mem_fraction,
        serve_env=list(ns.serve_env),
        serve_clean_env=ns.serve_clean_env,
        port=ns.port,
        server_start_timeout_s=ns.server_start_timeout_s,
        tasks=ns.tasks,
        instances=ns.instances,
        headless=ns.headless,
        write_video=ns.write_video,
        partial_scene_load=ns.partial_scene_load,
        max_steps=ns.max_steps,
        enable_wm_monitor=ns.enable_wm_monitor,
        serve_args=list(ns.serve_args),
        wm_first=ns.wm_first,
        skip_if_done=ns.skip_if_done,
    )


def main(args: Args) -> None:
    behavior1k_root = args.behavior1k_root.expanduser().resolve()
    openpi_root = args.openpi_comet_root.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve() / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    task_mapping_path = openpi_root / "scripts" / "task_mapping.json"
    if not task_mapping_path.exists():
        raise FileNotFoundError(f"Missing task mapping at: {task_mapping_path}")

    all_tasks = _load_task_list(task_mapping_path)
    tasks = args.tasks if args.tasks is not None and len(args.tasks) > 0 else all_tasks

    # We'll invoke the evaluator as a module (matches: python -m omnigibson.learning.eval).
    og_root = behavior1k_root / "OmniGibson"
    if not (og_root / "omnigibson").exists():
        raise FileNotFoundError(f"Missing OmniGibson package root at: {og_root / 'omnigibson'}")

    serve_py = openpi_root / "scripts" / "serve_b1k.py"
    if not serve_py.exists():
        raise FileNotFoundError(f"Missing policy server at: {serve_py}")

    run_meta = {
        "run_id": args.run_id,
        "timestamp": _dt.datetime.now().isoformat(),
        "behavior1k_root": str(behavior1k_root),
        "openpi_comet_root": str(openpi_root),
        "port": args.port,
        "tasks": tasks,
        "instances": args.instances,
        "wm_first": args.wm_first,
        "serve_python": args.serve_python,
        "eval_python": args.eval_python,
        "serve_runner": args.serve_runner,
        "xla_mem_fraction": args.xla_mem_fraction,
        "serve_env": args.serve_env,
        "serve_args": args.serve_args,
    }
    (out_root / "run_config.json").write_text(json.dumps(run_meta, indent=2))

    modes = [("wm", True), ("no_wm", False)]
    if not args.wm_first:
        modes = list(reversed(modes))

    # Ensure CTRL-C stops child processes.
    stop = {"requested": False}

    def _sigint(_sig, _frame):
        stop["requested"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint)

    for mode_name, wm_in_prompt in modes:
        mode_root = out_root / mode_name
        mode_root.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] mode={mode_name} wm_in_prompt={wm_in_prompt}", flush=True)

        for task_name in tasks:
            if stop["requested"]:
                return
            print(f"[TASK] {mode_name}/{task_name}", flush=True)

            task_out = mode_root / task_name
            task_out.mkdir(parents=True, exist_ok=True)

            metrics_dir = task_out / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            if args.skip_if_done:
                # NOTE: OmniGibson eval writes metrics using the *actual* instance id (from test_instances.csv),
                # not the 0..9 index. So we can't reliably predict filenames without parsing the dataset metadata.
                # Instead, treat "done" as: we already have >= N metrics jsons for this task.
                existing = list(metrics_dir.glob(f"{task_name}_*_0.json"))
                if len(existing) >= len(args.instances):
                    continue

            serve_log = open(task_out / "serve.log", "w", buffering=1)
            eval_log = open(task_out / "eval.log", "w", buffering=1)

            server_proc: subprocess.Popen | None = None
            try:
                # 1) Start policy server for this task.
                if args.serve_runner == "uv":
                    serve_prefix = ["uv", "run"]
                else:
                    serve_prefix = [args.serve_python]

                # Build serve command. Note: tyro parses bool flags as --flag / --no-flag, not --flag=true.
                serve_cmd = [
                    *serve_prefix,
                    str(serve_py),
                    f"--task_name={task_name}",
                    f"--port={args.port}",
                ]
                if wm_in_prompt:
                    serve_cmd.append("--wm-in-prompt")
                else:
                    serve_cmd.append("--no-wm-in-prompt")
                serve_cmd.extend(args.serve_args)

                serve_log.write("[CMD] " + " ".join(serve_cmd) + "\n")
                serve_log.flush()
                print(f"[SERVER] starting (log={serve_log.name})", flush=True)

                env = os.environ.copy()
                if args.serve_clean_env:
                    # Avoid leaking Isaac/Omni python path into the OpenPi venv.
                    # This is a common cause of numpy import failures due to ABI mismatch.
                    if "PYTHONPATH" in env:
                        parts = [p for p in env["PYTHONPATH"].split(os.pathsep) if p]
                        banned = ("isaac-sim", "omni.kit.pip_archive", "extscache", "pip_prebundle")
                        parts = [p for p in parts if not any(b in p for b in banned)]
                        if parts:
                            env["PYTHONPATH"] = os.pathsep.join(parts)
                        else:
                            env.pop("PYTHONPATH", None)
                    env.pop("PYTHONHOME", None)
                    # Reduce site-packages surprises.
                    env.setdefault("PYTHONNOUSERSITE", "1")
                if args.xla_mem_fraction is not None:
                    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
                for item in args.serve_env:
                    if not item:
                        continue
                    if "=" not in item:
                        raise ValueError(f"--serve_env must be KEY=VAL, got: {item!r}")
                    k, v = item.split("=", 1)
                    env[k] = v

                server_proc = subprocess.Popen(
                    serve_cmd,
                    cwd=str(openpi_root),
                    env=env,
                    stdout=serve_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )

                try:
                    _wait_for_server(args.port, timeout_s=args.server_start_timeout_s, proc=server_proc)
                except Exception as e:
                    raise RuntimeError(
                        f"Policy server failed to start for {mode_name}/{task_name}. "
                        f"Check {serve_log.name} for details."
                    ) from e
                print("[SERVER] healthy", flush=True)

                # 2) Run evaluator for requested instances.
                overrides = [
                    "policy=websocket",
                    f"task.name={task_name}",
                    f"log_path={task_out}",
                    f"headless={str(args.headless).lower()}",
                    f"write_video={str(args.write_video).lower()}",
                    f"partial_scene_load={str(args.partial_scene_load).lower()}",
                    f"model.host=127.0.0.1",
                    f"model.port={args.port}",
                    f"eval_instance_ids={_format_hydra_list(args.instances)}",
                ]
                if args.enable_wm_monitor:
                    overrides.append("+enable_wm_monitor=true")
                if args.max_steps is not None:
                    overrides.append(f"max_steps={args.max_steps}")

                eval_cmd = [args.eval_python, "-m", "omnigibson.learning.eval", *overrides]
                eval_log.write("[CMD] " + " ".join(eval_cmd) + "\n")
                eval_log.flush()
                print(f"[EVAL] starting (log={eval_log.name})", flush=True)

                eval_proc = subprocess.Popen(
                    eval_cmd,
                    cwd=str(og_root),
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                rc = eval_proc.wait()
                if rc != 0:
                    raise RuntimeError(f"Eval failed for {mode_name}/{task_name} (exit={rc})")
                print("[EVAL] done", flush=True)

            except KeyboardInterrupt:
                stop["requested"] = True
                raise
            finally:
                try:
                    eval_log.flush()
                    serve_log.flush()
                except Exception:
                    pass

                if server_proc is not None:
                    # Try to terminate the whole process group if possible.
                    try:
                        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
                    except Exception:
                        pass
                    _terminate_process(server_proc, "policy_server")

                try:
                    eval_log.close()
                except Exception:
                    pass
                try:
                    serve_log.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main(parse_args())

