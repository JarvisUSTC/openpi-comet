#!/usr/bin/env python3
"""
Working Memory Monitor v2
=========================
接收环境状态，以 "当前快照 + 关键事件 + 最近动作" 格式显示。
用于测试观察 WM 捕捉到的信息结构。

用法:
    python wm_monitor.py                       # 默认 localhost:9999
    python wm_monitor.py --port 9999 --raw     # 同时打印原始数据
"""

import json
import socket
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from collections import deque


# ============================================================
# B1K R1Pro Action 空间索引 (23-dim)
# base(3) + trunk(4) + left_arm(7) + right_arm(7) + left_grip(1) + right_grip(1)
# ============================================================
ACTION_SLICES = {
    'base':       (0, 3),
    'trunk':      (3, 7),
    'left_arm':   (7, 14),
    'right_arm':  (14, 21),
    'left_grip':  (21, 22),
    'right_grip': (22, 23),
}


class WMMonitor:
    def __init__(self, host='localhost', port=9999, show_raw=False, refresh_every=5):
        self.host = host
        self.port = port
        self.show_raw = show_raw
        self.refresh_every = refresh_every

        # --- 状态追踪 ---
        self.key_events = []                    # 关键事件列表（全量保留）
        self.action_history = deque(maxlen=5)   # 最近 N 个动作
        self.wm_history = deque(maxlen=200)     # WM 快照历史
        self.prev_wm = None                     # 上一帧 WM
        self.step_count = 0                     # 已处理帧数

        # --- 日志 ---
        self.log_dir = "wm_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file   = open(f"{self.log_dir}/wm_{ts}.jsonl", "w")
        self.event_file = open(f"{self.log_dir}/events_{ts}.jsonl", "w")

        # --- Socket 服务端 ---
        print(f"[WM Monitor v2] Starting on {host}:{port} ...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)

        print(f"[WM Monitor v2] Waiting for eval connection ...")
        self.conn, addr = self.server.accept()
        print(f"[WM Monitor v2] Connected from {addr}\n")

    # ================================================================
    # 1. WM 快照提取
    # ================================================================

    def extract_wm(self, env_state):
        """从原始 env_state 提取结构化 Working Memory 快照"""
        step      = env_state.get('step', 0)
        task_name = env_state.get('task_name', '')
        target    = env_state.get('target_object', 'unknown')
        grasped   = env_state.get('is_grasping', False)
        eef_pos   = env_state.get('eef_position', [0, 0, 0])
        obj_dist  = env_state.get('object_distance', 999.0)
        goal_dist = env_state.get('goal_distance', 999.0)
        max_steps = env_state.get('max_steps', 4300)
        gripper   = env_state.get('gripper_state', 0.0)

        stage = self._infer_stage(grasped, obj_dist, goal_dist)
        pct   = step / max_steps * 100 if max_steps > 0 else 0

        wm = {
            'step': step,
            'task_name': task_name,
            'stage': stage,
            'target': target,
            'grasped': grasped,
            'eef_position': eef_pos,
            'eef_distance': obj_dist,
            'goal_distance': goal_dist,
            'gripper_state': gripper,
            'progress': f"{step}/{max_steps}",
            'progress_pct': pct,
            'max_steps': max_steps,
        }
        # If eval broadcasts its own WM, include it for visibility (e.g. subtask_done).
        try:
            wm_eval = env_state.get('working_memory', None)
            if isinstance(wm_eval, dict):
                if 'subtask_done' in wm_eval:
                    wm['subtask_done'] = wm_eval.get('subtask_done')
                if 'holding' in wm_eval:
                    wm['holding'] = wm_eval.get('holding')
                if 'dropped' in wm_eval:
                    wm['dropped'] = wm_eval.get('dropped')
        except Exception:
            pass

        return wm

    def _infer_stage(self, grasped, obj_dist, goal_dist):
        """基于规则推断当前任务阶段"""
        if not grasped:
            if obj_dist > 0.5:
                return 'approach'
            elif obj_dist > 0.15:
                return 'reach'
            else:
                return 'grasp'
        else:
            if goal_dist > 0.3:
                return 'transport'
            elif goal_dist > 0.1:
                return 'align'
            else:
                return 'place'

    # ================================================================
    # 2. 关键事件检测
    # ================================================================

    def detect_events(self, wm):
        """对比前后帧，检测并记录关键事件"""
        step = wm['step']

        # 首帧 → 任务开始
        if self.prev_wm is None:
            self._add_event(step, f"Task started: {wm['task_name'] or '?'}")
            return

        prev = self.prev_wm

        # 阶段切换
        if wm['stage'] != prev['stage']:
            self._add_event(
                step,
                f"Stage: {prev['stage']} → {wm['stage']} "
                f"(distance: {prev['eef_distance']:.2f}m → {wm['eef_distance']:.2f}m)"
            )

        # 抓取 / 释放
        if wm['grasped'] != prev['grasped']:
            if wm['grasped']:
                self._add_event(step, f"Grasped: {wm['target']}")
            else:
                self._add_event(step, f"Released: {wm['target']}")

        # 距离突变 (> 0.3m 跳变)
        dist_delta = wm['eef_distance'] - prev['eef_distance']
        if abs(dist_delta) > 0.3:
            direction = "closer" if dist_delta < 0 else "farther"
            self._add_event(
                step,
                f"Distance {direction}: {prev['eef_distance']:.2f}m → {wm['eef_distance']:.2f}m"
            )

    def _add_event(self, step, description):
        event = {'step': step, 'desc': description}
        self.key_events.append(event)
        self.event_file.write(json.dumps(event) + "\n")
        self.event_file.flush()

    # ================================================================
    # 3. 动作解析 (23-dim → 可读)
    # ================================================================

    def parse_action(self, step, action_raw):
        """将 23-dim action 解析为可读描述"""
        if action_raw is None:
            return None

        try:
            # 处理各种可能的格式
            if isinstance(action_raw, dict):
                # 如果是 dict，取 'action' key 或第一个 array-like value
                if 'action' in action_raw:
                    arr = np.asarray(action_raw['action']).flatten()
                else:
                    # 尝试取第一个 list/array 类型的 value
                    arr = None
                    for v in action_raw.values():
                        if isinstance(v, (list, tuple)):
                            arr = np.asarray(v).flatten()
                            break
                    if arr is None:
                        return {'step': step, 'desc': f'dict_keys={list(action_raw.keys())}', 'raw_type': 'dict'}
            elif isinstance(action_raw, (list, tuple)):
                arr = np.asarray(action_raw).flatten()
            else:
                return {'step': step, 'desc': f'type={type(action_raw).__name__}', 'raw_type': 'unknown'}

            # 分解为各部分
            parts = {}
            norms = {}
            for name, (s, e) in ACTION_SLICES.items():
                if e <= len(arr):
                    parts[name] = arr[s:e]
                    norms[name] = float(np.linalg.norm(arr[s:e]))

            # 找主要运动方向
            if norms:
                dominant = max(norms, key=norms.get)
            else:
                dominant = '?'

            # 构建描述字符串
            desc_parts = []
            # 先显示 arm 和 base
            for name in ['left_arm', 'right_arm', 'base', 'trunk']:
                if name in parts and norms.get(name, 0) > 0.005:
                    vals_str = ', '.join(f"{v:.3f}" for v in parts[name])
                    desc_parts.append(f"{name}=[{vals_str}]")
            # 夹爪
            for gname in ['left_grip', 'right_grip']:
                if gname in parts:
                    val = float(parts[gname][0])
                    if abs(val) > 0.01:
                        desc_parts.append(f"{gname}={val:.2f}")

            return {
                'step': step,
                'dominant': dominant,
                'desc': ', '.join(desc_parts) if desc_parts else 'near-zero',
                'norms': norms,
                'dim': len(arr),
            }

        except Exception as e:
            return {'step': step, 'desc': f'parse_error: {e}', 'dominant': '?', 'norms': {}, 'dim': 0}

    # ================================================================
    # 4. 终端显示
    # ================================================================

    def display(self, wm):
        """ANSI 清屏后以结构化格式显示 Working Memory"""
        # 清屏
        print("\033[2J\033[H", end="")

        W = 70
        print("=" * W)
        print("  [WORKING_MEMORY]")
        print("=" * W)

        # --- Current State ---
        print()
        print("  Current State:")
        print(f"    stage:        {wm['stage']}")
        print(f"    target:       {wm['target']}")
        print(f"    grasped:      {wm['grasped']}")
        # Optional eval-side WM extras
        if 'holding' in wm:
            print(f"    holding:      {wm['holding']}")
        if 'dropped' in wm:
            print(f"    dropped:      {wm['dropped']}")
        if 'subtask_done' in wm:
            sd = wm['subtask_done']
            if isinstance(sd, dict):
                items = []
                for k in sorted(sd.keys(), key=lambda x: str(x)):
                    v = sd.get(k)
                    if hasattr(v, "item"):
                        v = v.item()
                    if isinstance(v, bool):
                        v = "true" if v else "false"
                    items.append(f"{k}={v}")
                sd_str = ", ".join(items)
            else:
                sd_str = str(sd)
            print(f"    subtask_done: {sd_str}")
        print(f"    eef_distance: {wm['eef_distance']:.2f}m")
        print(f"    eef_position: [{wm['eef_position'][0]:.3f}, "
              f"{wm['eef_position'][1]:.3f}, {wm['eef_position'][2]:.3f}]")
        print(f"    gripper:      {wm['gripper_state']:.3f}")
        print(f"    progress:     {wm['progress']} ({wm['progress_pct']:.1f}%)")

        # --- Key Events ---
        print()
        print("  Key Events:")
        shown = self.key_events[-10:]
        if len(self.key_events) > 10:
            print(f"    ... ({len(self.key_events) - 10} earlier events omitted)")
        for evt in shown:
            print(f"    [Step {evt['step']:>4d}] {evt['desc']}")
        # 添加 "当前位置" 标记
        print(f"    [Step {wm['step']:>4d}] Current position")

        # --- Last Actions ---
        n_actions = len(self.action_history)
        print()
        print(f"  Last {n_actions} Actions:")
        if n_actions == 0:
            print("    (no actions yet)")
        else:
            for act in self.action_history:
                dominant_tag = f"[{act.get('dominant', '?')}]" if act.get('dominant') else ""
                print(f"    [{act['step']:>4d}] {dominant_tag} {act['desc']}")

        # --- 底部状态栏 ---
        print()
        print("-" * W)
        print(f"  events: {len(self.key_events)} | "
              f"steps: {self.step_count} | "
              f"refresh: every {self.refresh_every} steps | "
              f"time: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * W)

    def display_raw(self, env_state):
        """调试模式：打印原始 JSON"""
        print(f"\n--- RAW env_state (step={env_state.get('step')}) ---")
        # 截断 action 显示（太长了）
        display = dict(env_state)
        action = display.get('action')
        if isinstance(action, list) and len(action) > 10:
            display['action'] = f"[{len(action)}-dim array] first5={action[:5]}"
        elif isinstance(action, dict):
            display['action'] = f"dict keys={list(action.keys())}"
        print(json.dumps(display, indent=2, default=str))
        print("---")

    # ================================================================
    # 5. 日志
    # ================================================================

    def save_snapshot(self, wm):
        """保存每帧 WM 快照"""
        entry = {**wm, 'ts': datetime.now().isoformat(), 'n_events': len(self.key_events)}
        self.log_file.write(json.dumps(entry, default=str) + "\n")
        self.log_file.flush()

    # ================================================================
    # 6. 主循环
    # ================================================================

    def run(self):
        """接收数据 → 提取WM → 检测事件 → 解析动作 → 显示"""
        try:
            buffer = ""
            while True:
                data = self.conn.recv(16384).decode('utf-8')
                if not data:
                    print("\n[WM Monitor] Connection closed by eval")
                    break

                buffer += data

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue

                    try:
                        env_state = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    self.step_count += 1

                    # (可选) 打印原始数据
                    if self.show_raw and self.step_count <= 3:
                        self.display_raw(env_state)

                    # 1) 提取 WM 快照
                    wm = self.extract_wm(env_state)

                    # 2) 检测关键事件
                    self.detect_events(wm)
                    self.prev_wm = wm

                    # 3) 解析动作
                    action_raw = env_state.get('action', None)
                    action_parsed = self.parse_action(wm['step'], action_raw)
                    if action_parsed is not None:
                        self.action_history.append(action_parsed)

                    # 4) 定期刷新显示
                    if self.step_count % self.refresh_every == 0 or self.step_count <= 3:
                        self.display(wm)

                    # 5) 保存日志
                    self.save_snapshot(wm)

                    # 6) 历史
                    self.wm_history.append(wm)

        except KeyboardInterrupt:
            print("\n[WM Monitor] Interrupted by user")

        finally:
            self.final_report()
            self.cleanup()

    # ================================================================
    # 7. 最终报告
    # ================================================================

    def final_report(self):
        if not self.wm_history:
            return

        W = 70
        print("\n" + "=" * W)
        print("  FINAL REPORT")
        print("=" * W)

        # 阶段分布
        stage_counts = {}
        for wm in self.wm_history:
            s = wm['stage']
            stage_counts[s] = stage_counts.get(s, 0) + 1

        print("\n  Stage Distribution:")
        total = len(self.wm_history)
        for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = '#' * int(pct / 2)
            print(f"    {stage:12s}: {count:4d} steps ({pct:5.1f}%)  {bar}")

        # 所有关键事件
        print(f"\n  All Key Events ({len(self.key_events)}):")
        for evt in self.key_events:
            print(f"    [Step {evt['step']:>4d}] {evt['desc']}")

        # 动作统计
        if self.action_history:
            print(f"\n  Action Stats:")
            last = list(self.action_history)[-1]
            print(f"    Last action dim: {last.get('dim', '?')}")
            print(f"    Last dominant:   {last.get('dominant', '?')}")
            if last.get('norms'):
                print(f"    Last norms:      {last['norms']}")

        print("\n" + "=" * W)

    def cleanup(self):
        self.log_file.close()
        self.event_file.close()
        self.conn.close()
        self.server.close()
        print(f"[WM Monitor] Logs saved to: {self.log_dir}/")


# ================================================================
# Entry point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Working Memory Monitor v2')
    parser.add_argument('--host', default='localhost', help='Listen host')
    parser.add_argument('--port', type=int, default=9999, help='Listen port')
    parser.add_argument('--raw', action='store_true', help='Print first 3 raw env_state for debugging')
    parser.add_argument('--refresh', type=int, default=5, help='Display refresh interval (steps)')
    args = parser.parse_args()

    monitor = WMMonitor(
        host=args.host,
        port=args.port,
        show_raw=args.raw,
        refresh_every=args.refresh,
    )
    monitor.run()


if __name__ == "__main__":
    main()
