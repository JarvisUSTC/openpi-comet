# behavior-benchmark

把 `SimpleRoboAgent` 里与 skill benchmark 相关的四条链路独立出来，做成一个单独 repo：

- `serve`：启动 policy server
- `eval`：跑 skill 级评测
- `judge`：对评测视频抽帧并调用 VLM 打分
- `snapshot`：生成 skill snapshot / 回放素材

目前这四条链都已经迁到新 repo 里原生实现，不再依赖旧 repo 里的 adapter 脚本做转发。

## 这个 repo 管什么

代码目录：

```text
benchmark/
  config/     # 顶层统一配置读取
  core/       # 公共路径、schema、JSON、inventory
  serve/      # server 启动与状态记录
  eval/       # 交互式选择 + skill eval runner
  judge/      # 抽帧 + VLM judge
  snapshot/   # 交互式选择 + snapshot engine
scripts/
  run_serve
  run_eval
  run_judge
  run_snapshot
```

推荐你平时直接用 `scripts/` 里的入口，而不是自己手动敲 `python -m ...`。因为这些脚本会自动：

1. 读取 repo 根目录下的 `.env.local` / `.env`
2. 导出统一环境变量
3. 按 `CONDA_BIN` + `CONDA_ENV` 激活 conda 环境
4. 再启动对应模块

## 快速开始

先进入 repo：

```bash
cd /home/simpleai/Jiawei/behavior-benchmark
```

第一次建议先准备本地配置：

```bash
cp .env.example .env.local
```

然后只改你自己机器相关的值。最少通常要看这几个：

```bash
SRA_ROOT=/home/simpleai/Jiawei/SimpleRoboAgent
OPENROUTER_API_KEY=your_api_key
CONDA_BIN=/home/simpleai/anaconda3/bin/conda
CONDA_ENV=behavior-comet
EVAL_CONFIGS_ROOT=/home/simpleai/ruike/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs
SNAPSHOT_CONFIGS_ROOT=/home/simpleai/ruike/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
EVAL_AUTO_JUDGE=true
```

最常用命令：

```bash
bash scripts/run_serve
bash scripts/run_eval
bash scripts/run_judge --interactive
bash scripts/run_snapshot
```

## 配置怎么改

统一配置入口在：

- 代码默认值：`benchmark/config/settings.py`
- 示例模板：`.env.example`
- 你机器上的真实配置：`.env.local`

推荐做法是：

- 不要直接改 `settings.py` 里的默认值，除非你想改 repo 的通用默认行为
- 你自己的路径、API key、显存限制、日志目录，优先改 `.env.local`

配置优先级：

1. CLI 显式传参
2. 当前 shell 的环境变量
3. repo 根目录下的 `.env.local`
4. repo 根目录下的 `.env`
5. `settings.py` 里的默认值

这也是为什么“改路径 / 改 API key / 改显存”推荐改 `.env.local`，而不是直接改代码。

## 最常改的配置项

### 路径

| 变量 | 作用 |
| --- | --- |
| `SRA_ROOT` | 旧仓库 `SimpleRoboAgent` 根目录 |
| `LOGS_ROOT` | 默认日志根目录，默认是 `SRA_ROOT/logs` |
| `SNAPSHOTS_ROOT` | `snapshot` 输出目录 |
| `META_ROOT` | `tasks.jsonl` 等 meta 所在目录 |
| `AUGMENTED_ANNOTATIONS_ROOT` | eval 里 augmented prompt 的标注目录 |
| `SERVER_STATE_PATH` | serve 和 eval 共享当前 task 的状态文件 |
| `EVAL_CONFIGS_ROOT` | eval 用的 OmniGibson config 根目录 |
| `SNAPSHOT_CONFIGS_ROOT` | snapshot 用的 OmniGibson config 根目录 |
| `SNAPSHOT_ANNOTATION_ROOT` | snapshot 标注目录 |
| `SNAPSHOT_RAW_ROOT` | snapshot 原始数据目录 |

### Judge

| 变量 | 作用 |
| --- | --- |
| `OPENROUTER_API_KEY` | judge 调 VLM 的 key |
| `OPENROUTER_BASE_URL` | OpenAI-compatible base URL |
| `JUDGE_MODEL` | judge 使用的模型名 |
| `JUDGE_API_KEY_ENV` | judge 从哪个环境变量里读 key |
| `JUDGE_TIMEOUT_SECONDS` | judge 请求超时 |

### Eval

| 变量 | 作用 |
| --- | --- |
| `EVAL_LOG_PATH` | eval 默认输出目录 |
| `EVAL_MAX_STEPS` | 每条 skill 默认最大步数 |
| `EVAL_VLA_TYPE` | 默认 policy 类型 |
| `EVAL_ENV_WRAPPER` | eval 默认 wrapper |
| `EVAL_AUTO_JUDGE` | eval 结束后是否自动触发 judge |

### Serve / 运行环境

| 变量 | 作用 |
| --- | --- |
| `SERVE_PORT` | server 监听端口 |
| `SERVE_BACKEND` | 后端 repo 名称；设了就不用每次交互选 |
| `SERVE_POLICY_CONFIG` | 默认 policy config |
| `SERVE_CHECKPOINT_DIR` | 默认 checkpoint 路径 |
| `OPENPI_ROOT` | openpi repo 根目录 |
| `TASK_NAME` | server 默认启动哪个真实 task |
| `CONDA_BIN` | conda 可执行文件 |
| `CONDA_ENV` | 要激活的环境名 |
| `PYTHON_BIN` | 启动时使用的 python |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | 是否预分配 JAX/XLA 显存 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX/XLA 最多吃多少比例显存 |

### Snapshot

| 变量 | 作用 |
| --- | --- |
| `SNAPSHOT_MAX_SKILLS_PER_EPISODE` | 每个 episode 最多生成多少个 skill snapshot；`-1` 表示不限制 |
| `SNAPSHOT_SEED` | snapshot 里随机选择 / 顺序相关逻辑的随机种子，用来保证可复现 |
| `SNAPSHOT_PLAYBACK_MODE` | `replay` 或 `state` |

## 推荐工作流

通常顺序是：

1. `run_serve`
2. `run_eval`
3. 看 `result_*.json` / `judge_result_*.json`
4. 需要单独复判时再 `run_judge`
5. 需要重新做素材时再 `run_snapshot`

## 1. 启动 serve

最简单：

```bash
bash scripts/run_serve
```

默认会进入交互菜单，依次选择：

1. backend repo
2. `POLICY_CONFIG`
3. `CHECKPOINT_DIR`
4. 真实 `task`

如果你想少走菜单，可以提前在 `.env.local` 里写这些：

```bash
SERVE_BACKEND=comet
SERVE_POLICY_CONFIG=pi05_base
SERVE_CHECKPOINT_DIR=/your/checkpoint/dir
TASK_NAME=setting_mousetraps
SERVE_PORT=8000
OPENPI_ROOT=/your/openpi/repo
```

然后直接运行：

```bash
bash scripts/run_serve
```

补充说明：

- `run_serve` 会把当前 task 记录到 `SERVER_STATE_PATH`
- `run_eval` 默认会读取这个状态，这样可以自动对齐当前 server 正在服务的 task
- 如果 server 很吃显存，优先改 `.env.local` 里的 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`

## 2. 运行 eval

最简单：

```bash
bash scripts/run_eval
```

默认会交互选择：

1. task
2. episode
3. skill 类别
4. 具体 skill
5. prompt
6. log 输出目录

常用参数示例：

```bash
bash scripts/run_eval --task-index 5
bash scripts/run_eval --task-name setting_mousetraps
bash scripts/run_eval --prompt "place the mousetrap on the floor next to the sink"
bash scripts/run_eval --prompt "place the mousetrap on the floor next to the sink" --prompt-prefix "Task:"
bash scripts/run_eval --log-path /home/simpleai/Jiawei/SimpleRoboAgent/logs/vla_skill_eval
bash scripts/run_eval --no-auto-judge
```

补充说明：

- 如果没传 `--prompt`，交互模式下会从 `AUGMENTED_ANNOTATIONS_ROOT` 里选 prompt，也支持手动输入
- 每次运行前都可以选择这次要不要给 prompt 加 prefix；也可以直接用 `--prompt-prefix "Task:"` 固定指定
- 如果没传 `--log-path`，交互模式下会让你选已有目录或手动输入新目录
- `EVAL_AUTO_JUDGE=true` 时，eval 成功后会自动只对本次新生成的 `result_*.json` 调一次 judge
- 如果你不想跟随当前 server task，可以加 `--ignore-server-task`

## 3. 单独运行 judge

交互模式：

```bash
bash scripts/run_judge --interactive
```

按单条结果跑：

```bash
bash scripts/run_judge --result-json /path/to/result_episode_xxx.json
```

按目录批量跑：

```bash
bash scripts/run_judge --log-path /home/simpleai/Jiawei/SimpleRoboAgent/logs/vla_skill_eval
```

一些常用参数：

```bash
bash scripts/run_judge --result-json /path/to/result.json --sample-every-seconds 3
bash scripts/run_judge --result-json /path/to/result.json --max-frames 12
bash scripts/run_judge --result-json /path/to/result.json --dry-run
```

补充说明：

- judge 会对视频按时间抽帧，并额外对末尾 3 秒做更密采样
- judge 会读取参考图、结果 JSON、视频帧，然后调用 VLM 输出 `judge_success`、`judge_reason`、`judge_checklist`
- 如果 `OPENROUTER_API_KEY` 没配好，judge 会直接报错退出

## 4. 运行 snapshot

最简单：

```bash
bash scripts/run_snapshot
```

常用参数示例：

```bash
bash scripts/run_snapshot --output-root /home/simpleai/Jiawei/SimpleRoboAgent/logs/skill_snapshots
bash scripts/run_snapshot --max-skills-per-episode 10
bash scripts/run_snapshot --seed 7
bash scripts/run_snapshot --playback-mode replay
bash scripts/run_snapshot --record-skills
```

补充说明：

- `SNAPSHOT_SEED` 只影响 snapshot 选择和可复现性，不影响模型推理本身
- `--playback-mode` 可选 `replay` 或 `state`
- `--no-require-strict-alignment` 可以放宽 episode 对齐要求

## 输出文件一般在哪

默认情况下，大多数产物会落在旧仓库的日志目录下，也就是 `LOGS_ROOT` 或它的子目录里。

常见输出：

- eval 视频：`.../task_xxxx/skill_xx/skill_*.mp4`
- eval 结果：`.../task_xxxx/skill_xx/result_*.json`
- judge 结果：`.../task_xxxx/skill_xx/judge_result_*.json`
- snapshot 输出：`SNAPSHOTS_ROOT`

## 常见改法

### 1. 旧仓库挪位置了

改 `.env.local`：

```bash
SRA_ROOT=/new/path/to/SimpleRoboAgent
LOGS_ROOT=/new/path/to/SimpleRoboAgent/logs
```

### 2. 想把 eval 输出到另一个日志目录

改 `.env.local`：

```bash
EVAL_LOG_PATH=/your/new/log/path
```

或者只对这一次生效：

```bash
bash scripts/run_eval --log-path /your/new/log/path
```

### 3. judge 的 key / 模型要改

改 `.env.local`：

```bash
OPENROUTER_API_KEY=your_api_key
JUDGE_MODEL=seed-2.0-lite
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 4. serve 太吃显存

改 `.env.local`：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

改完后要重启 `run_serve` 才会生效。

### 5. 不想 eval 后自动 judge

改 `.env.local`：

```bash
EVAL_AUTO_JUDGE=false
```

或者只关闭这一次：

```bash
bash scripts/run_eval --no-auto-judge
```

## 直接用 Python 入口也可以

如果你已经自己处理好了环境变量和 conda 环境，也可以直接用：

```bash
python -m benchmark.serve.cli
python -m benchmark.eval.cli
python -m benchmark.judge.cli
python -m benchmark.snapshot.cli
```

但平时还是更推荐你直接用：

```bash
bash scripts/run_serve
bash scripts/run_eval
bash scripts/run_judge
bash scripts/run_snapshot
```
