# 用 Git 管理 Stage-3 消融实验

实验列表在 [run_global_ablation.sh](run_global_ablation.sh)。PBS 只保留资源申请、工作目录、
总日志、GPU 监控、Conda 激活和 PyTorch 检查；**从原来的 `Stage-3 Experiments` 到文件末尾
全部替换为**：

```bash
bash examples/agentstream_trainer/run_global_ablation.sh
```

这行在 PBS 已经 `cd "$WORKDIR"`、且 `WORKDIR` 指向 SEED 根目录后执行。
不要使用 `exec bash`：保留 PBS 父 shell，才能执行其 `EXIT` trap 停止 GPU 监控。
保留原 PBS 的 `set -eo pipefail`，子脚本失败即让 PBS 任务以非零状态退出。

## 日常工作流

1. 在开发服务器修改脚本顶部的 `BATCH_SIZE`、`TOTAL_STEPS`、`COMMON_ENV` 或下面的四个
   `run_exp` 调用。新增/删除实验时，最终汇总自动同步，无需另改名称列表。
2. 提交、推送这些修改；超算登录节点拉取后，照常 `qsub` 现有 PBS 文件。不需要再用 nano
   把实验命令逐条复制进去。请在提交任务前完成拉取，避免运行过程中修改启动文件。
3. 机器路径、GPU 和密钥仍放在各服务器自己的 SEED 私有环境文件中，不随 Git 同步；
   公共默认值仍来自 [agentstream_full.env](agentstream_full.env)。

在 SEED 根目录预览四组命令（不要求安装训练依赖或存在模型）：

```bash
bash examples/agentstream_trainer/run_global_ablation.sh --dry-run
```

预览仍会读取本机环境文件和公共配置（配置中的临时目录初始化也可能执行），但不会调用
训练 launcher、启动检索服务或创建实验日志；它不是模型、GPU 或 Hydra 的端到端校验。

## 保留的行为与小修正

- 四组实际算法参数不变，依次为 SEED、SEED+Global、SEED+Global+Replay、SEED+Global+Opt。
- 默认每步 **6 题、20 个训练 step**，不是每题的 `AGENTSTREAM_MAX_STEPS`。
  原 PBS 名称中的 `steps24` 已改为随 `TOTAL_STEPS` 派生。三域各 96 题时，20 步只覆盖
  流的前 120 题，不是完整单遍；完整遍历需 48 步。
- 最后一组 `EMA_MODE=ref` 的含义是 EMA **KL reference**，不是 EMA teacher。
- 本机环境文件先读取一次，再应用脚本中的实验覆盖；子 launcher 不会重新读私有文件
  而盖掉消融开关。未列出的参数继承公共配置/环境，仍需在对比期间保持一致。
- 每组有独立日志；PBS 下文件名含 `PBS_JOBID`，本地运行自动用时间戳和 PID。
  日志位于 SEED 根目录下的 `logs/agentstream/`。
- 某组失败后继续其余组；最终同时报告 training 和 tee 状态，任一失败则整体退出 1。
- 实验名前缀由模型 tag、版本、任务数、profile、batch、训练步数和组别派生；若更改其他
  算法参数重新跑，可设置 `AGENTSTREAM_ABLATION_PREFIX` 为新的唯一前缀，避免复用旧产物。
  同时清除继承的低层 `EXPERIMENT_NAME`/`DEFAULT_LOCAL_DIR`，防止四组落入同一目录。

PBS 的 GPU 分配与 Conda 配置未改动。注意不能普遍把调度器分配的 GPU UUID 直接改成
`0,1`：物理 GPU 未必是前两张；请按集群规则确认最终 `AGENTSTREAM_RL_GPUS` 与作业分配一致。