# 从强化学习到 CFD 流动控制：JAX 科研编程实践

这是一套面向 **二维圆柱绕流、代理模型、强化学习自动减阻控制** 的独立 Jupyter Book 课程。

- 10 个 `.ipynb`，从 MDP/Bellman 到 Q-learning、REINFORCE、GAE、PPO、SAC/TD3、model-based RL 和 CFD 控制接口。
- **核心第 5 课是可运行的 JAX PPO 连续动作训练示例**，不依赖 PyTorch、Flax 或 Gymnasium；便于看清算法。
- 全课程不把教学代替系统冒充 DNS。第 8 课保留待接入 CFD 求解器的接口。
- 理论、代码、练习、实验验证在每章内完成。建议按 00 → 09 顺序学习。

## 使用（VS Code + uv）

在 VS Code 打开本项目文件夹并在终端运行：

```bash
uv sync
```

随后在 VS Code 选择 `.venv` 下的 Python 内核，打开任一 `ipynb`，按顺序运行单元格。首次运行 JAX 需要 JIT 编译时间。

用 Jupyter Book 2 预览网页（可选）：

```bash
uv sync --extra book
uv run jupyter book start
```

如果只需要学习与运行 ipynb，不必安装 Jupyter Book。`myst.yml` 采用 Jupyter Book 2 配置。

### 平台说明

- Windows 原生 JAX CPU 支持存在平台限制；初学可尝试 CPU 安装，GPU 训练建议采用 Linux/WSL2。
- Apple Silicon Mac 的 JAX CPU 可用；macOS GPU 支持状况应以官方安装文档为准。
- 本项目主要依赖 `numpy`、`matplotlib`、`jax`、`ipykernel`。

参考资料：[Jupyter Book 2](https://jupyterbook.org/) · [JAX 安装](https://docs.jax.dev/en/latest/installation.html) · [Gymnasium API](https://gymnasium.farama.org/api/env/) · [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) · [HydroGym](https://dynamicslab.github.io/hydrogym/)

## 核心设计原则

1. **先验证数学，再相信训练曲线**。
2. **区分时间截断与真正终止**，GAE 和 bootstrap 要正确。
3. **策略评估回到真实 CFD**，不能只看代理环境 reward。
4. 控制变量、随机种子、状态归一化、reward 定义应严格记录。
