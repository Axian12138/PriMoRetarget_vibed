---
name: primo-episode-to-g1
description: 在独立 PriMoRetarget 仓库内，从带 Axis BVH、原始 G1 motion 和对齐元数据的 episode 生成 G1 motion。使用足部优先、全轨迹 EEF 平滑权重优化，全部代码和入口位于 PriMo 内，固定资产使用相对路径。
---

# Episode → PriMo G1 motion

从 **PriMoRetarget 仓库根目录**执行。所有代码都在本仓库：
`src/primoretarget/episode.py`、`src/primoretarget/bvh_fk.py` 和求解器模块。
启动脚本只把本仓库的 `src` 加入 Python 搜索路径。

## 执行一次

```bash
./scripts/primo_episode_to_g1.sh --episode data/episodes/<episode-id>
```

仅输入 episode。默认输出 `outputs/<日期>/primo_episode/runs/<运行编号>/g1_motion.npz`。
指定输出目录时：

```bash
./scripts/primo_episode_to_g1.sh \
  --episode data/episodes/<episode-id> --output outputs/my-run
```

使用相对路径。运行一次并报告实际结果路径与退出状态。内部数值迭代属于
本次求解，不根据某条 motion 的残差反复调参、追加帧约束或后处理。
除非用户要求，不额外检查、消融、启动服务器。此前的优化产物不算本流程
的新复现结果；必须实际运行上述入口才能报告新的 motion 已生成。

## 输入与依赖

Episode 含 `human_motion.bvh`、`retarget/g1_motion.npz`、
`retarget/retarget.json`、`episode_manifest.json`、`scene_config.json`。
原始 G1 motion 是初值，BVH 与根对齐元数据用于重新生成 human world。

固定资产布局：

- G1 XML 和其 mesh 文件放在 `assets/robots/g1_29dof_dex3/`，默认 XML
  为 `g1_29dof_dex3_offsethand.xml`。相对 `--robot` 可以指定其他资产位置。
- 场景路径来自 manifest 的相对 `mujoco_conversion.task_xml`，从 episode
  及父目录自动查找。例如 episode 在 `data/episodes/`，场景在 `data/scenes/`。
  相对 `--assets-root` 可以明确资产包根目录。目标 ID 从 `scene_config.task.id` 推导。
- 模型及场景是固定资产，不导入提供资产的其他项目代码；episode 可以位于
  仓库外，但只能作为相对输入数据路径。
- 不执行 JSON 的历史绝对路径。缺资产时报告缺失，不用旧 motion 冒充。

依赖只有本仓库 Python 包及 NumPy、SciPy、MuJoCo。新环境可在本仓库执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## 固定求解契约

- P0：人体推导的支撑/摆脚目标和真实机器人足底几何；硬约束。
- P1：双手 human wrist XYZ 目标全轨迹存在。15 cm 交互窗口只改变权重：
  基线 1，窗内 1000，前后固定 0.3 秒 Gaussian 过渡。
- P2：原姿态、腕/足朝向、全轨迹速度/加速度与软避碰。加权 P1 与这些项
  在 P0 零空间共同求解，允许 EEF 精度和平滑度权衡，不使用逐帧 EEF 硬等式。

保持 30 次迭代上限及固定配置。单位、对齐和 schema 见
[输入输出契约](references/contract.md)。录制格式不同须明确适配，不写入
特定 episode 的帧号、腕角或方向例外。

交付 `g1_motion.npz` 和相对溯源记录。运行完成不代表视觉、碰撞或物理验收。
