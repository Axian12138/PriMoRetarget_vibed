# 输入、坐标与复现边界

## 从 episode 生成 human world

本仓库入口是 `src/primoretarget/episode.py`，骨架 reader 是
`src/primoretarget/bvh_fk.py`。
BVH reader 仅负责骨架 FK，不调用其他 retarget 求解器。

当前录制格式为 Axis BVH：厘米、X 向上，位置重排为 `[Z, X, Y]` 后乘
0.01。此约定属于采集格式，不是特定 motion 的数值修补。BVH、原 G1 和
retarget 元数据必须保持同样帧数与 FPS。

根旋转来自 `retarget.json.root_alignment.rotation_xyzw`；平移由
`actual_start_xyz - R(reference_start_xyz)` 推导。不要从旧优化输出估计对齐。
输出 `human_world.npz` 包含世界位置、joint names、edges、FPS、源帧编号、
根变换与 BVH hash。

## 固定配置

本 skill 封装 `18-49-54` 运行使用的算法与配置，不复制该结果文件：
PriMo、15 cm、全轨迹 EEF、权重 1/1000、0.3 秒过渡、30 次迭代上限，
场景净空 0.015 m、软避碰权重 100。时间正则沿用求解器按 FPS 换算的配置。
目标物体 ID 和精确窗口均由输入推导。

同输入、同资产和同求解器版本可复现该数值流程；跨 SciPy/MuJoCo 版本
不承诺 NPZ 文件逐字节相同。程序生成 `episode_pipeline.json`，记录相对
输入/资产路径、hash 和实际 solver 参数。资产依赖不属于额外 motion 输入。

## 输出

`g1_motion.npz` 保持七个字段：`fps`、`joint_pos`、`joint_vel`、
`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。
29 个非手指关节、30 个 body，quaternion 为 WXYZ；速度由结果重新生成。
固定手指关节沿用模型默认值。

当前适用条件是同步 episode、G1、平面支撑地面、一个静态目标。
不声称支持任意 BVH 格式、楼梯、下跪、动态物体或真实抓持。

`--prepare-only` 只生成 human world 和参数清单，不生成最终 motion。
它用于维护入口时的单次准备检查，不是生产流程中反复改进结果的步骤。

## 独立仓库边界

代码依赖仅为本仓库模块和 pyproject 中的第三方 Python 包。Skill 位于
`skills/primo-episode-to-g1/`，脚本位于 `scripts/`；不要查找父项目的代码
或使用父项目的 Python 搜索路径。固定模型文件默认在 `assets/robots/`。
这些大体积资产和录制数据不提交到代码仓库，部署时需提供 XML 及其引用
的全部 mesh 文件，或用相对资产路径配置。此前优化结果仅说明配置来源，
不能替代独立入口实际执行的新结果。
