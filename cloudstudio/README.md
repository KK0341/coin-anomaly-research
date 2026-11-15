# CloudStudio 准备与部署

本地项目：`E:\异常检测`；图像库：`E:\钱币鉴别`。本地编辑 → GitHub **私有**仓库 → 课程 CloudStudio GPU 空间。云端无需安装 Codex。

当前部署流程以 [最新操作说明](操作说明.md) 为准；数据上传与正式训练均尚未执行。A1 是首个路径适配示例，不代表已选择下一轮科研实验。其他实验尚不能直接套用本入口。

## 1. 已有私有仓库与认证

已核实 GitHub 仓库为 [KK0341/coin-anomaly-research](https://github.com/KK0341/coin-anomaly-research)，默认分支 main。本地使用系统 Git Credential Manager，保留远程初始 README 与提交历史，不使用强推。GitHub CLI 未安装且无需额外安装。

首次提交及后续同步前运行 `python cloudstudio/audit_sync.py`，暂存后运行 `python cloudstudio/audit_sync.py --staged` 并查看 diff。`.gitignore` 默认拒绝全部，只开放明确列出的源码、部署文件及轻量结果目录；新增实验代码要主动加入清单。数据、权重、缓存与本地接收目录不入库。

`source_sha256.json` 保存原始字节基线，`source_lf_sha256.json` 保存 Git 按 .gitattributes 转换 LF 后的基线，兼容 Windows/Linux 换行差异。算法内容变化仍会被报告。凭据只通过系统凭据存储或云端手动授权，不写进代码或 URL。

## 2. 云端环境

从课程入口进入获分配的 GPU 空间，先确认额度、GPU 型号和持久存储位置；不要默认创建另一个收费实例。课程权限、磁盘保留策略、停机规则需要以课程界面为准，本地无法核实。

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_PRIVATE_REPO.git coin-research
cd coin-research
nvidia-smi
python3 --version
bash cloudstudio/setup.sh
source .venv-cloud/bin/activate
python cloudstudio/check_env.py
```

优先复用课程镜像已有的兼容 torch/torchvision。setup 创建独立 venv（可见镜像基础包），安装 A1 的其余依赖，不改宿主环境。如果检查失败且需要自行安装 PyTorch，根据实际驱动从[官方安装选择器](https://pytorch.org/get-started/locally/)选择 Linux CUDA 包。与本地相同的候选组合如下，**仅在驱动支持 CUDA 12.8 wheel 时使用**：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip check
python cloudstudio/check_env.py
python -m pip freeze > /YOUR_PERSISTENT_DIR/environment-cloud.txt
```

建议 Python 3.11～3.13，能提供 3.13 时与本地一致。CUDA wheel 与驱动是否兼容需云端实测，不要盲目升级驱动。A10 的 DINOv2 本地源码等不在这个 A1 依赖集合中，需单独核对。

## 3. 数据与资产（本次不上传）

未来自行把资产放在**仓库外的持久目录**，例：

```text
/YOUR_PERSISTENT_DIR/coin/
  splits/训练集.csv、验证集.csv、测试集.csv
  images/数据采集/民国中央/... （对应 E:\钱币鉴别 以下的原结构）
  assets/A0/模型/convnext_tiny-983f1562.pth
  runs/a1-cloud-001/
```

三个 CSV 必须原字节保留，不转换换行、不改绝对路径；入口通过原代码的 CACHE_MAP 在内存映射为 Linux 路径，继续检查原划分 SHA256。预训练权重沿用 A0 的官方 ImageNet 文件，校验原哈希前缀。不要误用 A0 best.pt。入口检查图片存在，但不逐图重算内容哈希；正式迁移时应另做传输完整性验收。

## 4. 检查、训练与恢复

在 Linux Bash 中设定真实持久路径。先只检查，不加载训练模型：

```bash
BASE=/YOUR_PERSISTENT_DIR/coin
ARGS=(--splits "$BASE/splits" --images "$BASE/images" --assets "$BASE/assets" --output "$BASE/runs/a1-cloud-001" --gpu 0)
python cloudstudio/run_a1.py "${ARGS[@]}"
```

以后确定正式实验和数据就绪后才执行：

```bash
python -u cloudstudio/run_a1.py "${ARGS[@]}" --execute
# 中断后的恢复：
python -u cloudstudio/run_a1.py "${ARGS[@]}" --execute --resume
```

可在已有 tmux 会话中运行以承受网页断开；空间停机仍会中断进程。动态检测 CUDA，无 GPU 时明确停止，不悄悄改成 CPU；`CUDA_VISIBLE_DEVICES` 限制单卡，Linux 文件锁拒绝同一用户以同一 GPU 参数重复启动。统一使用相同的 GPU ID 表示方式，勿绕过入口并发运行。

入口调用原 A1 train()，不改网络、损失、数据增强、batch、优化器和选模。训练结束会按**原流程执行验证及测试评估**。输出隔离在 runs 下；新运行要求空目录，恢复要求源码/配置指纹一致及 last.pt、best.pt 都存在。只支持本适配器产生的运行直接恢复；迁移旧 Windows checkpoint 前须另行核对配置和来源，不能冒充全新运行。

`模型/last.pt` 每完整轮次原子保存模型、optimizer、AMP scaler、scheduler、epoch、history、best、stale 和 RNG。恢复调用原逻辑，未提交轮次重跑；原 A1 依赖逐轮/逐样本重新设种子，未显式加载已存 RNG。不同 GPU/软件环境不承诺逐位复现。首次完整轮次完成前没有可恢复 last.pt。不要删除 best.pt；最终评价会读取它。

## 5. 后续同步与获取结果

本地审查、commit、push；云端在训练停止时执行 `git pull --ff-only`，然后重跑检查。不要在训练中 pull，更不要用 reset/clean 解决冲突。轻量指标、报告与可视化使用 results.py 导出后经 Git 回传，见 [操作说明](操作说明.md)；大型权重用 CloudStudio 文件面板下载 best.pt、last.pt；停机前确认它们在课程持久盘，并另行备份。较大结果可在运行完成后于仓库外打包：

```bash
tar -czf "$BASE/a1-cloud-001.tar.gz" -C "$BASE/runs" a1-cloud-001
```

腾讯官方参考：[算力切换](https://cloud.tencent.com/document/product/1039/131806)。课程具体入口与存储政策尚未验证。
