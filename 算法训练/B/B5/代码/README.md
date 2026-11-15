# B5 云端执行

此目录是待云端实测的实现，不代表训练已开始。只允许已授权腾讯云资源，禁止旧5090服务器。A10优先，单卡；训练前比较实际吞吐与价格，不同时开启多个付费实例。正式训练固定GPU。

模型：共享ConvNeXt-Tiny双面512输入；B0全局MLP；六个768维查询在双面512个token上归一化注意力；每类768维证据权重与偏置生成辅助logit，直接相加。新增9222参数。仅加权CE，无中心损失、无额外局部图分支。

## 网页终端准备

1. 在CloudStudio界面确认GPU型号、单价、余额、自动停止策略，以及`/workspace`在停止、重启、切换规格时是否保留。当前账户的持久化规则尚未核实，不能仅凭目录名推断。优先A10；不要为检查同时启动第二个付费实例。
2. 执行`nvidia-smi`、`findmnt -T /workspace`、`df -h /workspace`、`python3 --version`，确认无其他计算进程、磁盘至少10GB可用，Python3.10–3.13。把输出交回本机用于确定CUDA wheel和GPU策略。
3. 使用CloudStudio已有GitHub授权克隆`https://github.com/KK0341/coin-anomaly-research.git`。已有仓库先`git status --short`及`git log -3 --oneline`，保留云端改动；不要reset/clean。没有改动时fetch并以独立worktree检出本机交付的完整commit。
4. 通过网页文件上传，将本机B5/传输包里的`b5-assets.tar`和`archive_sha256.txt`上传至`/workspace`。数据不进GitHub。如果网页限制单文件大小，请先解决平台已有文件传输通道，不公开暴露数据下载端口。

```bash
cd /workspace
sha256sum -c archive_sha256.txt
mkdir b5-assets
tar -xf b5-assets.tar -C b5-assets
# 将REPO改成明确commit的检出目录
REPO=/workspace/coin-anomaly-research
CODE="$REPO/算法训练/B/B5/代码"
# 默认cu126要求驱动兼容CUDA12.6；驱动支持12.8时可用B5_CUDA_WHEEL=cu128。
bash "$CODE/setup.sh"
```

独立venv不启用system-site-packages，不修改系统包。torch2.9.1/torchvision0.24.1匹配来源：https://pytorch.org/get-started/previous-versions/ 。版本与B0尽量对齐。安装失败先检查Python及驱动，勿全局调整Pillow。

## 先测试，再正式执行

以下命令只做8次优化器更新，不读取测试标签进行选模。记录真实输入512、batch4、有效batch8的loss、梯度、查询参数更新、显存和吞吐。若OOM，只允许降batch到2或1，有效batch保持8。不同GPU逐个测试，避免并行计费。BF16支持时自动选择BF16，否则FP16+GradScaler。

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/b5-venv/bin/python "$CODE/train.py" \
  --data /workspace/b5-assets --out /workspace/B5-hardware-check --smoke
cat /workspace/B5-hardware-check/smoke.json
```

将测试结果、当前GPU价格与持久化规则交回后确定GPU。正式执行前设置已核对的commit及存储状态（不能在未确认时直接照填存储标志）：

```bash
export B5_EXPECTED_COMMIT=这里填写本机交付的完整commit
export B5_STORAGE_CONFIRMED=yes
nohup bash "$CODE/run.sh" > /workspace/B5-launch.log 2>&1 < /dev/null &
```

nohup仅防网页终端断开，不能抵抗平台自动停止。正式训练60轮上限、patience10，验证Macro-F1选模，每轮原子保存last.pt含optimizer/scheduler/scaler；训练中间退出从完整轮次恢复。恢复使用相同commit、GPU、batch、精度与数据：`nohup bash "$CODE/run.sh" --resume > /workspace/B5-resume.log 2>&1 < /dev/null &`。不要覆盖原日志。恢复按B0规则逐轮重置随机种子，不保证跨硬件逐位一致。

run.sh训练完成后执行验证消融、注意力、同GPU B0/B5推理计时，以及一次最终描述性测试。测试缓存已存在时不再推理；若TEST_STARTED.json存在但缓存缺失，会中止，需检查中断原因。禁止依据测试调参。

## 回传与停机

完成后网页下载`/workspace/B5-results.tar.gz`和对应`.sha256`至本机B5目录。云端B5保留完整best.pt和last.pt；回传包省略last.pt，不含大型临时缓存。用本机SHA256核对归档，再解压到B5新子目录，对照`result_sha256.json`逐文件验收。之后把最终成果归入B5目录，避免覆盖已有文件；未验收前不得删除云端副本。

确认成果回传后使用CloudStudio界面“停止”算力；不删除工作空间。核对界面计费状态及存储保留期限。官方算力切换与停止入口：https://cloud.tencent.com/document/product/1039/131806 。课程账户具体计费和持久化规则仍以实际界面为准。
