# 云端轻量结果

仅由 `cloudstudio/results.py export` 生成结果包：

`cloud_results/<实验名>/<北京时间 YYYY-MM-DD>/<HHMMSS-随机ID>/`

每包带 bundle.json（文件大小、SHA256、导出时间和导出时提交）；最多 20 个文件、8 张图、单文件 2 MiB、总计 20 MiB。仅允许显式选中的指标 JSON、报告 Markdown、PNG/JPEG 可视化。请带上 cloud_run.json 记录训练源码/配置指纹。不要提交数据集、逐样本预测、权重、凭据或含签名的下载链接。

本地执行 `python cloudstudio/results.py receive --pull`，复制到 `云端实验结果/<实验名>/<日期>/<运行ID>/`。重复接收一致包时跳过，不一致时报错，绝不覆盖。该本地接收目录已被 Git 忽略。

完整操作见 [CloudStudio 操作说明](../cloudstudio/操作说明.md)。
