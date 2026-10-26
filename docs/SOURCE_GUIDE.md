# 源码阅读路径

报告主实验是 2026 年 10 月 25 日已完成的 direct PLS 四流程对照。

| 功能 | original_source 下的相对路径 |
|---|---|
| 主实验定义 训练 核验 测试 | lvi_prediction/run_dualroute_directpls_20261025.py |
| 已完成结果汇总与核查 | lvi_prediction/report_dualroute_studies_20261025.py |
| 40 次 bootstrap 方向一致性 | lvi_prediction/run_original_pls_direction_20261025.py 中 direction_stats 和 filtered |
| PLS 与正则化逻辑回归及临床预处理 | lvi_prediction/run_random120_five_models_20261023.py 及其 latent 依赖 |
| 训练部分内质量与重复特征筛选 | lvi_prediction/pure_habitat_robust_nonlinear_20261023.py |
| 有序特征及 MeanContrast 视图 | lvi_prediction/pure_habitat_latent_features_20261023.py |
| 轮廓变化与中位数汇总 | lvi_prediction/habitat_margin_features_20261024.py 和 run_habitat_margin_20261024.py |
| 患者内标准化 | lvi_prediction/normalization_habitat_features_20261012.py |
| T2 ADC 的患者内 K3 聚类 | lvi_prediction/localtexture_habitat_features_20261012.py |
| IVIM D f 的患者内联合聚类 | lvi_prediction/multiseq_habitat_features_20261012.py |
| 图像重采样与基础影像读取 | image_preprocessing.py 和 fold_locked_lvi_engine.py |

原入口的普通本地 import 经静态递归检查涉及 104 个源码模块，故完整保留这批模块及原目录层级；仅复制一两个入口会缺少依赖。某些旧实验模块由 import 链引入，保留它们不代表报告采用了那些旧实验。

源码快照未修改算法；docs/source_manifest.json 记录逐文件 SHA256。代码中保留了原工程的历史相对文件名与本机运行器路径，故需要原私有数据布局和历史缓存才能运行完整训练链。静态依赖闭包不保证涵盖所有运行时动态文件加载。

主入口支持 define/train/verify/test/report/seal，但它们是具有前置冻结状态和历史输入要求的研究阶段，不能在空仓库顺序执行就获得报告数值。默认公开验证使用 tools/verify_public_results.py；它不导入研究入口，不训练，也不接触患者数据。

请勿为让公开包运行而删除源程序的病例隔离、冻结文件或哈希保护检查。需要完整重训时，应在获授权的原工程副本中补齐输入并按照各阶段协议操作。
