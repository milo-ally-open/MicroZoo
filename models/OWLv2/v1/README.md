# OWLv2 v1

部署基线：`google/owlv2-base-patch16-ensemble`。

本版本按 MicroZoo 五阶段流程组织。第一阶段分析预训练模型、推理输入输出和部署可行性，不进行模型训练。当前阶段进展见 [`01_model_analysis/analysis_report.md`](01_model_analysis/analysis_report.md)，环境版本记录见 [`01_model_analysis/environment.md`](01_model_analysis/environment.md)。

1. `01_model_analysis`：模型结构、数据与部署可行性分析
2. `02_model_convert`：导出 ONNX 并构建目标运行时模型
3. `03_model_validate`：数值、任务精度、内存与兼容性验证
4. `04_model_optimize`：目标设备性能剖析与优化
5. `05_model_demo`：端侧演示程序
