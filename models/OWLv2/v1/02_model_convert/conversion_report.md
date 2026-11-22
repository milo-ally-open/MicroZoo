# OWLv2 v1 模型转换记录

## 目的

把 `01_model_analysis/checkpoints/owlv2-base-patch16-ensemble/model.safetensors` 中的权重加载到 Transformers 提供的 OWLv2 网络结构，然后导出 FP16 ONNX。`.safetensors` 是权重张量集合，不包含可单独运行的完整网络；改存为 `.pt` 只会改变文件容器，不会产生计算图，因此本阶段直接从 safetensors 加载。

模型来源和配置固定为 `google/owlv2-base-patch16-ensemble`，revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`。导出脚本使用本地快照，不访问模型网站。

## 图的划分

OWLv2 的图文匹配分数由视觉 patch 特征和文本查询特征共同计算。视觉特征同时供框回归和 objectness 预测使用。因此，按权重文件任意拆分既不对应网络执行边界，也不能得到可运行子模型。当前导出方案保留一张完整参考图，并按真实前向逻辑拆成三张可组合图：

| ONNX | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `owlv2_full_fp16.onnx` | `input_ids[Q,16]`、`attention_mask[Q,16]`、`pixel_values[1,3,960,960]` | `logits[1,3600,Q]`、`pred_boxes[1,3600,4]`、`objectness_logits[1,3600]` | 完整模型参考与单图部署备选 |
| `owlv2_text_fp16.onnx` | 文本 token 与 mask | 归一化 `text_embeds[Q,512]`、`query_mask[Q]` | 文本编码；提示词不变时可缓存结果 |
| `owlv2_vision_fp16.onnx` | `pixel_values[1,3,960,960]` | 图像类别特征、logit shift/scale、框、objectness | 视觉主干和仅依赖图像的检测分支 |
| `owlv2_match_fp16.onnx` | 视觉类别特征、shift/scale、文本特征与 mask | `logits[1,3600,Q]` | 将视觉与文本特征组合成类别分数 |

视觉图输入固定为模型默认 960×960，产生 60×60=3600 个 patch；文本查询数 `Q` 可变，token 长度固定为处理器产生的 16。当前部署按单张图像导出，不打开动态 batch 或动态图像尺寸。

三张分支图是可组合图，不保证比完整图更快。固定提示词时，文本图可只运行一次；若每帧都改变提示词，额外的图间调度可能抵消切分收益。后续要在 TensorRT 中验证图解析、精度和实际延迟，再选择完整图或分支组合。

## 运行

在仓库根目录执行：

```bash
conda run -n microzoo-owlv2-v1 python -m pip install -r models/OWLv2/v1/02_model_convert/requirements.txt
conda run -n microzoo-owlv2-v1 python models/OWLv2/v1/02_model_convert/01-export_onnx.py
conda run -n microzoo-owlv2-v1 python models/OWLv2/v1/02_model_convert/02-export_split_onnx.py
```

脚本从本地固定 checkpoint 和第一阶段示例图片读取输入。完整模型和切分图保存在 `02_model_convert/onnx/`；这些大文件由仓库 `.gitignore` 忽略，需要在目标设备重新生成。

## 数值核对

`02-export_split_onnx.py` 导出前会用同一 FP16 权重和输入，比较 Transformers 完整模型与文本、视觉、匹配三段组合的 logits、框和 objectness，容差为 `rtol=1e-3, atol=1e-3`。通过该检查只说明 PyTorch 子图重组与完整前向一致；ONNX Runtime / TensorRT 的数值和任务精度验证属于第三阶段。

## 当前状态

- [ ] 完整 FP16 ONNX 导出并检查。
- [ ] 三张 split FP16 ONNX 导出，并通过 PyTorch 分支重组数值核对。
- [ ] TensorRT FP16 engine 构建；需要先确认完整图和拆分图在目标 TensorRT parser 上的兼容性。
- [ ] ONNX/TensorRT 数值与任务精度验证留给 `03_model_validate`。
