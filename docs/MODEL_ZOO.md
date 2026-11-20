# MicroZoo 计划

| 编号 | 主要任务 | 预训练基线 / 具体模型 | 可能迁移到的垂直场景 | 官方资料 |
|---|---|---|---|---|
| **C-01** | 图像分类 | MobileNetV3-Small，Torchvision ImageNet-1K 预训练权重 | 产线零件或包装分类；可回收物分类；设备指示灯状态分类 | [Torchvision 模型文档](https://docs.pytorch.org/vision/main/models/mobilenetv3.html) · [官方实现](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py) |
| **C-02** | 图像分类 | YOLO26n-cls，ImageNet 预训练 | 产品外观等级分类；工具与耗材分类；检测后对局部区域做状态分类 | [Ultralytics 分类任务与预训练权重](https://docs.ultralytics.com/tasks/classify) |
| **D-01** | 目标检测 | YOLO26n，COCO 预训练 | 仓库盘点；人员、安全帽、工具和设备检测；移动机器人障碍物或遗留物发现 | [Ultralytics 检测任务与预训练权重](https://docs.ultralytics.com/tasks/detect) |
| **D-02** | 目标检测 | RT-DETRv2-S（R18），COCO 预训练，约 20M 参数 | 复杂背景下的设备与零件检测；与 YOLO26n 对比小物体、遮挡和速度表现 | [RT-DETR 官方仓库与基准表](https://github.com/lyuwenyu/RT-DETR) · [官方权重入口](https://github.com/lyuwenyu/RT-DETR/blob/main/hubconf.py) |
| **S-01** | 实例分割 | YOLO26n-seg，COCO 预训练 | 传送带重叠零件；包裹和货品实例计数；为后续分拣提供实例区域 | [Ultralytics 实例分割任务与预训练权重](https://docs.ultralytics.com/tasks/segment) |
| **S-02** | 实例分割 | RTMDet-Ins-tiny，COCO 预训练 | 小型、密集零件的实例分割；作为边缘端实例分割对照 | [MMDetection 官方模型库、配置与权重](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md) |
| **P-01** | 2D 姿态估计 | RTMPose-tiny，COCO 人体关键点预训练 | 工人不安全姿势与人体工效分析；协作区动作状态；跌倒预警线索 | [MMPose 官方 COCO 配置与权重记录](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.yml) |
| **P-02** | 2D 姿态估计 | YOLO26n-pose，COCO Keypoints 预训练 | 驾驶员或操作员头部姿态；运动动作计数；疲劳驾驶系统中的姿态线索 | [Ultralytics 姿态任务与预训练权重](https://docs.ultralytics.com/tasks/pose) |
| **Z-01** | 单目深度估计 | FastDepth，NYU Depth V2 基线 | 室内移动机器人对通道、货架和障碍物的粗略空间判断 | [FastDepth 官方项目](https://fastdepth.mit.edu/) · [官方代码与模型说明](https://github.com/dwofk/fast-depth) |
| **Z-02** | 单目深度估计 | YOLO26n-depth，多数据集预训练 | RGB 单摄像头下的近似深度图；狭窄通道、台阶与货架空间辅助判断 | [Ultralytics 深度估计任务与预训练权重](https://docs.ultralytics.com/tasks/depth) |
| **Z-03** | 单目深度估计 | Depth Anything V2-Small，预训练相对深度模型 | 室内外场景的前后关系、遮挡和空间层次分析；为路径理解提供视觉先验 | [Depth Anything V2 官方代码与模型列表](https://github.com/DepthAnything/Depth-Anything-V2) |
| **T-01** | 视频跟踪 | ByteTrack + D-01 检测器；ByteTrack 是跟踪算法，不单独提供检测器权重 | 人员、车辆、包裹跨帧轨迹；客流计数；物品进出区域和停留时间分析 | [ByteTrack 官方仓库](https://github.com/FoundationVision/ByteTrack) · [D-01 检测器](https://docs.ultralytics.com/tasks/detect) |
| **T-02** | 视频跟踪 / ReID | OSNet-x0.25，行人 ReID 基线（Market1501） | 多摄像头间的人员身份关联；员工跨区域移动分析。该基线针对行人，不是通用物品 ReID | [Torchreid 官方模型库](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md) |
| **O-01** | 开放语义检测 | OWLv2-base-patch16-ensemble，零样本文本查询检测 | 用文字查询低频工具、零件或物品；开放词表仓库盘点与异常候选初筛 | [Google 官方 Hugging Face 模型卡](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| **O-02** | 开放语义分割 | SAM3；实施时采用官方最新 SAM3.1 checkpoint | 按文本或视觉提示分割指定物品；为质检、盘点和机器人视觉流程生成目标掩码 | [Meta SAM3 官方仓库](https://github.com/facebookresearch/sam3) · [SAM3.1 发布说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) · [官方 SAM3.1 Hugging Face 权重](https://huggingface.co/facebook/sam3.1) |
| **V-01** | 视觉语言模型 | Qwen2-VL-2B-Instruct | 设备面板问答；巡检图像描述；理解“找出缺失的连接器”等自然语言视觉问题 | [Qwen 官方 Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| **V-02** | 视觉语言模型 / Orin 友好 | Gemma 4 E2B-it；Jetson AI Lab 提供 Orin Nano 8GB 的 llama.cpp 部署配置 | Orin Nano 本地视觉问答；设备读表与检查清单辅助；机器人任务的场景理解 | [Google 官方 Hugging Face 模型卡](https://huggingface.co/google/gemma-4-E2B-it) · [Jetson AI Lab 部署教程](https://www.jetson-ai-lab.com/tutorials/gemma4-on-jetson/) |
| **E-01** | 具身策略 / 模仿学习 | ACT；基线 `lerobot/act_aloha_sim_transfer_cube_human` | 仿真中的重复搬运、物件放置和装配动作；迁移到实体机器人时用对应机械臂数据适配 | [LeRobot 官方 ACT 模型卡](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) |
| **E-02** | 具身 VLA | SmolVLA，`lerobot/smolvla_base` | 桌面机器人按语言指令执行分拣、物件放置或简单装配；用自有示教数据微调 | [LeRobot 官方 SmolVLA 模型卡](https://huggingface.co/lerobot/smolvla_base) · [LeRobot 文档](https://huggingface.co/docs/lerobot/v0.4.4/smolvla) |
| **E-03** | 具身 VLA | TurboVLA，`H-EmbodVis/TurboVLA` | LIBERO、RoboTwin 仿真任务；多任务操作策略的对照与迁移实验 | [官方 Hugging Face 模型卡与权重](https://huggingface.co/H-EmbodVis/TurboVLA) · [官方代码仓库](https://github.com/H-EmbodVis/TurboVLA) |
