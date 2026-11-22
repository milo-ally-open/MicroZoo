from pathlib import Path
from time import perf_counter

import numpy as np
import tensorrt as trt
import torch

project_dir = Path(__file__).resolve().parent
onnx_path = project_dir / "onnx" / "yolo26n_pruned.onnx"
calibration_path = project_dir / "calibration" / "yolo26n_calibration.npz"
calibration_cache_path = project_dir / "calibration" / "yolo26n_int8_entropy.cache"
engine_dir = project_dir / "engine"
engine_path = engine_dir / "yolo26n_pruned_int8.engine"
input_name = "images"
input_shape = (1, 3, 640, 640)
batch_size = input_shape[0]
expected_image_count = 24
workspace_size = 512 << 20

if not hasattr(trt, "IInt8EntropyCalibrator2"):
    raise RuntimeError(
        "当前 TensorRT 不再提供 IInt8EntropyCalibrator2；此示例针对 TensorRT 10.x。"
        "TensorRT 11 请改用显式 Q/DQ ONNX。"
    )
if not torch.cuda.is_available():
    raise RuntimeError("TensorRT 校准数据需要 CUDA；当前 PyTorch 未检测到 CUDA")
if not onnx_path.is_file():
    raise FileNotFoundError("先运行 02-prune-onnx.py：{}".format(onnx_path))
if not calibration_path.is_file():
    raise FileNotFoundError("先运行 03-prepare-calibration.py：{}".format(calibration_path))

with np.load(calibration_path, allow_pickle=False) as calibration_file:
    calibration_images = calibration_file["images"]
if calibration_images.shape != (expected_image_count,) + input_shape[1:]:
    raise RuntimeError("校准输入 shape 异常：{}".format(calibration_images.shape))
if calibration_images.dtype != np.float32:
    raise RuntimeError("TensorRT calibrator 需要 FP32 输入，实际为 {}".format(calibration_images.dtype))


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """把预处理好的 FP32 batch 交给 TensorRT，并缓存 TensorRT 生成的校准表。"""

    def __init__(self, images, cache_path, tensor_name, calibrator_batch_size):
        super().__init__()
        self.images = images
        self.cache_path = cache_path
        self.tensor_name = tensor_name
        self.batch_size = calibrator_batch_size
        self.image_index = 0
        # 复用一块 GPU batch buffer，避免校准期间每批都分配新显存。
        self.device_batch = torch.empty(
            (self.batch_size,) + tuple(images.shape[1:]),
            dtype=torch.float32,
            device="cuda",
        )

    def get_batch_size(self):
        # ONNX 是静态 batch=1，所以每次向 TensorRT 提供一张图。
        return self.batch_size

    def get_batch(self, names):
        # TensorRT 反复调用本方法取数据；返回值是 GPU 地址，不是 NumPy 数组。
        if list(names) != [self.tensor_name]:
            raise RuntimeError("校准器收到意外的输入名：{}".format(names))
        if self.image_index + self.batch_size > len(self.images):
            return None

        host_batch = np.ascontiguousarray(
            self.images[self.image_index:self.image_index + self.batch_size],
            dtype=np.float32,
        )
        # 用系统 PyTorch 把当前 batch 搬到 GPU；保留 tensor 引用，确保 TensorRT 使用指针期间内存有效。
        self.device_batch.copy_(torch.from_numpy(host_batch))
        torch.cuda.synchronize()
        self.image_index += self.batch_size
        print("TensorRT 校准读取图片：{}/{}".format(
            self.image_index, len(self.images)
        ), flush=True)
        return [int(self.device_batch.data_ptr())]

    def read_calibration_cache(self):
        # 缓存存在时 TensorRT 可跳过统计，直接用已有的校准表构建 engine。
        if self.cache_path.is_file():
            print("读取 TensorRT 校准缓存：{}".format(self.cache_path), flush=True)
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        # 第一次校准后，TensorRT 把计算出的量化参数交给这里保存。
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        print("保存 TensorRT 校准缓存：{}".format(self.cache_path), flush=True)


print("TensorRT 版本：{}".format(trt.__version__), flush=True)
print("剪枝后 FP32 ONNX：{} ({:.2f} MiB)".format(
    onnx_path, onnx_path.stat().st_size / 1024 ** 2
), flush=True)
print("EntropyCalibrator2：{} 张，batch size={}，输入 shape={}".format(
    len(calibration_images), batch_size, input_shape
), flush=True)
print("TensorRT calibration cache：{}".format(calibration_cache_path), flush=True)

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)
print("解析 ONNX...", flush=True)
if not parser.parse_from_file(str(onnx_path)):
    for index in range(parser.num_errors):
        print(parser.get_error(index))
    raise RuntimeError("TensorRT 无法解析剪枝 ONNX：{}".format(onnx_path))

if network.num_inputs != 1:
    raise RuntimeError("预期单输入网络，实际输入数为 {}".format(network.num_inputs))
network_input = network.get_input(0)
if network_input.name != input_name or tuple(network_input.shape) != input_shape:
    raise RuntimeError("ONNX 输入不匹配：name={} shape={}，预期 name={} shape={}".format(
        network_input.name, tuple(network_input.shape), input_name, input_shape
    ))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
calibrator = EntropyCalibrator(
    calibration_images,
    calibration_cache_path,
    input_name,
    batch_size,
)
config.int8_calibrator = calibrator

# TensorRT 在 build 阶段运行 FP32 校准前向、统计各激活直方图，
# 依据熵校准算法生成量化范围，然后继续构建 INT8 engine。
engine_dir.mkdir(parents=True, exist_ok=True)
print("开始校准并构建 TensorRT INT8 engine...", flush=True)
build_started_at = perf_counter()
serialized_engine = builder.build_serialized_network(network, config)
build_seconds = perf_counter() - build_started_at
if serialized_engine is None:
    raise RuntimeError("TensorRT INT8 engine 构建失败")

with open(engine_path, "wb") as engine_file:
    engine_file.write(serialized_engine)
print("校准与 engine 构建完成，用时 {:.1f} 秒".format(build_seconds), flush=True)
print("Engine 文件：{}".format(engine_path.resolve()), flush=True)
print("Engine 大小：{:.2f} MiB".format(engine_path.stat().st_size / 1024 ** 2), flush=True)
