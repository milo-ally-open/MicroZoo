# Load Conda Pillow before system OpenCV to avoid their JPEG library load-order conflict on Jetson.
import importlib.metadata
import sys

from PIL import Image
import cv2
import torch


print("Python:", sys.version.replace("\n", " "))
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Pillow:", Image.__version__)
print("OpenCV:", cv2.__version__)

for package in ["transformers", "huggingface-hub", "safetensors", "scipy"]:
    try:
        print(package + ":", importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package + ": NOT INSTALLED")

try:
    import tensorrt as trt
    print("TensorRT:", trt.__version__)
except Exception as error:
    print("TensorRT import error:", repr(error))

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print("CUDA memory free/total: {:.0f}/{:.0f} MiB".format(
        free_bytes / (1024 * 1024), total_bytes / (1024 * 1024)
    ))
    sample = torch.ones((16,), device="cuda")
    torch.cuda.synchronize()
    print("CUDA tensor smoke check:", float(sample.sum().item()))
