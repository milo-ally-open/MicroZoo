from pathlib import Path
from urllib.request import urlretrieve

url = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
output = Path(__file__).parent / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"

output.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(url, output)

print("预训练权重下载完成")
print("下载地址：{}".format(url))
print("权重文件：{}".format(output.resolve()))
print("文件大小：{:.2f} MiB".format(output.stat().st_size / 1024 ** 2))
