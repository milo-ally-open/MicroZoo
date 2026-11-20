from pathlib import Path
from urllib.request import urlretrieve

url = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
output = Path(__file__).parent / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"

output.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(url, output)
