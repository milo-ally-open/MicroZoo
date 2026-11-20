"""使用 trtexec 对 baseline 中的 TensorRT engine 做逐层剖析并生成图表。"""

import html
import json
import os
from pathlib import Path
import re
import subprocess

project_dir = Path(__file__).resolve().parent
engine_path = project_dir / "engine" / "mobilenet_v3_small_fp16.engine"
trtexec_path = Path("/usr/src/tensorrt/bin/trtexec")
profile_path = project_dir / "02-tensorrt-profile.log"
profile_json_path = project_dir / "02-tensorrt-profile.json"
layer_info_path = project_dir / "02-tensorrt-layer-info.json"
profile_html_path = project_dir / "02-tensorrt-profile.html"
warmup_ms = 5000
iterations = 5000
average_runs = 100
cpu_affinity = sorted(os.sched_getaffinity(0))

if not engine_path.is_file():
    raise FileNotFoundError("找不到 TensorRT engine：{}".format(engine_path))
if not trtexec_path.is_file():
    raise FileNotFoundError("找不到 trtexec：{}".format(trtexec_path))

command = [
    str(trtexec_path),
    "--loadEngine={}".format(engine_path),
    "--dumpProfile",
    "--separateProfileRun",
    "--exportProfile={}".format(profile_json_path),
    "--exportLayerInfo={}".format(layer_info_path),
    "--noDataTransfers",
    "--avgRuns={}".format(average_runs),
    "--warmUp={}".format(warmup_ms),
    "--duration=0",
    "--iterations={}".format(iterations),
]

print("剖析 TensorRT engine：{}".format(engine_path))
print("CPU affinity：{}".format(cpu_affinity))
print("预热 {} ms，运行 {} 次，按每 {} 次推理汇总；关闭 spin-wait，不计数据传输。".format(
    warmup_ms, iterations, average_runs,
))
print("完整输出同时保存到：{}".format(profile_path))
print("逐层数据保存到：{}".format(profile_json_path))
print("Engine 层信息保存到：{}".format(layer_info_path))
print("执行命令：{}".format(" ".join(command)))

with profile_path.open("w", encoding="utf-8") as profile_file:
    process = subprocess.Popen(
        command,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        profile_file.write(line)
        profile_file.flush()
    return_code = process.wait()

if return_code != 0:
    raise RuntimeError("trtexec profiling 失败，退出码：{}".format(return_code))

profile_data = json.loads(profile_json_path.read_text(encoding="utf-8"))
layer_info_data = json.loads(layer_info_path.read_text(encoding="utf-8"))
profile_log = profile_path.read_text(encoding="utf-8")
layers = [
    layer
    for layer in profile_data
    if isinstance(layer, dict) and "name" in layer and "averageMs" in layer
]
if not layers:
    raise RuntimeError("profile JSON 中没有可绘制的逐层耗时数据：{}".format(profile_json_path))

profile_iteration_count = next(
    (int(item["count"]) for item in profile_data if isinstance(item, dict) and "count" in item),
    0,
)
layer_names = layer_info_data.get("Layers", []) if isinstance(layer_info_data, dict) else []
if layer_names and all(isinstance(layer_name, str) for layer_name in layer_names):
    layer_info_note = (
        "当前 engine 构建时使用 layer_names_only，因此 layer info 只有层名和 binding 名；"
        "若要查看算子类型、张量形状、格式和 tactic，需要以 detailed profiling verbosity 重建 engine。"
    )
else:
    layer_info_note = "Engine layer info 已导出；具体字段由 engine 构建时保存的 profiling verbosity 决定。"

latency_match = re.search(
    r"GPU Compute Time:.*?mean = ([\d.]+) ms, median = ([\d.]+) ms, "
    r"percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
    r"percentile\(99%\) = ([\d.]+) ms",
    profile_log,
)
if latency_match:
    latency_values = [float(value) for value in latency_match.groups()]
    latency_metrics_html = "".join(
        "<div class='metric'><span>{}</span><strong>{:.3f} ms</strong></div>".format(label, value)
        for label, value in zip(("Mean", "Median", "P90", "P95", "P99"), latency_values)
    )
else:
    latency_metrics_html = "<p>trtexec GPU latency statistics are unavailable in the log.</p>"

stability_note = "逐层剖析会增加测量开销；请用独立延迟测试记录正式性能基准。"

layers.sort(key=lambda layer: float(layer["averageMs"]), reverse=True)
max_average_ms = max(float(layer["averageMs"]) for layer in layers)
total_average_ms = sum(float(layer["averageMs"]) for layer in layers)
chart_rows = []
for layer in layers:
    layer_name = html.escape(str(layer["name"]))
    average_ms = float(layer["averageMs"])
    median_ms = float(layer.get("medianMs", average_ms))
    percentage = float(layer.get("percentage", 0.0))
    bar_width = average_ms / max_average_ms * 100
    chart_rows.append(
        "<tr>"
        "<th scope='row'>{}</th>"
        "<td><div class='bar-track'><div class='bar' style='width:{:.2f}%'></div></div></td>"
        "<td class='number'>{:.5f} ms</td>"
        "<td class='number'>{:.5f} ms</td>"
        "<td class='number'>{:.2f}%</td>"
        "</tr>".format(layer_name, bar_width, average_ms, median_ms, percentage)
    )

layer_info_json = html.escape(json.dumps(layer_info_data, ensure_ascii=False, indent=2))
profile_html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TensorRT 逐层性能剖析</title>
  <style>
    body {{ margin: 2rem auto; padding: 0 1rem; max-width: 1200px; color: #17202a;
           font: 15px/1.5 system-ui, sans-serif; }}
    h1 {{ font-size: 1.6rem; }}
    p {{ color: #52616b; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ padding: .55rem .5rem; border-bottom: 1px solid #e5e9eb; text-align: left; }}
    thead th {{ position: sticky; top: 0; background: #fff; }}
    tbody th {{ width: 40%; font-weight: 500; overflow-wrap: anywhere; }}
    .bar-track {{ height: 1rem; background: #edf1f3; border-radius: .25rem; }}
    .bar {{ height: 100%; min-width: 2px; background: #1687a7; border-radius: .25rem; }}
    .number {{ width: 7rem; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
    details {{ margin-top: 2rem; }}
    .notice {{ padding: .8rem 1rem; color: #704214; background: #fff4df; border-radius: .4rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr)); gap: .6rem; margin: 1rem 0; }}
    .metric {{ padding: .7rem; background: #f5f7f8; border-radius: .4rem; }}
    .metric span {{ display: block; color: #52616b; font-size: .85rem; }}
    .metric strong {{ font-variant-numeric: tabular-nums; }}
    pre {{ overflow: auto; max-height: 36rem; padding: 1rem; background: #f5f7f8; border-radius: .4rem; }}
  </style>
</head>
<body>
  <h1>TensorRT 逐层性能剖析</h1>
  <p>Engine：{}；CPU affinity：{}；TensorRT profile 层数：{}；profile 迭代数：{}；逐层平均耗时总和：{:.5f} ms。</p>
  <div class="metrics">{}</div>
  <p class="notice">{}</p>
  <p>按平均耗时从高到低排列；柱长相对最慢层归一化。逐层耗时之和不一定等于端到端延迟。</p>
  <table>
    <thead><tr><th>TensorRT 层</th><th>平均耗时图</th><th class="number">平均耗时</th><th class="number">中位耗时</th><th class="number">占比</th></tr></thead>
    <tbody>{}</tbody>
  </table>
  <details>
    <summary>Engine 层信息（TensorRT exportLayerInfo）</summary>
    <p>{}</p>
    <pre>{}</pre>
  </details>
</body>
</html>
""".format(
    html.escape(str(engine_path)),
    cpu_affinity,
    len(layers),
    profile_iteration_count,
    total_average_ms,
    latency_metrics_html,
    html.escape(stability_note),
    "\n".join(chart_rows),
    html.escape(layer_info_note),
    layer_info_json,
)
profile_html_path.write_text(profile_html, encoding="utf-8")

print("TensorRT 逐层剖析完成。图表：{}".format(profile_html_path))
