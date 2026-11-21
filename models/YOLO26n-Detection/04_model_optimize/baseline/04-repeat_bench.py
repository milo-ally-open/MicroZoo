"""对固定 shape TensorRT engine 做三轮无 profiler 延迟测量。"""

import json
import re
import subprocess
from pathlib import Path

project_dir = Path(__file__).resolve().parent
engine_path = project_dir / "engine" / "yolo26n_fp16.engine"
output_dir = project_dir / "profiling" / "trtexec_repeats"
output_dir.mkdir(parents=True, exist_ok=True)
trtexec = Path("/usr/src/tensorrt/bin/trtexec")
if not trtexec.is_file():
    raise FileNotFoundError("TensorRT trtexec 未安装：{}".format(trtexec))

results = []
for mode in ["eager", "cuda_graph"]:
    for repeat in range(1, 4):
        log_path = output_dir / "{}-{}.log".format(mode, repeat)
        command = [str(trtexec), "--loadEngine={}".format(engine_path), "--warmUp=500", "--duration=3", "--noDataTransfers"]
        if mode == "cuda_graph":
            command.append("--useCudaGraph")
        with log_path.open("w") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        log_text = log_path.read_text()
        latency = re.search(r"Latency: .*mean = ([0-9.]+) ms, median = ([0-9.]+) ms, percentile\(90%\) = ([0-9.]+) ms, percentile\(95%\) = ([0-9.]+) ms, percentile\(99%\) = ([0-9.]+) ms", log_text)
        enqueue = re.search(r"Enqueue Time: .*mean = ([0-9.]+) ms, median = ([0-9.]+) ms, percentile\(90%\) = ([0-9.]+) ms, percentile\(95%\) = ([0-9.]+) ms, percentile\(99%\) = ([0-9.]+) ms", log_text)
        throughput = re.search(r"Throughput: ([0-9.]+) qps", log_text)
        variance = re.search(r"coefficient of variance = ([0-9.]+)%", log_text)
        if not latency or not enqueue or not throughput:
            raise RuntimeError("无法从 trtexec 日志解析性能结果：{}".format(log_path))
        result = {
            "mode": mode,
            "repeat": repeat,
            "log": log_path.name,
            "throughput_qps": float(throughput.group(1)),
            "latency_mean_ms": float(latency.group(1)),
            "latency_median_ms": float(latency.group(2)),
            "latency_p90_ms": float(latency.group(3)),
            "latency_p95_ms": float(latency.group(4)),
            "latency_p99_ms": float(latency.group(5)),
            "enqueue_mean_ms": float(enqueue.group(1)),
            "enqueue_median_ms": float(enqueue.group(2)),
            "enqueue_p90_ms": float(enqueue.group(3)),
            "enqueue_p95_ms": float(enqueue.group(4)),
            "enqueue_p99_ms": float(enqueue.group(5)),
            "gpu_latency_cv_percent": float(variance.group(1)) if variance else None,
        }
        results.append(result)
        print("{} run {}: median={:.3f} ms P95={:.3f} ms throughput={:.1f} qps CV={}".format(
            mode, repeat, result["latency_median_ms"], result["latency_p95_ms"], result["throughput_qps"],
            "{:.1f}%".format(result["gpu_latency_cv_percent"]) if result["gpu_latency_cv_percent"] is not None else "n/a",
        ), flush=True)

summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps(results, indent=2) + "\n")
print("三轮原始日志与汇总：{}".format(output_dir))
