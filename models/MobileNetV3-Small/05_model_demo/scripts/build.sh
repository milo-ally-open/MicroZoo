#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$demo_dir" -B "$demo_dir/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$demo_dir/build" --parallel 2
