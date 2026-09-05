#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SOURCE="$PROJECT_ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
METAL_BUILD="$PROJECT_ROOT/.build/mlxdlss-mlx-metallib"
METALLIB="$METAL_BUILD/mlx/backend/metal/kernels/mlx.metallib"
MIN_MACOS_VERSION="${MLXDLSS_MIN_MACOS_VERSION:-14.0}"
BUILD_JOBS="${MLXDLSS_BUILD_JOBS:-2}"

# Fast path: a prebuilt mlx.metallib (the `mlx` pip wheel of the SAME MLX version as the
# mlx-swift checkout ships one at site-packages/mlx/lib/mlx.metallib). Compiling the kernels
# needs the Metal compiler, which Command Line Tools do not include (full Xcode only).
if [[ -n "${MLXDLSS_METALLIB:-}" ]]; then
  if [[ ! -s "$MLXDLSS_METALLIB" ]]; then
    echo "MLXDLSS_METALLIB does not exist: $MLXDLSS_METALLIB" >&2
    exit 66
  fi
  if [[ "$#" -gt 0 ]]; then destinations=("$@"); else destinations=("$(swift build --package-path "$PROJECT_ROOT" --show-bin-path)"); fi
  for destination in "${destinations[@]}"; do
    mkdir -p "$destination"; cp "$MLXDLSS_METALLIB" "$destination/mlx.metallib"
  done
  echo "$MLXDLSS_METALLIB"
  exit 0
fi

for command_name in cmake ninja xcrun swift; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 69
  fi
done

if [[ ! -f "$MLX_SOURCE/CMakeLists.txt" ]]; then
  swift package --package-path "$PROJECT_ROOT" resolve
fi
if [[ ! -f "$MLX_SOURCE/CMakeLists.txt" ]]; then
  echo "MLX sources were not resolved at $MLX_SOURCE" >&2
  exit 66
fi

if [[ -z "${MLXDLSS_SKIP_TESTS:-}" ]]; then
  swift build --package-path "$PROJECT_ROOT" --build-tests --jobs "$BUILD_JOBS"
else
  swift build --package-path "$PROJECT_ROOT" --jobs "$BUILD_JOBS"     # MLXDLSS_SKIP_TESTS=1: no XCTest on Command Line Tools
fi

cmake \
  -S "$MLX_SOURCE" \
  -B "$METAL_BUILD" \
  -G Ninja \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_GGUF=OFF \
  -DMLX_BUILD_SAFETENSORS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_MACOS_VERSION"

cmake --build "$METAL_BUILD" --target mlx-metallib --parallel "$BUILD_JOBS"

if [[ ! -s "$METALLIB" ]]; then
  echo "MLX metallib was not produced at $METALLIB" >&2
  exit 70
fi

if [[ "$#" -gt 0 ]]; then
  destinations=("$@")
else
  bin_path="$(swift build --package-path "$PROJECT_ROOT" --show-bin-path)"
  destinations=("$bin_path")
  while IFS= read -r destination; do
    destinations+=("$destination")
  done < <(find "$bin_path" -type d -path '*.xctest/Contents/MacOS' -print)
fi

for destination in "${destinations[@]}"; do
  mkdir -p "$destination"
  cp "$METALLIB" "$destination/mlx.metallib"
done

echo "$METALLIB"
