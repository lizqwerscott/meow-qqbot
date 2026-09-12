#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frontend_dir="$project_root/core/webui/frontend"

if ! command -v pnpm >/dev/null 2>&1; then
  printf '%s\n' "未找到 pnpm，请先安装 pnpm。" >&2
  exit 1
fi

case "${1:-}" in
  "")
    pnpm --dir "$frontend_dir" install --frozen-lockfile
    ;;
  --no-install)
    ;;
  --help|-h)
    printf '%s\n' "用法: scripts/build_webui.sh [--no-install]"
    printf '%s\n' "默认先冻结安装前端依赖，再执行生产构建。"
    printf '%s\n' "--no-install 直接使用已有依赖执行构建。"
    exit 0
    ;;
  *)
    printf '%s\n' "未知参数: $1" >&2
    printf '%s\n' "用法: scripts/build_webui.sh [--no-install]" >&2
    exit 2
    ;;
esac

pnpm --dir "$frontend_dir" run build
