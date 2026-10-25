#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./setup_env.sh ENV_NAME [--root PATH]

Install the current ADLR RWM repository as a locked, uv-managed Isaac Lab 3.0
environment with the full Isaac Sim 6.2 / PhysX stack.

Arguments:
  ENV_NAME       Name of the isolated environment (letters, digits, . _ -).

Options:
  --root PATH    Parent directory for environments.
                 Default: $ADLR_RWM_ENV_ROOT or $HOME/venvs
  -h, --help     Show this message.

Run this script from any directory. The script installs the project containing this script.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

ENV_NAME="$1"
shift
ENV_ROOT="${ADLR_RWM_ENV_ROOT:-$HOME/venvs}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo '[ERROR] --root requires a nonempty path.' >&2
                exit 2
            fi
            ENV_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$ENV_NAME" == '.' || "$ENV_NAME" == '..' ]]; then
    echo "[ERROR] Invalid environment name: $ENV_NAME" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo '[ERROR] This setup currently targets Linux x86_64.' >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
PROJECT_VENV_LINK="$PROJECT_ROOT/.venv"

if ! command -v uv >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        echo '[ERROR] uv and curl are missing. Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2
        exit 1
    fi
    echo '[INFO] Installing uv for the current user...'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo '[ERROR] Could not locate uv after installation.' >&2
        exit 1
    fi
fi

mkdir -p -- "$ENV_ROOT"
ENV_ROOT="$(cd -- "$ENV_ROOT" && pwd -P)"
ENVIRONMENT_DIR="$ENV_ROOT/$ENV_NAME"
VENV_DIR="$ENVIRONMENT_DIR/.venv"

# Symlink to make uv run work
if [[ -e "$PROJECT_VENV_LINK" || -L "$PROJECT_VENV_LINK" ]]; then
    if [[ ! -L "$PROJECT_VENV_LINK" ]]; then
        echo "[ERROR] $PROJECT_VENV_LINK already exists and is not a symlink." >&2
        echo '[ERROR] Refusing to overwrite an existing environment.' >&2
        exit 1
    fi
    LINK_TARGET="$(readlink -- "$PROJECT_VENV_LINK")"
    if [[ "$LINK_TARGET" != "$VENV_DIR" ]]; then
        echo "[ERROR] $PROJECT_VENV_LINK points to a different environment: $LINK_TARGET" >&2
        echo '[ERROR] Use the matching ENV_NAME/--root or explicitly remove the old symlink.' >&2
        exit 1
    fi
fi

mkdir -p -- "$ENVIRONMENT_DIR"

echo "[INFO] Project: $PROJECT_ROOT"
echo "[INFO] Environment: $VENV_DIR"
echo "[INFO] uv: $(command -v uv)"

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export UV_PYTHON_PREFERENCE=only-managed

cd -- "$PROJECT_ROOT"

# Never silently rewrite an existing lockfile. On the very first invocation,
# uv produces one using the exact dependency requirements in pyproject.toml.
if [[ -f uv.lock ]]; then
    echo '[INFO] Synchronizing from the existing lockfile...'
    uv sync --locked --python 3.12
else
    echo '[INFO] No uv.lock found; resolving and creating the initial lockfile...'
    uv sync --python 3.12
    echo '[WARNING] Initial uv.lock created. Review and commit it to version control.' >&2
fi

if [[ ! -L "$PROJECT_VENV_LINK" ]]; then
    ln -s -- "$VENV_DIR" "$PROJECT_VENV_LINK"
fi

# Verify distribution metadata without launching an expensive graphical Kit app.
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import sys

assert sys.version_info[:2] == (3, 12), sys.version

for name in ("isaaclab", "isaacsim", "torch", "torchvision", "adlr-rwm"):
    print(f"[CHECK] {name}: {importlib.metadata.version(name)}")

import torch

print(f"[CHECK] PyTorch CUDA runtime: {torch.version.cuda}")
if not torch.version.cuda or not torch.version.cuda.startswith("13.0"):
    raise RuntimeError("Expected PyTorch built for CUDA 13.0.")
print(f"[CHECK] CUDA GPU accessible: {torch.cuda.is_available()}")
PY

cat <<EOF

--------------------------------- SETUP COMPLETE ---------------------------------
Main repo:  $PROJECT_ROOT
Environment:    $VENV_DIR

Normal commands work from the main repository because .venv is linked:
  cd "$PROJECT_ROOT"
  uv run --locked isaaclab list_envs --all --show_presets
  uv run --locked isaaclab random_agent --task Isaac-Lift-Franka-v0 --num_envs 4 physics=isaacsim_physx preset=cube
  uv run --locked python -m pytest tests
----------------------------------------------------------------------------------
EOF
