#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./setup_env.sh ENV_NAME [--root PATH] [--experiments-root PATH]
       ./setup_env.sh [--root PATH] [--experiments-root PATH]

Create an independent Git clone of the committed source code and a separate Isaac Lab venv.
This makes it easy to run ablations or other experiments in a new source tree and allows
tracking all the changes for each experiment in one shared dir without the Isaac Lab deps.

From within an existing experiment, specifying its own ENV_NAME (or omitting it)
recreates/synchronizes its environment without cloning again.

Arguments:
  ENV_NAME                Experiment name (letters, digits, '.', '_' or '-').
                          Optional only within an existing experiment.

Options:
  --root PATH             Parent directory for disposable virtual environments.
                          Default: $ADLR_RWM_ENV_ROOT or $HOME/venvs
  --experiments-root PATH Parent directory for experiment source trees.
                          Default: $ADLR_RWM_EXPERIMENTS_ROOT or
                                   $HOME/Documents/adlr-experiments
  -h, --help              Show this help.

Example:
  ./setup_env.sh env_ablation_test
  cd ~/Documents/adlr-experiments/env_ablation_test
  uv run --locked --no-sync python -m pytest tests

Cloning requires a clean Git working tree and committed uv.lock.
Only committed files are cloned; raw data and virtual environments are not copied.
EOF
}

ENV_NAME=""
ENV_ROOT="${ADLR_RWM_ENV_ROOT:-$HOME/venvs}"
EXPERIMENTS_ROOT="${ADLR_RWM_EXPERIMENTS_ROOT:-$HOME/Documents/adlr-experiments}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root|--experiments-root)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "[ERROR] $1 requires a nonempty path." >&2
                exit 2
            fi
            if [[ "$1" == "--root" ]]; then
                ENV_ROOT="$2"
            else
                EXPERIMENTS_ROOT="$2"
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$ENV_NAME" ]]; then
                echo "[ERROR] Unexpected extra argument: $1" >&2
                exit 2
            fi
            ENV_NAME="$1"
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
mkdir -p -- "$ENV_ROOT" "$EXPERIMENTS_ROOT"

ENV_ROOT="$(cd -- "$ENV_ROOT" && pwd -P)"
EXPERIMENTS_ROOT="$(cd -- "$EXPERIMENTS_ROOT" && pwd -P)"

# If no name is given, infer it from an existing experiment directory.
if [[ -z "$ENV_NAME" ]]; then
    if [[ "$(dirname -- "$PROJECT_ROOT")" != "$EXPERIMENTS_ROOT" ]]; then
        echo '[ERROR] ENV_NAME is required when creating a new experiment.' >&2
        usage >&2
        exit 2
    fi
    ENV_NAME="$(basename -- "$PROJECT_ROOT")"
fi

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$ENV_NAME" == '.' || "$ENV_NAME" == '..' ]]; then
    echo "[ERROR] Invalid experiment name: $ENV_NAME" >&2
    exit 2
fi

EXPERIMENT_SOURCE_DIR="$EXPERIMENTS_ROOT/$ENV_NAME"
ENVIRONMENT_DIR="$ENV_ROOT/$ENV_NAME"
VENV_DIR="$ENVIRONMENT_DIR/.venv"
OWNER_FILE="$ENVIRONMENT_DIR/.adlr_rwm_project_root"

# Protect existing environments against accidental reuse by another source tree.
if [[ "$ENVIRONMENT_DIR" == "$EXPERIMENT_SOURCE_DIR" ]]; then
    echo '[ERROR] Environment and experiment directories must have different roots.' >&2
    exit 1
fi

if [[ -e "$ENVIRONMENT_DIR" && ! -d "$ENVIRONMENT_DIR" ]]; then
    echo "[ERROR] Environment path is not a directory: $ENVIRONMENT_DIR" >&2
    exit 1
fi

if [[ -d "$ENVIRONMENT_DIR" ]]; then
    if [[ -f "$OWNER_FILE" ]]; then
        IFS= read -r EXISTING_PROJECT_ROOT < "$OWNER_FILE" || true

        if [[ "$EXISTING_PROJECT_ROOT" != "$EXPERIMENT_SOURCE_DIR" ]]; then
            echo "[ERROR] Environment '$ENV_NAME' belongs to a different source tree:" >&2
            echo "        $EXISTING_PROJECT_ROOT" >&2
            exit 1
        fi
    elif [[ -n "$(find "$ENVIRONMENT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "[ERROR] Nonempty environment directory lacks an ownership marker:" >&2
        echo "        $ENVIRONMENT_DIR" >&2
        exit 1
    fi
fi

# Create a new independent source tree when invoked outside the target experiment.
if [[ "$PROJECT_ROOT" != "$EXPERIMENT_SOURCE_DIR" ]]; then

    if [[ -e "$EXPERIMENT_SOURCE_DIR" || -L "$EXPERIMENT_SOURCE_DIR" ]]; then
        echo "[ERROR] Experiment directory already exists: $EXPERIMENT_SOURCE_DIR" >&2
        echo "        Run setup_env.sh from that experiment to reinstall it." >&2
        exit 1
    fi

    echo "[INFO] Cloning committed source into $EXPERIMENT_SOURCE_DIR"

    # --no-hardlinks makes Git object storage independent of the source repo.
    # This does not affect uv's separate dependency-file hardlinking.
    git clone --no-hardlinks -- "$PROJECT_ROOT" "$EXPERIMENT_SOURCE_DIR"

    # Give the experiment its own development branch.
    git -C "$EXPERIMENT_SOURCE_DIR" switch -c "experiment/${ENV_NAME//./-}"

    PROJECT_ROOT="$EXPERIMENT_SOURCE_DIR"
else
    echo "[INFO] Using existing experiment: $PROJECT_ROOT"
fi

PROJECT_VENV_LINK="$PROJECT_ROOT/.venv"

# Never overwrite an actual directory/file located at .venv.
if [[ -e "$PROJECT_VENV_LINK" && ! -L "$PROJECT_VENV_LINK" ]]; then
    echo "[ERROR] Refusing to overwrite an existing .venv directory or file:" >&2
    echo "        $PROJECT_VENV_LINK" >&2
    exit 1
fi

# Install uv if it is not available.
if ! command -v uv >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        echo '[ERROR] Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
        exit 1
    fi

    echo '[INFO] Installing uv for the current user...'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo '[ERROR] uv is unavailable after installation.' >&2
        exit 1
    fi
fi

mkdir -p -- "$ENVIRONMENT_DIR"
printf '%s\n' "$PROJECT_ROOT" > "$OWNER_FILE"

# Hardlinks require the uv cache and venv to be on the same filesystem.
CACHE_DIR="$(uv cache dir)"
mkdir -p -- "$CACHE_DIR"

printf '[INFO] Source tree: %s\n' "$PROJECT_ROOT"
printf '[INFO] Environment: %s\n' "$VENV_DIR"
printf '[INFO] uv cache: %s\n' "$CACHE_DIR"

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export UV_PYTHON_PREFERENCE=only-managed

cd -- "$PROJECT_ROOT"

# Sync dependencies using the experiment's own lockfile.
if [[ -f uv.lock ]]; then
    echo '[INFO] Synchronizing from uv.lock...'
    uv sync --locked --python 3.12
else
    echo '[WARNING] No uv.lock found; resolving dependencies.' >&2
    uv sync --python 3.12
    echo '[WARNING] Review and commit the new uv.lock.' >&2
fi

# Verify dependencies and that the editable package points to THIS experiment.
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Expected Python 3.12, got {sys.version}")

for distribution in ("isaaclab", "isaacsim", "torch", "torchvision", "adlr-rwm"):
    print(f"[CHECK] {distribution}: {importlib.metadata.version(distribution)}")

import torch

print(f"[CHECK] PyTorch CUDA runtime: {torch.version.cuda}")

if torch.version.cuda is None or not torch.version.cuda.startswith("13.0"):
    raise RuntimeError("Expected a PyTorch build for CUDA 13.0")

print(f"[CHECK] CUDA GPU available: {torch.cuda.is_available()}")

spec = importlib.util.find_spec("adlr_rwm")
expected = (Path.cwd() / "src" / "adlr_rwm").resolve()

if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Cannot locate the installed adlr_rwm package")

actual = [
    Path(location).resolve()
    for location in spec.submodule_search_locations
]

print(f"[CHECK] adlr_rwm source: {', '.join(map(str, actual))}")

if expected not in actual:
    raise RuntimeError(
        f"Wrong editable package source! Expected {expected}, found {actual}"
    )
PY

# Create or repair the project-local convenience symlink only after success.
# The actual environment and all dependencies remain in ~/venvs.
if [[ ! -L "$PROJECT_VENV_LINK" ]] || [[ "$(readlink -- "$PROJECT_VENV_LINK")" != "$VENV_DIR" ]]; then
    TEMP_LINK="$PROJECT_ROOT/.venv.tmp.$$"

    trap 'rm -f -- "$TEMP_LINK"' EXIT
    ln -s -- "$VENV_DIR" "$TEMP_LINK"
    mv -Tf -- "$TEMP_LINK" "$PROJECT_VENV_LINK"
    trap - EXIT
fi

cat <<EOF

------------------------------- SETUP COMPLETE -------------------------------
Experiment:   $PROJECT_ROOT
Environment:  $VENV_DIR
Git commit:   $(git -C "$PROJECT_ROOT" rev-parse --short HEAD)

Enter the experiment:
  cd "$PROJECT_ROOT"

To recreate this environment after deleting ~/venvs, run:
  ./setup_env.sh

To create another experiment, run setup_env.sh with a different ENV_NAME
from a committed source repository.
-------------------------------------------------------------------------------
EOF
