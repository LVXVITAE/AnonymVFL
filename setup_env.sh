#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

ENV_NAME="${1:-sf-env}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

##############################
# 1. 检查 Python 3.10
##############################
log_info "检查 Python 3.10 ..."

PYTHON_BIN=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oP '3\.10')
        if [ "$ver" = "3.10" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if command -v conda &>/dev/null; then
    if conda search python=3.10 --info 2>/dev/null | grep -q '3.10'; then
        log_info "检测到 conda，将使用 conda 创建环境: $ENV_NAME"
        USE_CONDA=true
    else
        log_warn "conda 存在但未找到 python=3.10 包，回退到 venv"
        USE_CONDA=false
    fi
else
    USE_CONDA=false
fi

if [ "$USE_CONDA" = false ] && [ -z "$PYTHON_BIN" ]; then
    log_error "未找到 Python 3.10，请先安装 Python 3.10"
    echo "  Ubuntu/Debian: sudo apt-get install python3.10 python3.10-venv python3.10-dev"
    echo "  或使用 conda:   conda create -n $ENV_NAME python=3.10"
    exit 1
fi

##############################
# 2. 安装系统依赖
##############################
log_info "检查系统依赖 ..."

install_system_deps() {
    if command -v apt-get &>/dev/null; then
        local pkgs=()
        dpkg -s build-essential &>/dev/null || pkgs+=(build-essential)
        dpkg -s curl            &>/dev/null || pkgs+=(curl)
        dpkg -s git             &>/dev/null || pkgs+=(git)
        if [ ${#pkgs[@]} -gt 0 ]; then
            log_info "安装系统依赖: ${pkgs[*]}"
            sudo apt-get update -qq
            sudo apt-get install -y "${pkgs[@]}"
        else
            log_info "系统依赖已就绪"
        fi
    elif command -v yum &>/dev/null; then
        sudo yum install -y gcc gcc-c++ make curl git
    else
        log_warn "未检测到 apt-get/yum，请手动安装: build-essential curl git"
    fi
}

install_system_deps

##############################
# 3. 创建虚拟环境 & 安装 Python 依赖
##############################
DISTRIBUTED_MODE=false
if [ "${2:-}" = "--distributed" ] || [ "${2:-}" = "-d" ]; then
    DISTRIBUTED_MODE=true
fi

if [ "$USE_CONDA" = true ]; then
    if conda env list | grep -q "^${ENV_NAME}\s"; then
        log_warn "conda 环境 '$ENV_NAME' 已存在，跳过创建"
    else
        log_info "创建 conda 环境: $ENV_NAME (python=3.10)"
        conda create -n "$ENV_NAME" python=3.10 -y
    fi

    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
    PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"
else
    VENV_DIR="$SCRIPT_DIR/.venv"
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        log_info "创建虚拟环境: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
fi

log_info "升级 pip ..."
"$PIP" install --upgrade pip -q

LOCK_FILE="$SCRIPT_DIR/requirements-lock.txt"
if [ -f "$LOCK_FILE" ]; then
    log_info "安装锁定依赖 (requirements-lock.txt，与本机版本完全一致) ..."
    "$PIP" install -r "$LOCK_FILE"
elif [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    log_info "安装主要依赖 (requirements.txt) ..."
    "$PIP" install -r "$SCRIPT_DIR/requirements.txt"
    if [ "$DISTRIBUTED_MODE" = true ]; then
        log_info "分布式模式: 安装 Web UI 依赖 (web_ui/requirements.txt) ..."
        "$PIP" install -r "$SCRIPT_DIR/web_ui/requirements.txt"
    fi
else
    log_error "未找到 requirements-lock.txt 或 requirements.txt"
    exit 1
fi

##############################
# 4. 设置环境变量
##############################
log_info "设置环境变量 ..."

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SHELL_RC=""
case "$SHELL" in
    *bash) SHELL_RC="$HOME/.bashrc" ;;
    *zsh)  SHELL_RC="$HOME/.zshrc"  ;;
esac

ENV_VARS=(
    'export JAX_PLATFORMS=cpu'
    'export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"'
    'export OMP_NUM_THREADS=1'
    'export PYTHONUNBUFFERED=1'
)

if [ -n "$SHELL_RC" ] && ! grep -q "JAX_PLATFORMS=cpu" "$SHELL_RC" 2>/dev/null; then
    log_info "将环境变量写入 $SHELL_RC"
    {
        echo ""
        echo "# AnonymVFL 环境变量"
        for var in "${ENV_VARS[@]}"; do
            echo "$var"
        done
        if [ "$USE_CONDA" = true ]; then
            echo "# 激活 conda 环境: conda activate $ENV_NAME"
        else
            echo "# 激活虚拟环境: source $VENV_DIR/bin/activate"
        fi
    } >> "$SHELL_RC"
else
    for var in "${ENV_VARS[@]}"; do
        log_info "$var"
    done
fi

##############################
# 5. 验证
##############################
log_info "验证安装 ..."

"$PYTHON_BIN" -c "import secretflow; print('  secretflow', secretflow.__version__)" || log_error "secretflow 导入失败"
"$PYTHON_BIN" -c "import ray;       print('  ray',       ray.__version__)"         || log_error "ray 导入失败"
"$PYTHON_BIN" -c "import rbcl;      print('  rbcl',      rbcl.__version__)"        || log_error "rbcl 导入失败"
"$PYTHON_BIN" -c "import jax;       print('  jax',       jax.__version__)"         || log_error "jax 导入失败"
"$PYTHON_BIN" -c "import flask;     print('  flask',     flask.__version__)"       || true

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  安装完成!${NC}"
echo -e "${GREEN}============================================${NC}"

if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "  激活环境:  conda activate $ENV_NAME"
else
    echo ""
    echo "  激活环境:  source $VENV_DIR/bin/activate"
fi
echo "  Company 训练: cd company && bash startA.sh"
echo "  Partner 加入: cd partner && bash joinB.sh"
echo "  Web UI:       cd web_ui   && bash start_web_ui.sh"
echo ""
