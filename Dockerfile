# 多方安全计算 - 移动联邦学习项目
FROM python:3.10
# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 验证 Ray Python API（Ray CLI 在容器中有 bug，跳过验证）
RUN python3 -c "import ray; print('✅ Ray import OK:', ray.__version__)"

# 复制项目文件
COPY company/ ./company/
COPY partner/ ./partner/
COPY trans/ ./trans/
# COPY test/ ./test/  # test 目录在 .dockerignore 中已排除
COPY web_ui/ ./web_ui/
COPY data_full.csv .

# 创建模型目录
RUN mkdir -p /app/company/models /app/partner/models

# 设置环境变量
ENV JAX_PLATFORMS=cpu \
    XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false" \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# 暴露端口
# 9394, 9395, 9396: SPU 节点端口
# 8080: Web UI 端口
# 6379: Ray dashboard 端口
EXPOSE 8080 9394 9395 9396 6379

# 默认命令（可以被 Helm Chart 覆盖）
CMD ["python", "company/truerun.py", "--mode", "multi_distributed", \
    "--company_spu_addr", "0.0.0.0:9394", \
    "--partner_spu_addr", "partner:9395", \
    "--coordinator_spu_addr", "coordinator:9396", \
    "--ray_head_addr", "localhost:6379"]

