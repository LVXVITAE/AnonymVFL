# 基于 AnonymVFL 项目的 Dockerfile 示例
# 1. 选择基础镜像
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 复制项目文件
COPY . /app

# 4. 安装依赖
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 5. 设置环境变量（如有需要）
# ENV PYTHONUNBUFFERED=1

# 6. 默认启动命令（可根据实际情况修改）
CMD ["python", "AnonymVFL/run.py"]
