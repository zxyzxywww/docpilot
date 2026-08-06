# MediDoc 镜像:纯 API 方案,无本地模型权重(镜像轻量,CPU 可跑)
FROM python:3.12-slim

WORKDIR /app

# 依赖与 pyproject.toml 对齐(纯 API,无 torch/模型)
COPY pyproject.toml ./
COPY src/ ./src/
COPY config.yaml ./

RUN pip install --no-cache-dir \
    openai pydantic python-dotenv pyyaml qdrant-client pypdf streamlit

# 数据目录(挂载卷,不入镜像);API key 走环境变量
RUN mkdir -p /app/data
EXPOSE 8501

CMD ["streamlit", "run", "src/app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
