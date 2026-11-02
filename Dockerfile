FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY ml ./ml
RUN pip install --no-cache-dir '.[db,dashboard]' && useradd --create-home guard
COPY alembic.ini ./
COPY migrations ./migrations
COPY configs ./configs
COPY dashboard ./dashboard
USER guard
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
