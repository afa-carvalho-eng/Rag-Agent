# One image for both runtimes: the Fargate service runs the default command,
# the ingestion Lambda overrides the entrypoint with the Lambda runtime client.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[aws]"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "ragagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
