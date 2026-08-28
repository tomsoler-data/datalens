FROM python:3.13-slim

LABEL org.opencontainers.image.title="DataLens API"
LABEL org.opencontainers.image.description="Local-first DataLens FastAPI runtime"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DATALENS_SQLITE_PATH=/app/var/datalens.sqlite3 \
    DATALENS_PREPARATION_ARTIFACT_STORE_PATH=/app/var/preparation/artifacts \
    DATALENS_ANALYSIS_ARTIFACT_STORE_PATH=/app/var/reporting/analysis_artifacts.json \
    DATALENS_REPORT_SELECTION_STORE_PATH=/app/var/reporting/report_selection.json \
    DATALENS_AI_TRACE_PATH=/app/var/observability/ai_traces.jsonl \
    DATALENS_RUNTIME_TRACE_PATH=/app/var/observability/runtime_requests.jsonl \
    DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT=/app/var/preparation/deletion-quarantine

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --requirement /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

RUN groupadd \
        --gid 10001 \
        datalens \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --create-home \
        --shell /usr/sbin/nologin \
        datalens \
    && mkdir -p \
        /app/var/preparation/artifacts \
        /app/var/preparation/deletion-quarantine \
        /app/var/reporting \
        /app/var/observability \
    && chown -R \
        datalens:datalens \
        /app

COPY --chown=datalens:datalens app ./app

USER datalens

EXPOSE 8000

HEALTHCHECK \
    --interval=10s \
    --timeout=5s \
    --start-period=5s \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]