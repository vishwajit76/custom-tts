# CPU image (default). Fully offline at runtime: voices are baked in at build time.
FROM python:3.12-slim
WORKDIR /srv
COPY requirements*.txt ./
# more engines: --build-arg REQUIREMENTS=requirements-engines.txt --build-arg VOICES="hi_IN-rohan-medium supertonic kokoro"
ARG REQUIREMENTS=requirements.txt
RUN pip install --no-cache-dir -r $REQUIREMENTS
COPY app app
COPY voices/catalog.json voices/catalog.json
COPY scripts scripts
ARG VOICES="hi_IN-rohan-medium"
RUN python scripts/download_voices.py $VOICES && rm -rf /root/.cache/huggingface
RUN useradd --system --uid 10001 tts && chown -R tts /srv
USER tts
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
# one process per container: the model lives in-process; scale with replicas (docs/deployment in README)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "20", "--ws-ping-timeout", "20", "--timeout-graceful-shutdown", "10"]
