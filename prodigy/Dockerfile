FROM python:3.10-slim

# tini + minimal build deps (gcc only if you truly need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini gcc \
    && rm -rf /var/lib/apt/lists/*

# If this is an ARM wheel, build for ARM; otherwise swap in an amd64 wheel
COPY whl/prodigy-1.11.7-cp310-cp310-linux_aarch64.whl /tmp/prodigy-1.11.7-cp310-cp310-linux_aarch64.whl

# Keep versions compatible with spaCy 3.4.x and Prodigy 1.11.x
RUN python -m pip install --no-cache-dir "pip==23.2.1" "setuptools<70" "wheel<0.43" \
 && pip install --no-cache-dir "numpy<1.24" \
 && pip install --no-cache-dir /tmp/prodigy-1.11.7-cp310-cp310-linux_aarch64.whl \
 && pip install --no-cache-dir "spacy==3.4.4" \
 && python -m spacy download en_core_web_sm

# Non-root user + app dir
RUN useradd -m prodigy && mkdir -p /app/data && chown -R prodigy:prodigy /app
USER prodigy
WORKDIR /app

# Bring in your data (optional; you can also mount at runtime)
COPY --chown=prodigy:prodigy data/ /app/data/

# App Runner/ECS expect a stable port and 0.0.0.0 binding
ENV PRODIGY_HOST=0.0.0.0
ENV PRODIGY_PORT=8080
EXPOSE 8080

# Make the run-time configurable without rebuilds
ENV RECIPE="spans.manual" \
    DATASET="cyberbullying_dataset" \
    MODEL="en_core_web_sm" \
    DATA_FILE="/app/data/twitter_parsed_clean.txt" \
    LABELS="CYBERBULLYING,NOT_CYBERBULLYING,BYSTANDER,VICTIM,BULLY,VISUAL,VERBAL,EXCLUSION,OTHER"

# Copy a tiny entrypoint that execs Prodigy
COPY --chown=prodigy:prodigy entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini","--","/app/entrypoint.sh"]
