FROM python:3.10-bullseye

# System deps for native builds + tini
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ make tini curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Prodigy wheel
COPY whl/prodigy-1.11.7-cp310-cp310-linux_x86_64.whl /tmp/prodigy-1.11.7-cp310-cp310-linux_x86_64.whl

ENV PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade "pip==23.2.1" "setuptools<70" "wheel<0.43"
RUN pip install "numpy==1.23.5"

# Install Prodigy and show /tmp for sanity
RUN ls -l /tmp && \
    pip install -v /tmp/prodigy-1.11.7-cp310-cp310-linux_x86_64.whl

# spaCy + model (Step 4 == spans.manual)
RUN pip install "spacy==3.4.4"
COPY whl/en_core_web_sm-3.4.0-py3-none-any.whl /tmp/
# sanity check size so we don't build with a truncated file
RUN python - <<'PY'
import os
p="/tmp/en_core_web_sm-3.4.0-py3-none-any.whl"
s=os.path.getsize(p)
print("Wheel size:", s, "bytes"); assert s>5_000_000, "Model wheel looks truncated"
PY
RUN pip install /tmp/en_core_web_sm-3.4.0-py3-none-any.whl && \
    python - <<'PY'
import spacy
print("spaCy:", spacy.__version__)
nlp = spacy.load("en_core_web_sm")
print("model OK:", nlp.meta["name"], nlp.meta["version"])
PY

# Install your recipes as a package so Prodigy can discover the entry point
COPY recipes_pkg/ /app/recipes_pkg/
RUN pip install /app/recipes_pkg

# Non-root runtime user + app dirs
RUN useradd -m prodigy && mkdir -p /app/data && chown -R prodigy:prodigy /app
USER prodigy
WORKDIR /app

# Data only (recipes are already installed as a package)
COPY --chown=prodigy:prodigy data/ /app/data/

ENV PRODIGY_HOST=0.0.0.0
ENV PRODIGY_PORT=8080
EXPOSE 8080

# No -F needed; recipe is registered
CMD ["prodigy","bullying-workflow","cyberbullying_dataset","/app/data/twitter_parsed_clean.txt","-l","txt"]
