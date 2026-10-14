# Fixed environment for reproducing the CAP5150 project: Python 3.11 and pinned packages.
#   docker build -t cap5150-drift:1.0 .
#   docker run --rm -v "$PWD":/work -w /work cap5150-drift:1.0 make reproduce VENV=/opt/venv
FROM python:3.11.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# make for the experiment pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade "pip==24.2" \
    && /opt/venv/bin/pip install -r /tmp/requirements.lock \
    && touch /opt/venv/.installed

WORKDIR /work
CMD ["make", "help"]
