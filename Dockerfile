

# ------------------------------------------------------------------------------
# knowai: Container image
# ------------------------------------------------------------------------------
# This Dockerfile builds a lightweight image that installs the *published*
# knowai package from PyPI.  At runtime you can either launch the included
# Streamlit chat UI or invoke the KnowAIAgent in your own Python script.
#
# Build (latest knowai):
#   docker build -t knowai:latest .
#
# Build a specific version (e.g. 0.3.1):
#   docker build --build-arg KNOWAI_VERSION=0.3.1 -t knowai:0.3.1 .
#
# Run Streamlit UI (default CMD):
#   docker run --rm -p 8501:8501 knowai:latest
#
# Run an interactive shell:
#   docker run -it --rm knowai:latest /bin/bash
#
# Execute your own Python that imports knowai:
#   docker run --rm -v "$(pwd)":/workspace -w /workspace knowai:latest \
#       python my_script.py
# ------------------------------------------------------------------------------

FROM python:3.13.3-slim AS base

# --------------------------------------------------------------------------- #
# Build‑time arguments and environment
# --------------------------------------------------------------------------- #
ARG KNOWAI_VERSION=latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    # Streamlit binds to 0.0.0.0 automatically when this is set
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# --------------------------------------------------------------------------- #
# Install knowai and optional extras
# --------------------------------------------------------------------------- #
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ "${KNOWAI_VERSION}" = "latest" ]; then \
        pip install knowai streamlit; \
    else \
        pip install "knowai==${KNOWAI_VERSION}" streamlit; \
    fi

# --------------------------------------------------------------------------- #
# Copy example Streamlit app (optional)
# --------------------------------------------------------------------------- #
# If the repo includes the UI file, copy it so the default CMD works out‑of‑box.
# Fall back to an internal knowai demo if the file is missing at build time.
COPY --chmod=644 app_chat_simple.py* /app/ 2>/dev/null || true

EXPOSE 8501

# --------------------------------------------------------------------------- #
# Entrypoint / default command
# --------------------------------------------------------------------------- #
# By default, launch the Streamlit chat UI.  Override CMD at `docker run`
# to execute other entrypoints (e.g., `python -m knowai.cli …`).
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["streamlit", "run", "app_chat_simple.py", "--server.port", "8501"]