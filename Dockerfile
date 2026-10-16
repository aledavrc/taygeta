# syntax=docker/dockerfile:1
#
# Main Taygeta application image (section 11.2).
# Multi-stage with uv: a reproducible virtual environment is built and copied to
# a runtime image with Pandoc and Typst CLI installed and verified.
# It does not include model weights, database, documents or figures.

# ---------------------------------------------------------------------------
# Stage 1: dependency resolution and installation with uv
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer cached against uv.lock / pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Code and non-secret configuration.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Installs the project inside the virtual environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ARG TYPST_VERSION=0.15.1
ARG PANDOC_VERSION=3.12.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# System dependencies: curl (healthchecks/downloads), ca-certificates and xz.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/*

# Pandoc CLI (official .deb package of the given release).
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    pkg="pandoc-${PANDOC_VERSION}-1-${arch}.deb"; \
    curl -fsSL -o /tmp/pandoc.deb \
      "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/${pkg}"; \
    dpkg -i /tmp/pandoc.deb; \
    rm -f /tmp/pandoc.deb; \
    pandoc --version

# Typst CLI (official static musl binary).
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) target="x86_64-unknown-linux-musl" ;; \
      arm64) target="aarch64-unknown-linux-musl" ;; \
      *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/typst.tar.xz \
      "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-${target}.tar.xz"; \
    tar -xJf /tmp/typst.tar.xz -C /tmp; \
    install -m 0755 "/tmp/typst-${target}/typst" /usr/local/bin/typst; \
    rm -rf /tmp/typst.tar.xz "/tmp/typst-${target}"; \
    typst --version

# Non-root user.
RUN groupadd --system --gid 1001 taygeta \
 && useradd --system --uid 1001 --gid taygeta --create-home taygeta

WORKDIR /app

# Already resolved virtual environment + code.
COPY --from=builder --chown=taygeta:taygeta /app/.venv /app/.venv
COPY --chown=taygeta:taygeta pyproject.toml uv.lock ./
COPY --chown=taygeta:taygeta src ./src

# Runtime directories (overridden by persistent volumes).
RUN mkdir -p /app/data /app/uploads /app/downloads \
 && chown -R taygeta:taygeta /app

USER taygeta

# 8000: MCP/download HTTP. 8501: NiceGUI HITL UI (service `ui`, same image).
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# MCP server start command (script installed by the project).
# The UI starts with: taygeta-ui (service `ui` in docker-compose.yml).
CMD ["taygeta-server"]
