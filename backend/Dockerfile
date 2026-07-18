FROM ghcr.io/astral-sh/uv:0.11.29 AS uv

FROM python:3.13-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_BUILD=1 \
    UV_NO_PROGRESS=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-build --no-install-project

COPY backend ./backend
COPY scripts ./scripts
COPY models ./models
COPY Figuras ./Figuras
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

RUN mkdir -p /app/logs /app/tmp /app/results /app/data/uploads/datasets /app/models/trained \
    && chown -R app:app /app/logs /app/tmp /app/results /app/data /app/models/trained \
    && chmod -R 755 /app \
    && chmod -R 750 /app/logs /app/tmp /app/results /app/data /app/models/trained

USER app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
