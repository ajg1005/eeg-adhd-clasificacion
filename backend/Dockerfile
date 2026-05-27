FROM python:3.12-slim

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts
COPY models ./models
COPY Figuras ./Figuras
COPY alembic.ini .
COPY alembic ./alembic

RUN mkdir -p /app/logs /app/tmp /app/results \
    && chown -R app:app /app/logs /app/tmp /app/results \
    && chmod -R 755 /app \
    && chmod -R 750 /app/logs /app/tmp /app/results

USER app

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
