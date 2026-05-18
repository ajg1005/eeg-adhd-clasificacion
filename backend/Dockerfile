FROM python:3.12-slim

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts

RUN mkdir -p /app/logs /app/tmp \
    && chown -R app:app /app/logs /app/tmp \
    && chmod -R 755 /app \
    && chmod -R 750 /app/logs /app/tmp

USER app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]