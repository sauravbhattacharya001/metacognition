# --- Build stage ---
FROM python:3.14-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# --- Runtime stage ---
FROM python:3.14-slim

LABEL org.opencontainers.image.source="https://github.com/sauravbhattacharya001/metacognition"
LABEL org.opencontainers.image.description="Metacognitive Byzantine Fault Tolerance reference implementation"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY pyproject.toml .

USER app

ENTRYPOINT ["python", "-m", "mbft"]
