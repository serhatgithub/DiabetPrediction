FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY diabetes.csv .
COPY src ./src

CMD ["/opt/venv/bin/python", "src/train.py"]
