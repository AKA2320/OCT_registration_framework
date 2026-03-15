FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN python -m venv .venv
ENV PATH="/.venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install uv --no-cache-dir && \
    uv pip install torch==2.9.0 torchvision==0.24.0 --torch-backend cpu --no-cache-dir && \
    uv pip install maturin patchelf --no-cache-dir

COPY rust_bindings/ ./rust_bindings/
WORKDIR /rust_bindings
RUN export LIBTORCH_USE_PYTORCH=1 && \
    maturin build --release --auditwheel skip --out dist && \
    WHEEL_FILE=$(ls dist/*.whl | head -1) && \
    uv pip install "$WHEEL_FILE"

COPY pyproject.toml ./
COPY registration_scripts/ ./registration_scripts/
COPY utils/ ./utils/
RUN uv pip install . --no-cache-dir

# Runtime stage
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder .venv .venv
ENV PATH="/.venv/bin:$PATH"

WORKDIR /app

RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && rm -rf /var/lib/apt/lists/*
USER app

COPY --chown=app:app registration_script.py .
COPY --chown=app:app models/feature_detect_yolov12best.pt ./models/

CMD ["python", "registration_script.py"]
