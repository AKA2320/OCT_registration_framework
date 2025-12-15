FROM python:3.12 AS builder

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
    pip install uv && \
    uv pip install torch==2.9.0 && \
    uv pip install maturin patchelf

COPY rust_bindings/ ./rust_bindings/
WORKDIR /rust_bindings
RUN export LIBTORCH_USE_PYTORCH=1 && \
    maturin develop --release

COPY pyproject.toml ./
COPY registration_scripts/ ./registration_scripts/
COPY utils/ ./utils/
RUN uv pip install "oct-proc[gui]"

# Runtime stage
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder .venv .venv
ENV PATH="/.venv/bin:$PATH"

COPY utils/ /app
COPY registration_scripts/ /app
COPY registration_script.py /app
COPY datapaths.yaml /app
COPY models/ /app
COPY pyside_gui.py /app
COPY pyproject.toml /app
COPY uv.lock /app
WORKDIR /app

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app && rm -rf /var/lib/apt/lists/*
USER app

CMD ["python", "pyside_gui.py"]