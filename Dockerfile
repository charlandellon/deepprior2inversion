FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=4 \
    DEVITO_LANGUAGE=openmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make git libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-windows.txt /tmp/requirements.txt
RUN python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r /tmp/requirements.txt \
    && python -m pip check

WORKDIR /workspace
COPY setup.py README.md ./
COPY dp4inversion ./dp4inversion
RUN python -m pip install --no-deps -e .

CMD ["bash"]
