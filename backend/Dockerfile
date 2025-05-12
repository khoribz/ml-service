FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends locales git \
    && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

RUN update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

RUN git clone --depth 1 https://github.com/khoribz/ml-service.git /code

WORKDIR /code
COPY pyproject.toml ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

COPY . .

ENV PYTHONPATH=/code
CMD ["uvicorn", "src.ml_service.api.main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]
