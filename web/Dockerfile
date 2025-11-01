# web/Dockerfile (corrigido)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# dependências de sistema necessárias
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libpq-dev netcat-openbsd git && \
    rm -rf /var/lib/apt/lists/*

# copiar requirements (agora relativo ao contexto ./web)
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# copia todo o conteúdo do contexto ./web para /app
COPY . /app/

# entrypoint (relativo ao contexto ./web)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PAYSIM_OUTPUT=/data/transactions.csv

EXPOSE 8000
CMD ["/app/entrypoint.sh"]
