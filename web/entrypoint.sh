#!/usr/bin/env sh
set -e

# variáveis esperadas: POSTGRES_HOST, POSTGRES_PORT (definidas no compose/.env)
: "${POSTGRES_HOST:=postgres}"
: "${POSTGRES_PORT:=5432}"

# espera até o Postgres ficar pronto
echo "Waiting for postgres at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
while ! nc -z ${POSTGRES_HOST} ${POSTGRES_PORT}; do
  sleep 1
done

echo "Postgres is up - running migrations"
python manage.py migrate --noinput

# opcional: criar superuser se variáveis estiverem presentes (não habilitado por padrão)
# python manage.py createsuperuser --noinput || true

# iniciar gunicorn
exec gunicorn web_project.wsgi:application --bind 0.0.0.0:8000 --workers 2
