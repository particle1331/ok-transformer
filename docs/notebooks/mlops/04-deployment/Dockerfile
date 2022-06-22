FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock",  "./"]
COPY [ "setup.py", "pyproject.toml", "MANIFEST.in",  "./"]
COPY [ "ride_duration",  "./ride_duration"]
COPY [ "app",  "./app"]

RUN pipenv install --system --deploy

EXPOSE 9696

# https://stackoverflow.com/a/71092624/1091950
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "--timeout=600", "app.main:app" ]
