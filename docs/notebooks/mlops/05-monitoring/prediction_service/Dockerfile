FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install pipenv
RUN pip install -U psutil

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock",  "./"]
COPY [ "model.bin",  "./"]
COPY [ "app.py",  "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

# https://stackoverflow.com/a/71092624/1091950
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "--timeout=600", "app.main:app" ]
