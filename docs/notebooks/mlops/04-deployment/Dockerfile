FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install poetry

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock",  "./"]

COPY [ "ride_duration",  "./ride_duration"]

COPY [ "app",  "./app"]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app.main:app" ]
