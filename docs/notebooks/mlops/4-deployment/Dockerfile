FROM python:3.9.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock",  "./"]

COPY [ "setup.py", "pyproject.toml",  "./"]

COPY [ "ride_duration",  "./ride_duration"]

RUN pipenv install --system --deploy

COPY [ "app.py", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]
