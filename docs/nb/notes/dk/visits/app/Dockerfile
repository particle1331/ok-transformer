FROM python:3.9.15-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /usr/app

COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

COPY ./src/ ./src/

CMD gunicorn src.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8081
