FROM python:3.9.15-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /usr/app

COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

COPY ./src/ ./src/

CMD uvicorn src.main:app --port 8080 --host 0.0.0.0
