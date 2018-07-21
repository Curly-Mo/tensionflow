FROM python:3.6-slim

MAINTAINER Colin Fahy <colin@cfahy.com>

# install poetry
RUN pip install poetry

# create application directory
RUN mkdir /app
WORKDIR /app

COPY pyproject.* ./

# copy project directory
COPY tensionflow/ tensionflow/

# install dependencies:
RUN poetry install -n --no-dev

# Run everything with poetry env
ENTRYPOINT  [ "poetry", "run"]
