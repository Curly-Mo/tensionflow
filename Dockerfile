FROM python:3.6

MAINTAINER Colin Fahy <colin@cfahy.com>

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

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
