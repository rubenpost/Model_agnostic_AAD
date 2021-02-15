#pip freeze > requirements.txt


FROM python:3.7-slim-buster

WORKDIR /app
ADD . /app

ENV ACCEPT_EULA=Y
RUN apt-get --assume-yes update \
    && apt-get --assume-yes install python3-dev \
    && apt-get --assume-yes install gcc \
    && apt-get --assume-yes install -y git \
    && apt-get --assume-yes install graphviz
RUN git --version
RUN pip install -r requirements.txt
