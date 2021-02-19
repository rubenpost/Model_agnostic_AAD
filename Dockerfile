#pip freeze > requirements.txt


FROM python:3.7-slim-buster
WORKDIR /app
ADD . /app

RUN apt-get update -yq \
    && apt-get install curl gnupg -yq \
    && curl -sL https://deb.nodesource.com/setup_11.x | bash \
    && apt-get install nodejs -yq

ENV ACCEPT_EULA=Y
RUN apt-get --assume-yes update \
    && apt-get --assume-yes install python3-dev \
    && apt-get --assume-yes install gcc \
    && apt-get --assume-yes install -y git \
    && apt-get --assume-yes install graphviz \
    && apt-get --assume-yes install xdg-utils 

RUN git --version
RUN pip install -r requirements.txt
