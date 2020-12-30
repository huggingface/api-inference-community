FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y -q bash \
                      build-essential \
                      git \
                      curl \
                      ca-certificates \
                      libfreetype6-dev \
                      libsndfile1-dev \
                      pkg-config \
                      python3 \
                      python3-pip \
                      software-properties-common \
                      wget

RUN python3 --version && pip3 --version

WORKDIR /app
COPY . /app

RUN pip3 install torch starlette aiofiles uvicorn

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]
