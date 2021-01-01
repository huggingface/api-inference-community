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
                      wget \
                      ffmpeg

RUN python3 --version && pip3 --version

RUN pip3 install torch starlette aiofiles uvicorn

WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python3", "main.py"]
