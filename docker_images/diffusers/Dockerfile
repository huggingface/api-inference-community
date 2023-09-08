FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Nicolas Patry <nicolas@huggingface.co>"

# Add any system dependency here
# RUN apt-get update -y && apt-get install libXXX -y

ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install your desired Python version
ARG PYTHON_VERSION=3.9.1
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Upgrade pip and install your desired packages
ARG PIP_VERSION=22.3.1
RUN pip install --no-cache-dir --upgrade pip==${PIP_VERSION} setuptools wheel && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Most DL models are quite large in terms of memory, using workers is a HUGE
# slowdown because of the fork and GIL with python.
# Using multiple pods seems like a better default strategy.
# Feel free to override if it does not make sense for your library.
ARG max_workers=1
ENV MAX_WORKERS=$max_workers
ENV HUGGINGFACE_HUB_CACHE=/data
ENV DIFFUSERS_CACHE=/data
ENV HF_HOME=/data

# Necessary on GPU environment docker.
# TIMEOUT env variable is used by nvcr.io/nvidia/pytorch:xx for another purpose
# rendering TIMEOUT defined by uvicorn impossible to use correctly
# We're overriding it to be renamed UVICORN_TIMEOUT
# UVICORN_TIMEOUT is a useful variable for very large models that take more
# than 30s (the default) to load in memory.
# If UVICORN_TIMEOUT is too low, uvicorn will simply never loads as it will
# kill workers all the time before they finish.
COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /app/ /app
COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /start.sh /
COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /gunicorn_conf.py /
COPY app/ /app/app

COPY ./prestart.sh /app/

RUN sed -i 's/TIMEOUT/UVICORN_TIMEOUT/g' /gunicorn_conf.py

CMD ["/start.sh"]
