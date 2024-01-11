FROM tiangolo/uvicorn-gunicorn:python3.9
LABEL maintainer="me <me@example.com>"

# Add any system dependency here
RUN apt-get update -y && \
    apt-get install libsndfile1 ffmpeg -y

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install your desired Python version
ARG PYTHON_VERSION=3.10.1
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash


# See PyTorch releases for pip here: https://download.pytorch.org/whl/torch_stable.html
COPY ./requirements.txt /app
RUN pip install torch==2.0.1 && \
    pip install Cython numpy==1.26.3
RUN pip install -r requirements.txt
COPY ./prestart.sh /app/


# Most DL models are quite large in terms of memory, using workers is a HUGE
# slowdown because of the fork and GIL with python.
# Using multiple pods seems like a better default strategy.
# Feel free to override if it does not make sense for your library.
ARG max_workers=1
ENV MAX_WORKERS=$max_workers
ENV HUGGINGFACE_HUB_CACHE=/data
ENV NEMO_CACHE_DIR=/data/nemo_cache/

# Necessary on GPU environment docker.
# TIMEOUT env variable is used by nvcr.io/nvidia/pytorch:xx for another purpose
# rendering TIMEOUT defined by uvicorn impossible to use correctly
# We're overriding it to be renamed UVICORN_TIMEOUT
# UVICORN_TIMEOUT is a useful variable for very large models that take more
# than 30s (the default) to load in memory.
# If UVICORN_TIMEOUT is too low, uvicorn will simply never loads as it will
# kill workers all the time before they finish.
RUN sed -i 's/TIMEOUT/UVICORN_TIMEOUT/g' /gunicorn_conf.py
COPY ./app /app/app
