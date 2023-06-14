FROM tiangolo/uvicorn-gunicorn:python3.9
LABEL maintainer="me <me@example.com>"

# Add any system dependency here
RUN apt-get update -y && \
    apt-get install libsndfile1 ffmpeg -y

# See PyTorch releases for pip here: https://download.pytorch.org/whl/torch_stable.html
COPY ./requirements.txt /app
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp39-cp39-linux_x86_64.whl && \
    pip install Cython numpy==1.21.6
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
