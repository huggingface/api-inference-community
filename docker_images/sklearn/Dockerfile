FROM condaforge/mambaforge
LABEL maintainer="Adrin Jalali adrin@hf.co"

# Add any system dependency here
# RUN apt-get update -y && apt-get install libXXX -y

# Most DL models are quite large in terms of memory, using workers is a HUGE
# slowdown because of the fork and GIL with python.
# Using multiple pods seems like a better default strategy.
# Feel free to override if it does not make sense for your library.
ARG max_workers=1
ENV MAX_WORKERS=$max_workers
ENV HUGGINGFACE_HUB_CACHE=/data

COPY ./app /app/app
