FROM mambaorg/micromamba
LABEL maintainer="Adrin Jalali adrin@hf.co"

# micromamba comes with a default non-root user. But we need root to install
# our required system packages.
USER root
RUN apt-get update && apt-get install -y curl jq
USER $MAMBAUSER

# Most our dockerfiles start from tiangolo/uvicorn-gunicorn:python3.8, but
# since here we'd like to start from micromamba, we copy necessary files from
# the uvicorn docker image using `COPY --from=...` commands. These steps are
# taken from:
# https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.8-slim.dockerfile

COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /start.sh /start.sh
RUN chmod +x /start.sh

COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /gunicorn_conf.py /gunicorn_conf.py

COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /start-reload.sh /start-reload.sh
RUN chmod +x /start-reload.sh

COPY --from=tiangolo/uvicorn-gunicorn:python3.8 /app /app
WORKDIR /app/

ENV PYTHONPATH=/app

EXPOSE 80


# This part is new and only specific to scikit-learn image.
ENV HUGGINGFACE_HUB_CACHE=/data

COPY ./app /app/app

COPY run_app.sh /run_app.sh
RUN chmod +x /run_app.sh

CMD /run_app.sh
