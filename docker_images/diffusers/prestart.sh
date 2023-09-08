echo "Prestart start at " $(date)

METRICS_ENABLED=${METRICS_ENABLED:-"0"}

if [ "$METRICS_ENABLED" = "1" ];then
    echo "Spawning metrics server"
    gunicorn -k "uvicorn.workers.UvicornWorker" --bind :${METRICS_PORT:-9400} "app.healthchecks:app" &
    pid=$!
    echo "Metrics server pid: $pid"
fi

echo "Prestart done at " $(date)
