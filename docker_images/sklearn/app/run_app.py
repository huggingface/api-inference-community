import uvicorn


uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
