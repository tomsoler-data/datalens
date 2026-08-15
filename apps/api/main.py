from fastapi import FastAPI


app = FastAPI(
    title="DataLens API",
    description="Backend API for DataLens.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "datalens-api",
        "version": "0.1.0",
    }