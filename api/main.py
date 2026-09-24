from fastapi import FastAPI

app = FastAPI(title="RecSys API")

@app.get("/health")
def health():
    return {"status": "ok"}
