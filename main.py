# main.py
from fastapi import FastAPI

app = FastAPI(title="BotDetector API")

@app.get("/")
def read_root():
    return {"message": "BotDetector is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}