from fastapi import FastAPI

app = FastAPI(title="Debug Test")

@app.get("/")
async def root():
    return {"message": "Hello World", "status": "working"}

@app.get("/health")
async def health():
    return {"status": "healthy", "debug": "minimal version"}

# Vercel handler
handler = app
