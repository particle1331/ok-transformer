from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def f(x: int):
    return {"Hello": x}

uvicorn.run(app)