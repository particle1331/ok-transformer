from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def f(x: int, y: int):
    return {"Hello": x + y}
    