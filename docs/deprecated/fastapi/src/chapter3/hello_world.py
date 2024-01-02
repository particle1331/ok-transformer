from fastapi import FastAPI, Body
app = FastAPI()


@app.get("/")
async def hello_world():
    return {"hello": "world"}

@app.post("/")
async def message(msg: str = Body(..., embed=True)):
    return {"message": msg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hello_world:app", reload=True)