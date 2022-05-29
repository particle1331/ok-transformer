from fastapi import FastAPI, Request
app = FastAPI()

@app.get("/random-path")
async def get_request_object(request: Request):
    return {"path": request.url.path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("request_object:app", reload=True)