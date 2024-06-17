from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/")
async def get_request_object(request: Request):
    return {
        "path": request.url.path,
        "hostname": request.base_url.hostname,
        "port": request.base_url.port,
    }
