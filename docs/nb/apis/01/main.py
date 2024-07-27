import time
from typing import Annotated
from fastapi import FastAPI, Response, Cookie, Request

app = FastAPI()

@app.get("/set-header")
async def custom_header(response: Response):
    response.headers["Custom-Header"] = "Custom-Header-Value"
    return {"hello": "world"}

@app.get("/set-cookie")
async def custom_cookie(request: Request, response: Response, cookie_name: Annotated[str | None, Cookie()] = None):
    response.set_cookie("cookie-name", "cookie-value", max_age=86400)  # seconds
    return {
        "cookie_value": request.cookies.get("cookie-name"), 
        "request_headers": request.headers, 
        "response_headers": response.headers,
    }

@app.get("/cookie")
async def get_cookie(hello: Annotated[str | None, Cookie()] = None):
    return {"hello": hello}
