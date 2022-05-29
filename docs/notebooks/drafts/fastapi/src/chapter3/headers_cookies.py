from fastapi import FastAPI, Header, Cookie
from typing import Optional
app = FastAPI()

@app.get("/")
async def get_header(hello_world: str = Header(...), cookie: Optional[str] = Cookie(None)):
    return {"hello_world": hello_world, "cookie": cookie}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("headers_cookies:app", reload=True)
