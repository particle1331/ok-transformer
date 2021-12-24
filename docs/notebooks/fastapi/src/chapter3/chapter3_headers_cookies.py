from fastapi import FastAPI, Header, Cookie
from typing import Optional
app = FastAPI()

@app.get("/")
async def get_header(hello_world: str = Header(...), coke: Optional[str] = Cookie(None)):
    return {"hello_world": hello_world, "coke": coke}