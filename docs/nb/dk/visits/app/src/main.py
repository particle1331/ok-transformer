from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import os
import redis

app = FastAPI()
r = redis.Redis(host='redis-server', port=6379)
if not r.exists('visits'):
    r.set('visits', 0)


@app.get("/", response_class=HTMLResponse)
def home():
    visits = int(r.get('visits')) + 1
    r.set('visits', visits)
    return f"Number of visits: {visits}"

@app.get("/kill")
async def kill_uvicorn():
    parent_pid = os.getppid()
    os.kill(parent_pid, 9)
