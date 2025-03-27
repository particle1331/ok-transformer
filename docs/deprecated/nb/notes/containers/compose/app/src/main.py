import redis
from fastapi import FastAPI

app = FastAPI()
r = redis.Redis(host="redis", port=6379)

if not r.exists("visits"):
    r.set("visits", 0)


@app.get("/")
def root():
    visits = int(r.get("visits")) + 1
    r.set("visits", visits)
    return {
        "message": "Hello world!",
        "visit_count": r.get("visits"),
    }
