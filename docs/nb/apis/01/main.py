from fastapi import FastAPI
from v1.posts import router as router1
from v2.posts import router as router2


app = FastAPI()
app.include_router(router1, prefix="/v1", tags=["/v1/posts"])
app.include_router(router2, prefix="/v2", tags=["/v2/posts"])


@app.get("/")
def root():
    return {"message": "Hello, world!"}
