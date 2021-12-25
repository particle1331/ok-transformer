from fastapi import FastAPI, Response, status
from pydantic import BaseModel
app = FastAPI()

class Post(BaseModel):
    title: str
    nb_views: int

posts = {
    1: Post(title="Post #1", nb_views=1)
}

@app.put("/posts/{id}")
async def update_or_create_post(id: int, post: Post, response: Response):
    if id not in posts.keys():
        response.status_code = status.HTTP_201_CREATED
    posts[id] = post
    