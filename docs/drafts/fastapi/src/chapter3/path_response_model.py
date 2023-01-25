from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class Post(BaseModel):
    title: str
    nb_views: int

class PublicPost(BaseModel):
    title: str

# Dummy database
posts = {
    1: Post(title="Post #1", nb_views=100), # Private
}

@app.get("/posts/{id}", response_model=PublicPost)
async def get_post(id: int):
    return posts[id] # Public


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("path_response_model:app", reload=True)