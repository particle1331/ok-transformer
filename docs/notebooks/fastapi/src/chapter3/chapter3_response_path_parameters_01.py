from fastapi import FastAPI, status, Response
from pydantic import BaseModel
app = FastAPI()

class Post(BaseModel):
    title: str

# Dummy database
posts = {}

@app.post("/posts", status_code=status.HTTP_201_CREATED)
async def create_post(post: Post):
    posts[len(posts) + 1] = post
    return post

@app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(id: int):
    posts.pop(id, None)
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app)