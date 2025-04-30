from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel
app = FastAPI()


class Post(BaseModel):
    id: int
    title: str

class db:
    posts = {
        1: Post(id=1, title="Post #1"),
        2: Post(id=2, title="Post #2"),
    }


async def get_post_or_404(id: int) -> Post:
    try:
        return db.posts[id]
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@app.get("/posts/{id}")
async def get_post(post: Post = Depends(get_post_or_404)):
    return post


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)