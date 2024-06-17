from fastapi import APIRouter
from fastapi.params import Body


router = APIRouter()


@router.get("/posts")
def get_posts():
    return {"posts": "These are your posts..."}


@router.post("/posts")
def create_posts(post: dict = Body(...)):
    return {"message": "This is your new post.", "post": post}
