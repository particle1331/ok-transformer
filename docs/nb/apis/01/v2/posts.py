from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field


class Post(BaseModel):
    title: str = Field(min_length=1)
    content: str
    published: bool = True
    rating: Optional[int] = None


router = APIRouter()

posts = [
    {
        "id": 1,
        "title": "First post",
        "content": "This is your first post.",
        "rating": None,
        "published": True,
    },
    {
        "id": 2,
        "title": "Second post",
        "content": "This is your second post.",
        "rating": None,
        "published": False,
    },
]


@router.get("/posts")
def get_posts():
    return posts


@router.get("/posts/{id}")
def get_post_by_id(id: int):
    return [post for post in posts if post["id"] == id][0]


@router.post("/posts")
def create_posts(post: Post):
    post_dict = post.model_dump()
    post_dict["id"] = max([post["id"] for post in posts]) + 1
    posts.append(post_dict)
    return post_dict
