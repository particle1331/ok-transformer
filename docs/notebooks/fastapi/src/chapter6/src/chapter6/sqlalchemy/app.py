from fastapi import FastAPI, status, Depends, HTTPException, Query
from h11 import Data
from soupsieve import select
import sqlalchemy
from typing import Tuple, List
from databases import Database
from chapter6.sqlalchemy.database import database, sqlalchemy_engine, get_database
from chapter6.sqlalchemy.models import posts, metadata, PostDB, PostCreate
from chapter6.sqlalchemy.models import PostPartialUpdate

app = FastAPI()


########################### Dependencies #################################

async def get_post_or_404(
    id: int,
    database: Database=Depends(get_database) 
) -> PostDB:

    select_query = posts.select().where(posts.c.id == id) # overloaded
    raw_post = await database.fetch_one(select_query)
    if raw_post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    return PostDB(**raw_post)


async def pagination(skip: int=Query(0, ge=0), limit: int=Query(10, ge=0)) -> Tuple[int, int]:
    capped_limit = min(100, limit)
    return (skip, capped_limit)


############################# Endpoints ##################################

@app.on_event("startup")
async def startup():
    await database.connect()
    metadata.create_all(sqlalchemy_engine)


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


@app.post("/posts", response_model=PostDB, status_code=status.HTTP_201_CREATED)
async def create_post(
    post: PostCreate,
    database: Database=Depends(get_database)
) -> PostDB:

    insert_query = posts.insert().values(post.dict())
    post_id = await database.execute(insert_query)
    post_db = await get_post_or_404(post_id, database)
    return post_db


@app.get("/posts/{id}", response_model=PostDB)
async def get_post(post: PostDB=Depends(get_post_or_404)) -> PostDB:
    return post


@app.get("/posts")
async def list_posts(
    pagination: Tuple[int, int]=Depends(pagination), 
    database: Database=Depends(get_database)
) -> List[PostDB]:

    skip, limit = pagination
    select_query = posts.select().offset(skip).limit(limit)
    rows = await database.fetch_all(select_query)
    results = [PostDB(**row) for row in rows]
    return results


@app.patch("/posts/{id}", response_model=PostDB)
async def update_post(
    post_update: PostPartialUpdate,
    post: PostDB=Depends(get_post_or_404),
    database: Database=Depends(get_database)
) -> PostDB:

    update_query = (
        posts.update()
        .where(posts.c.id == post.id)
        .values(post_update.dict(exclude_unset=True))
    )

    post_id = await database.execute(update_query)
    post_db = await get_post_or_404(post_id, database)
    return post_db


@app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post: PostDB=Depends(get_post_or_404),
    database: Database=Depends(get_database)
):
    delete_query = posts.delete().where(posts.c.id == post.id)
    await database.execute(delete_query)
