from fastapi import FastAPI
from enum import Enum

class UserType(str, Enum):
    STANDARD = "standard"
    ADMIN = "admin"

app = FastAPI()

@app.get("/users/{type}/{id}")
async def get_user(id: int, type: UserType):
    return {"id": id, "type": type}