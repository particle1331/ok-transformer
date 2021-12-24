from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.post("/users")
async def user(user: User):
    return user