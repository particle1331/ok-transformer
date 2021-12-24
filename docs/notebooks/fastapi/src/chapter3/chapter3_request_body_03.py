from fastapi import FastAPI, Body
from pydantic import BaseModel
app = FastAPI()

class User(BaseModel):
    name: str
    age: int

class Company(BaseModel):
    name: str

@app.post("/users")
async def create_user(
    user: User, 
    company: Company, 
    priority: int = Body(..., ge=1, le=3)):
    
    return {
        "user": user, 
        "company": company, 
        "priority": priority
    }