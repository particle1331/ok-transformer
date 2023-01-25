from fastapi import FastAPI, Body
from pydantic import BaseModel
app = FastAPI()

class User(BaseModel):
    name: str
    age: int

class Company(BaseModel):
    name: str

@app.post("/v1/users")
async def user(
    name: str = Body(...), 
    age:  int = Body(...),
):
    return {
        "name": name, 
        "age":  age,
    }

@app.post("/v2/users")
async def user(user: User):
    return user

@app.post("/v3/users")
async def create_user(
    user: User, 
    company: Company, 
    priority: int = Body(..., ge=1, le=3)):
    
    return {
        "user": user, 
        "company": company, 
        "priority": priority
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("request_body:app", reload=True)