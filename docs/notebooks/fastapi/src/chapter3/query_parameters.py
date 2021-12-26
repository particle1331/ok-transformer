from fastapi import FastAPI, Path
from enum import Enum

app = FastAPI()

class UserType(str, Enum):
    STANDARD = "standard"
    ADMIN = "admin"


@app.get("/users/{type}/{id}")
async def user(
    type: UserType, 
    id: int = Path(..., ge=1), 
    page: int = 1,
    size: int = 1):
    return {
        "type": type,
        "id": id,
        "page": page, 
        "size": size,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("query_parameters:app", reload=True)