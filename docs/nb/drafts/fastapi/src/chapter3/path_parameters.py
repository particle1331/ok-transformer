from fastapi import FastAPI, Path
from enum import Enum
app = FastAPI()


class UserType(str, Enum):
    STANDARD = "standard"
    ADMIN = "admin"


@app.get("/v1/users/{id}")
async def get_user(id: int):
    return {"id": id}

@app.get("/v2/users/{type}/{id}")
async def get_user(id: int, type: UserType):
    return {"id": id, "type": type}

@app.get("/v3/users/{id}")
async def get_user(id: int = Path(..., ge=1)):
    return {"id": id}

@app.get("/username/{username}")
async def get_username(username: str = Path(..., min_length=1, max_length=20)):
    return {"username": username}

@app.get("/license-plates/{license}")
async def get_license_plate(license: str = Path(..., regex=r"^\w{2}-\d{3}-\w{2}$")):
    return {"license": license}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("path_parameters:app", reload=True)