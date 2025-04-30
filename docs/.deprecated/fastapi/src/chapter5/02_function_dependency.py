from fastapi import FastAPI, Depends
from typing import Tuple
app = FastAPI()

async def pagination(skip: int = 0, limit: int = 10) -> Tuple[int, int]:
    return (skip, limit)

@app.get("/items")
async def list_items(p: Tuple[int, int] = Depends(pagination)):
    skip, limit = p
    return {"skip": skip, "limit": limit}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)