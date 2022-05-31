from typing import Tuple 
from fastapi import FastAPI, Query, Depends


class Pagination:
    def __init__(self, maximum_limit: int = 100):
        self.maximum_limit = maximum_limit
    
    async def __call__(self,
        skip: int = Query(0, ge=0), 
        limit: int = Query(10, ge=0),
    ) -> Tuple[int, int]:
        capped_limit = min(self.maximum_limit, limit)
        return (skip, capped_limit)


app = FastAPI()
pagination_50 = Pagination(maximum_limit=50)

@app.get("/items")
async def list_items(p: Tuple[int, int] = Depends(pagination_50)):
    skip, limit = p
    return {"skip": skip, "limit": limit}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)