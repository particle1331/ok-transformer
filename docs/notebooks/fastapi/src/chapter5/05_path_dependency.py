from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, status
from typing import Optional
app = FastAPI()


def secret_header(secret_header: Optional[str] = Header(None)) -> None:
    print(secret_header)
    if (not secret_header) or (secret_header != "SECRET_VALUE"):
        raise HTTPException(status.HTTP_403_FORBIDDEN)


# Version 1
router1 = APIRouter(dependencies=[Depends(secret_header)])

@router1.get("/route-1")
async def router_route_1():
    return {"route": "route-1"}

@router1.get("/route-2")
async def router_route_2():
    return {"route": "route-2"}


# Version 2
router2 = APIRouter()
@router2.get("/route-1")
async def router_route_1():
    return {"route": "route-1"}

@router2.get("/route-2")
async def router_route_2():
    return {"route": "route-2"}


app = FastAPI()
app.include_router(router1, prefix="/router/v1")
app.include_router(router2, prefix="/router/v2", dependencies=[Depends(secret_header)])

@app.get("/protected-route", dependencies=[Depends(secret_header)]) # Add > 1 deps
async def protected_route():
    return {"hello": "world"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)