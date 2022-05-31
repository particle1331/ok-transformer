from fastapi import FastAPI, Response
app = FastAPI()

@app.get("/custom-header")
async def custom_header(response: Response):
    response.headers["Custom-Header"] = "Custom-Header-Value"
    return {"hello": "world"}

@app.get("/custom-cookie")
async def custom_cookie(response: Response):
    response.set_cookie("cookie-name", "cookie-value", max_age=86400) # -> name value pair
    return {"hello": "world"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)