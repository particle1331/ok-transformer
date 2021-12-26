from fastapi import FastAPI, Form
app = FastAPI()

@app.post("/users")
async def create_user(name: str = Form(...), age: int = Form(...)):
    return {"name": name, "age": age}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("form_data:app", reload=True)