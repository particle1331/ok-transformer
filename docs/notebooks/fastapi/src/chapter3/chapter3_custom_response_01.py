from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
app = FastAPI()

@app.get("/dog")
async def get_dog():
    root_directory = Path(__file__).absolute().parents[2]
    img_path = root_directory / "assets" / "dog.jpeg"
    return FileResponse(img_path)
