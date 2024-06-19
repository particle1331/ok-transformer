import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/files")
async def upload_file(file: UploadFile):
    img = Image.open(file.file)
    return {
        "filename": file.filename,
        "size": file.size,
        "dims": np.array(img).shape, 
    }
