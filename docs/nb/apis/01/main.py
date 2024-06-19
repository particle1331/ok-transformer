import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/files")
async def upload_files(files: list[UploadFile]):
    return [
        {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size,
            "dims": np.array(Image.open(file.file)).shape, 
        }
        for file in files
    ]
