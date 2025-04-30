from fastapi import FastAPI, File, UploadFile
from typing import List
app = FastAPI()

@app.post("/v1/files")
async def upload_file(file: bytes = File(...)):
    return {"file_size": len(file)}

@app.post("/v2/files")
async def upload_file(file: UploadFile = File(...)):
    return {
        "file_name": file.filename, 
        "content_type": file.content_type
    }

@app.post("/v3/files")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    return [
        {
            "file_name": file.filename,
            "content_type": file.content_type
        }
        for file in files
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("file_uploads:app", reload=True)

