import os
import sys
import json
import shutil
import nbformat
from pathlib import Path


if __name__ == "__main__":
    arg = sys.argv[1]
    title = sys.argv[2]
    if "/" in arg:
        folder, filename = arg.split("/")
    else:
        folder, filename = "tmp", arg

    root = Path(__file__).parents[1]
    folder_path = root / "docs" / "nb" / folder
    folder_path.mkdir(exist_ok=True)
    
    # Replace the placeholder with the filename
    template_path = root / "extras" / "template.ipynb"
    template = nbformat.read(template_path, as_version=nbformat.NO_CONVERT)
    for cell in template["cells"]:
        if "{filename}" in cell["source"]:
            cell["source"] = cell["source"].replace("{filename}", f"{folder}/{filename}")
        
        if "{title}" in cell["source"]:
            cell["source"] = cell["source"].replace("{title}", title)

    # Save the new file
    nbformat.write(template, (folder_path / (filename + ".ipynb")))
