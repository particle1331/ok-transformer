import sys
import nbformat
from pathlib import Path


if __name__ == "__main__":
    path  = sys.argv[1]
    title = sys.argv[2]

    if "/" in path:
        folder, filename = path.split("/")
    else:
        folder, filename = "", path

    root = Path(__file__).parents[1]
    folder_path = root / "docs" / "nb" / folder
    folder_path.mkdir(exist_ok=True)

    # Replace the placeholder with the filename
    template_path = root / "extras" / "template.ipynb"
    template = nbformat.read(template_path, as_version=nbformat.NO_CONVERT)
    for cell in template["cells"]:
        if "{filename}" in cell["source"]:
            local_path = f"{folder}/{filename.removesuffix('.ipynb')}"
            cell["source"] = cell["source"].replace("{filename}", local_path)

        if "{title}" in cell["source"]:
            cell["source"] = cell["source"].replace("{title}", title)

    # Save the new file
    nbformat.write(template, (folder_path / filename))
