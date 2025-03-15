from IPython.core.magic import register_cell_magic
from IPython.display import Code
import importlib
import os


@register_cell_magic
def save(line, cell):
    # Append the cell content to the file
    with open("chapter.py", "a") as f:
        f.write(cell + "\n")

    # Execute the cell content
    exec(cell, globals())

    # Import or reload the saved module
    module_name = os.path.splitext("chapter.py")[0]  # Get module name ('chapter' from 'chapter.py')
    if module_name in globals():
        importlib.reload(globals()[module_name])  # Reload if already imported
    else:
        globals()[module_name] = __import__(module_name)  # First-time import


    return Code(cell, language="python")
