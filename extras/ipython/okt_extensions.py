from IPython.core.magic import register_cell_magic
from IPython.display import Code


@register_cell_magic
def save(line, cell):
    # Append the cell content to the file
    with open("chapter.py", "a") as f:
        f.write(cell + "\n")

    # Execute the cell content
    exec(cell, globals())

    return Code(cell, language="python")
