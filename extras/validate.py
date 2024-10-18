import git
import json
import glob
import pathlib

import inspect
import functools

from tqdm import tqdm


def wrap_args(func):
    @functools.wraps(func)
    def wrapper(*a, **kw):
        sig = inspect.signature(func)
        params = sig.parameters
        kw_ = {k: v for k, v in kw.items() if k in params}
        return func(*a, **kw_)

    return wrapper


def get_code_cells(notebook: dict):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


@wrap_args
def save_remove_input(notebook: dict):
    tags_required = ["remove-input"]
    tags_allowed = ["remove-input", "hide-output"]
    code_cells = get_code_cells(notebook)
    for cell in code_cells:
        source = " ".join(cell["source"])
        if r"%%save" in source:
            try:
                tags = cell["metadata"].get("tags", [])
                assert set(tags_required) <= set(tags) <= set(tags_allowed)
            except:
                return 1, "Improper tags for save cell."
    return 0, ""


@wrap_args
def chapter_module_remove_cell(notebook: dict):
    code_cells = get_code_cells(notebook)
    for cell in code_cells:
        source = " ".join(cell["source"])
        if "chapter" in source:
            try:
                tags = cell["metadata"].get("tags", [])
                assert "remove-cell" in tags
            except:
                return 1, "Tag remove-cell not found."
    return 0, ""


def committed_changes(filepath: str) -> bool:
    """Check if current file has no uncommitted changes."""
    try:
        root = pathlib.Path(__file__).parents[1]
        repo = git.Repo(root)
        diff = repo.index.diff(None)
        return filepath not in [item.a_path for item in diff]
    except:
        return False


@wrap_args
def combine_multiline_outputs(notebook: dict, filepath: str):
    changed = 0
    code_cells = get_code_cells(notebook)
    for cell in code_cells:
        outputs = []
        stream_flag = 0
        for output in cell["outputs"]:
            if output.get("name") == "stdout" and output.get("output_type") == "stream":
                if not stream_flag:
                    stream_flag = 1
                    outputs.append(output)
                else:
                    changed += 1
                    out = outputs[-1]
                    out["text"] = out["text"] + output["text"]
                    outputs[-1] = out
            else:
                stream_flag = 0
                outputs.append(output)
        cell["outputs"] = outputs

    if changed > 0:
        if committed_changes(filepath):
            with open(filepath, "w") as f:
                f.write(json.dumps(notebook, indent=1))
            return 1, "File overwritten: combined outputs."
        else:
            return 1, "Unable to overwrite: error / uncommited changes exist."
    else:
        return 0, ""


if __name__ == "__main__":
    CHECKLIST = [
        save_remove_input,
        chapter_module_remove_cell,
        combine_multiline_outputs,
    ]

    PATHS = list(glob.glob("docs/nb/**/*.ipynb", recursive=True))
    print(f"Checking {len(PATHS)} notebooks...")

    logs, flag = [], 0
    for path in tqdm(PATHS):
        try:
            with open(path, "r") as f:
                notebook = f.read()
                notebook = json.loads(notebook)

                for check_fn in CHECKLIST:
                    kw = {"notebook": notebook, "filepath": path}
                    exit_code, message = check_fn(**kw)
                    flag += exit_code
                    if len(message) > 0:
                        logs.append((path, check_fn.__name__, message))

        except Exception as e:
            print("⚠️ Skipped", path)
            print("    ", e)

    print()
    for log in logs:
        print(log[0] + "\n  " + log[1] + "\n    " + log[2])

    print()
    if flag == 0:
        print("✅ All good! ⸜(｡˃ ᵕ ˂ )⸝")
    else:
        print("❌ Error ( •̀ - •)")
        exit(1)
