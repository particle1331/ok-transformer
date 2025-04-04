import re
import git
import json
import glob
import pathlib
from tqdm import tqdm
from copy import deepcopy


def get_code_cells(notebook: dict):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]

def committed_changes(filepath: str) -> bool:
    try:
        root = pathlib.Path(__file__).parents[1]
        repo = git.Repo(root)
        diff = repo.index.diff(None)
        return filepath not in [item.a_path for item in diff]
    except:
        return False


class NotebookChecker:
    def __init__(self):
        self.checks = []
        self.flag = 0
        self.logs = {}

    def register(self, func):
        self.checks.append(func)
        return func
    
    def run_checks(self, filepath: dict, debug: int):
        with open(filepath, "r") as f:
            notebook_orig = json.load(f)

        notebook = deepcopy(notebook_orig)
        for check in self.checks:
            exit_code, message, notebook = check(notebook)
            self.flag += exit_code
            if message:
                self.logs[filepath] = self.logs.get(filepath, []) + [(check.__name__, message)]
        
        if notebook_orig != notebook:
            if debug or committed_changes(filepath):
                with open(filepath, "w") as f:
                    json.dump(notebook, f, indent=2)
            else:
                self.logs[filepath] += [("COMMIT_CHECK", "⚠️ cannot write nb plz commit first")]


checker = NotebookChecker()


@checker.register
def save_remove_input(notebook: dict):
    tags_required = ["remove-input"]
    tags_allowed = ["remove-input", "hide-output"]
    for cell in get_code_cells(notebook):
        source = " ".join(cell["source"])
        if r"%%save" in source:
            try:
                tags = cell["metadata"].get("tags", [])
                assert set(tags_required) <= set(tags) <= set(tags_allowed)
            except:
                return 1, "Improper tags for save cell.", notebook
    return 0, "", notebook


@checker.register
def triggers_remove_cell(notebook: dict):
    triggers = [
        "chapter",
        "def savefig(",
        "def directive(",
    ]
    for cell in get_code_cells(notebook):
        source = " ".join(cell["source"])
        for word in triggers:
            if word in source:
                try:
                    tags = cell["metadata"].get("tags", [])
                    assert "remove-cell" in tags
                except:
                    return 1, f"Tag remove-cell not found: '{word}'", notebook
    return 0, "", notebook


@checker.register
def chapter_module_remove_cell(notebook: dict):
    for cell in get_code_cells(notebook):
        source = " ".join(cell["source"])
        if "chapter" in source:
            try:
                tags = cell["metadata"].get("tags", [])
                assert "remove-cell" in tags
            except:
                return 1, "Tag remove-cell not found.", notebook
    return 0, "", notebook


@checker.register
def combine_tqdm_outputs(notebook: dict):
    changed = 0
    pattern_tqdm = re.compile(r"^\s*\d+%\|")
    pattern_text = re.compile(r"\S")
    for cell in get_code_cells(notebook):
        outputs_else = []
        outputs_tqdm = []
        for out in cell["outputs"]:
            name = out.get("name")
            type = out.get("output_type")
            if name == "stderr" and type == "stream":
                text = ["".join(out.get("text", ["NO_TEXT"])).strip() + "\n\n"]  # flatten
                if pattern_tqdm.match(text[0]):
                    out["text"] = text
                    outputs_tqdm.append(out)
                elif not pattern_text.match(text[0]):
                    continue
            else:
                outputs_else.append(out)

        # insert final progress bar at beginning
        outputs = outputs_tqdm[-1:] + outputs_else
        if cell["outputs"] != outputs:
            changed += 1
            cell["outputs"] = outputs
            
    if changed:
        return 1, "To combine tqdm outputs.", notebook
    else:
        return 0, "", notebook


@checker.register
def combine_multiline_outputs(notebook: dict):
    changed = 0
    for cell in get_code_cells(notebook):
        outputs = []
        stream_flag = 0
        for output in cell["outputs"]:
            if output.get("name") == "stdout" and output.get("output_type") == "stream":
                if not stream_flag:
                    stream_flag = 1
                    outputs.append(output)
                else:
                    changed += 1
                    outputs[-1]["text"] += output["text"]
            else:
                stream_flag = 0
                outputs.append(output)
        cell["outputs"] = outputs

    if changed > 0:
        return 1, "To combine multi-line outputs.", notebook
    else:
        return 0, "", notebook


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    DEBUG = args.debug
    PATHS = list(glob.glob("docs/**/*.ipynb", recursive=True))

    print(f"Checking {len(PATHS)} notebooks...")

    for path in tqdm(PATHS):
        try:
            checker.run_checks(path, debug=DEBUG)
        except Exception as e:
            print(f"⚠️ Skipped {path}\n    {e}")

    for path in checker.logs:
        logs = checker.logs[path]
        print("\n" + path)
        for log in logs:
            func_name = log[0]
            message = log[1]
            print(f"  ◎ {func_name}\n    {message}")

    print()
    if checker.flag == 0:
        print("✅ All good! ⸜(｡˃ ᵕ ˂ )⸝")
    else:
        print("❌ Error ( •̀ - •)")
        exit(1)
