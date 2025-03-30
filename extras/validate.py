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
    
def is_whitespace_or_empty(s: str) -> bool:
    # \S matches any non-whitespace character
    return not bool(re.search(r"\S", s))


class NotebookChecker:
    def __init__(self):
        self.checks = []
        self.flag = 0
        self.logs = {}

    def register(self, func):
        self.checks.append(func)
        return func
    
    def run_checks(self, filepath: dict):
        with open(filepath, "r") as f:
            notebook_orig = json.load(f)

        notebook = deepcopy(notebook_orig)
        for check in self.checks:
            exit_code, message, notebook = check(notebook)
            self.flag += exit_code
            if message:
                self.logs[filepath] = self.logs.get(filepath, []) + [(check.__name__, message)]
        
        if notebook_orig != notebook:
            if committed_changes(filepath):
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


@checker.register
def combine_tqdm_outputs(notebook: dict):
    progress_bar_pattern = re.compile(r'^\s*\d+%\|')
    changed = 0
    for cell in get_code_cells(notebook):
        # delete whitespace
        outputs = []
        for out in cell["outputs"]:
            # non-trivial behavior: skip whitespace stderr
            if (
                out.get("name") == "stderr" and \
                out.get("output_type") == "stream" and \
                is_whitespace_or_empty("".join(out.get("text", ["NO_TEXT"])))
            ):
                changed += 1
                continue
            
            # default behavior is to append
            outputs.append(out)
        cell["outputs"] = outputs

        # Delete all except last progress bar
        outputs = []
        progress_flag = 0
        for out in cell["outputs"]:
            if (
                out.get("name") == "stderr" and \
                out.get("output_type") == "stream" and \
                bool(progress_bar_pattern.match("".join(out.get("text"))))
            ):
                if not progress_flag:
                    progress_flag = 1
                    outputs.append(out)
                else:
                    changed = +1
                    outputs[-1] = out
            else:
                progress_flag = 0
                outputs.append(out)
        cell["outputs"] = outputs

    if changed > 0:
        return 1, "To combine tqdm outputs.", notebook
    else:
        return 0, "", notebook


if __name__ == "__main__":
    PATHS = list(glob.glob("docs/**/*.ipynb", recursive=True))
    print(f"Checking {len(PATHS)} notebooks...")

    for path in tqdm(PATHS):
        try:
            checker.run_checks(path)
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
