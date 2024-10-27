import glob
import json

from tqdm import tqdm


def save_remove_input(notebook_json: dict):
    for cell in notebook_json["cells"]:
        if cell["cell_type"] == "code" and len(cell["source"]) >= 1:
            for line in cell["source"]:
                if r"%%save" in line:
                    try:
                        assert "remove-input" in cell["metadata"]["tags"]
                    except:
                        return 1
    return 0


flagged = []
checklist = [save_remove_input]  # add notebook checks here
files = list(glob.glob("./docs/nb/**/*.ipynb", recursive=True))

print(f"Checking {len(files)} notebooks...")

for path in tqdm(files):
    with open(path, "r") as f:
        notebook = f.read()
        notebook = json.loads(notebook)
        
        count, logs = 0, []
        for check_fn in checklist:
            if check_fn(notebook):
                count += 1
                logs.append(check_fn.__name__)
        
        if count >= 1:
            flagged.append((path, logs))


if len(flagged) == 0:
    print("✅ All good! :)")
else:
    print("❌ Fix the ff:")
    for path, logs in flagged:
        print(path)
        for log in logs:
            print("\t", log)
    exit(1)
