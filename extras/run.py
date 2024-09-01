import os
import glob
import pprint
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str)
args = parser.parse_args()

os.chdir(f"./docs/nb/{args.folder}")
notebooks = sorted(glob.glob("*.ipynb"))
pprint.pprint(notebooks)

for nb in notebooks:
    try:
        subprocess.check_call(["pdm", "run", "papermill", nb, nb])
    except subprocess.CalledProcessError:
        print(f"Execution of {nb} failed. Stopping.")
        break
