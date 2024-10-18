import os
import glob
import pprint
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", type=str)
args = parser.parse_args()


workdir = args.pattern.split("/")[:-1]
pattern = args.pattern.split("/")[-1]
os.chdir("docs/nb/" + "/".join(workdir))

PATHS = sorted(glob.glob(f"{pattern}*.ipynb"))
pprint.pprint(PATHS)

for nb in PATHS:
    try:
        subprocess.check_call(["pdm", "run", "papermill", nb, nb])
    except subprocess.CalledProcessError:
        print(f"Execution of {nb} failed. Stopping.")
        break
