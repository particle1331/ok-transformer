brew install libomp
conda create -n ml python=3.8.12
conda activate ml
conda install -c apple tensorflow-deps
conda install -c conda-forge py-xgboost
conda install -c pytorch pytorch
conda install -c pytorch torchvision
pip install tensorflow-macos
pip install tensorflow-metal
pip install tensorflow-datasets
pip install -r requirements.txt     # Other data science libraries.
pip install papermill               # For running notebooks on the terminal; "compiling" notebooks