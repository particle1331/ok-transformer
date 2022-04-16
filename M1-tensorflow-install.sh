brew install libomp
xcode-select --install
bash Miniforge3-MacOSX-arm64.sh  # https://github.com/conda-forge/miniforge
conda create -n ml python~=3.9.0
conda activate ml
conda install -c apple tensorflow-deps -y
conda install -c conda-forge py-xgboost -y
conda install -c pytorch pytorch -y
conda install -c pytorch torchvision -y
pip install tensorflow-macos
pip install tensorflow-metal
pip install tensorflow-datasets

# Other requirements: building + data science
pip install -r requirements-build.txt
pip install -e src
