conda install -c nvidia cuda-toolkit=11.8 -y
pip install -e ".[torch,metrics]" --no-build-isolation