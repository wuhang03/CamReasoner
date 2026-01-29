export WANDB_API_KEY=1f7c29719b1ce4fbcc10357cbdebaf69e24cc312
conda install -c nvidia cuda-toolkit=11.8 -y
pip install -e ".[torch,metrics]" --no-build-isolation