pip install -e .
pip install --no-cache-dir flash_attn==2.8.1 --no-build-isolation
sudo cp /usr/include/crypt.h /venv/eval/include/python3.10/