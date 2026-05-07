python -m venv .venv
source .venv/bin/activate
pip install packaging wheel torch
MAX_JOBS=64 pip install -r requirements.txt