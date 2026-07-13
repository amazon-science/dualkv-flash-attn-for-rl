#!/usr/bin/env bash
# Set these before running:
#   WORKDIR=<build root>  REPO=<gemma4-dev checkout>  VENV=<target venv dir>
: "${WORKDIR:?}" "${REPO:?}" "${VENV:?}"
set -x
source ${VENV}/bin/activate
# ray (memory: verl 0.8 async server used ray 2.5x on other stacks; match a modern ray)
pip install -q ray==2.49.0 wandb pandas pyarrow
# verl editable, no-deps (deps already in venv or handled)
pip install -q --no-deps -e ${REPO}/verl
# common verl runtime deps not in a vllm-only venv
pip install -q "antlr4-python3-runtime==4.9.3" omegaconf hydra-core tensordict codetiming dill   multiprocess python-dateutil xxhash orjson pylatexenc
python -c "import verl; print(\"verl\", verl.__version__)"
python -c "import ray; print(\"ray\", ray.__version__)"
