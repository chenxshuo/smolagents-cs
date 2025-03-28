# for smolagents
# first fork the official repo then clone my forked version.
# this should be done in ../smolagents-cs folder

# env

# setup python 3.13 if not; otherwise cannot install smolagents in editable mode.
# for python 3.12 -> error: no module named certifi
# on linux
#conda init
#deactivate
#conda create -n py31016_env python=3.10 # use an other name if existed
#conda activate py31016_env # rather than base.
# on mac
#brew update
#brew upgrade pyenv
#pyenv install 3.10.16
#pyenv local 3.10.16

python --version

python -m venv venv-smolagents-deep-research --prompt="smolagents-deep-research"
source venv-smolagents-deep-research/bin/activate
which pip
which python
pip install -e '.[dev]'

pip install -r examples/open_deep_research/requirements.txt

pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents

# API:
#SERPAPI_API_KEY: https://serpapi.com/manage-api-key
#SERPER_API_KEY: https://serper.dev/api-key
#openai: https://platform.openai.com/docs/guides/prompt-engineering

# VLLM
# VLLM needs python >= 3.9 < 3.13
pip install vllm


# install cs-deep-research
cd cs-deep-research
pip install -e .

pip install pre-commit
pre-commit install


# strongreject
pip install git+https://github.com/dsbowen/strong_reject.git@main # usage: import strong_reject

# Hydra
pip install hydra-core

# needed in run.sh
pip install almost_unique_id
pip install colorlog

# langfuse
pip install langfuse
pip install 'smolagents[telemetry]'