# for smoagents
# first fork the official repo then clone my forked version.
# this should be done in ../smolagents-cs folder
#git clone git@github.com:chenxshuo/smolagents-cs.git
#cd smolagents-cs
#git remote add upstream https://github.com/huggingface/smolagents.git
#git remote -v
#git checkout -b cs-deep-research
#git tag -a cs0.0 -m "init cs0.0" # tag to show the start point of this project

# env

# setup python 3.13 if not; otherwise cannot install smolagents in editable mode.
# on linux
#deactivate
#conda create -n py313_env python=3.13
#conda activate py313_env # rather than base.
# on mac
#brew update
#brew upgrade pyenv
#pyenv install 3.13.2
#pyenv local 3.13.2
#python --version

python -m venv venv-smolagents-deep-research --prompt="smolagents-deep-research"
source venv-smolagents-deep-research/bin/activate
which pip
which python
pip install -r examples/open_deep_research/requirements.txt
pip install -e '.[dev]'


# API:
SERPAPI_API_KEY: https://serpapi.com/manage-api-key
SERPER_API_KEY: https://serper.dev/api-key
openai: https://platform.openai.com/docs/guides/prompt-engineering