# for smoagents
# first fork the official repo then clone my forked version.
# this should be done in ../smolagents-cs folder
git clone git@github.com:chenxshuo/smolagents-cs.git
cd smolagents-cs
git remote add upstream https://github.com/huggingface/smolagents.git
git remote -v
git checkout -b cs-deep-research
git tag -a cs0.0 -m "init cs0.0"