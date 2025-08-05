# NOTES

```bash

conda create -q -y -n test_RoundRobinGroupChat python=3.11
conda activate test_RoundRobinGroupChat

conda deactivate
conda env remove -q -y -n test_RoundRobinGroupChat

conda install -q -y pytorch::pytorch

pip install "autogen-agentchat"
pip install "autogen-core"
pip install "autogen-ext[openai]"
pip install "autogen-ext[web-surfer]"

pip install pytest-playwright
playwright install --with-deps
playwright install chrome

# pip install -U autogenstudio
# autogenstudio ui --host 0.0.0.0 --port 8080 --appdir ./myapp

./run.sh
```
