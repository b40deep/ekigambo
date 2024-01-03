setting up a virtual envirionment

downloading the dependencies

downloading the transformers
git clone https://github.com/huggingface/transformers.git 
cd transformers
pip install -e .

installing an older torch version to avoid all the problems I ran into
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
from https://pytorch.org/get-started/previous-versions/

symlink to move cache to another location
mklink /J C:\users\.cache\huggingface E:\boyd_cache\huggingface

tracking the model tests:
- flan was for trasnlation. works well but base model is not good.
- phi is for text gen. not great output.
- dolly was better but also not good output.
- zephyr gave excellent output, so now I need to learn to adjust it to my needs.

thoughts:
- all experiments have been on commandline. If I load the model into gradio, then maybe it will not need to cold boot so I'll get faster subsequent responses?