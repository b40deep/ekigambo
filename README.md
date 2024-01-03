setting up a virtual envirionment
i'm using python venv 3.11.7
setting up one venv on cmd 
python -m venv name_of_virtual_env
if you're using vscode, install the Python extension and set up the venv inside your project folder. This way, VS Code will run the venv automatically when you run a Python file.

downloading the dependencies
I used a requirements file because it's cleaner for me to keep track of. I added comments to it to let me know which model needs which deps since I'm using more than one model.
pip install -r .\requirements.txt

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
    - sorted this out. the model now remains in VRAM until the gradio is closed and server is terminated (Ctrl+C in terminal). 
- I need to find a way to read entire paragraphs with bark. it's currently not completing ALL the input I give to it.
- then I need to add speech to text so that we can speak the query rather than type it.