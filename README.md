# Ekigambo: a speech-based LLM assistant!

## Scenario

A 40-year-old man stands at the door of his house, eyes closed and listening intently through the downpour. The Village Health Team rep can usually be heard from a far off thanks to his rattling motorbike, but not today. The man faces back indoors and shuts the door behind him. "Betulinda baluddewo. Sija kufiirwa mukyala n'omwana wange!(The VHT has delayed, and I'm not about to lose my wife and baby!)" He paces the dim-lit room decidedly toward his bed-laden wife who is talking to him in between huffs and puffs, "aiMusawo agambye nyik'esuuka mu mazi agabuguma weetegeke. Omwana ajja! (The aiDoctor has said to wet the towel in warm water now. Get ready, the baby's coming!)" He kneels down to get the small saucepan of water off the charcoal stove and reaches out to hold her hand. She squeezes his hand and grimaces in pain. It's time. He guides the baby out of her as she pushes, finally falling back on her head after her last push. "Ake?" He calls out, but she's too exhausted to give him a response. He feels around the bed for her phone that still has aiDoctor open and shouts into it, "Omwana azze! Omwana azze! Nkole ki kati? Ake alinga azilise! (The baby is out! What should I do now? Ake seems faint!)" The phone beeps and boops and responds, "Tofaayo, omwana musyonje, omusabike mu ssuka gyewategese. Awo, funa ejirita etukula osale ewuzi emugase ku Ake. (Don't worry, clean the baby and wrap him in the prepared clean sheets. After that, use a clean razor to clip the umbilical cord.)" He wrapped up the baby with tears rolling down his eyes. "Ake? Omwana azze! (Ake? The child has come!") He squeezed her hand and she squeezed back lightly. "Ake akooye? Nze silaba, sisobola kusala kawuzi. (Is Ake tired? She should cut his cord since I'm blind.)" he spoke into the phone again. It responded, "Ake akooye. Muwe buwi ku mazi g'okunywa. (Ake is a bit tired, give her a drink of water.)" He turned around with the baby in hand, and realised he'd have to leave the baby to go get Ake a drink. Just then, the downpour outside develops into a rhythmic rattle. A moment later, he hears a knock at the door.

## Why does this project exist?

I started this project to immerse myself in the world of open-source AI models. My goal is to add accessibility to LLM interaction using speech (hence the real-world scenario above that includes BLV in the usecases). I wonder if I can build a platform where I can dictate my journal entry, and have it transcibed, and stored securely. Then I want it processed by the LLM assistant and the feedback is given to me as an audio I can listen to.
It would entail:

- speech to text model
- llm to do sentiment analysis, and response given my requirements which I will give it beforehand (perhaps in the form of template string or prompt)
- text to speech model that will 'speak' back to me what the LLM has generated.

## How to set it up

### setting up a virtual envirionment

I'm using python venv 3.11.7
setting up one venv on cmd
`python -m venv name_of_virtual_env`
if you're using vscode, install the Python extension and set up the venv inside your project folder. This way, VS Code will run the venv automatically when you run a Python file.

### downloading the dependencies

I used a requirements file because it's cleaner for me to keep track of. I added comments to it to let me know which model needs which deps since I'm using more than one model.
`pip install -r .\requirements.txt`
I also added a `req_list_from_pip.txt` file that has all the versions, even of dependency dependencies.

### downloading the transformers

```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

### installing an older torch version to avoid all the problems I ran into

`pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118`
from https://pytorch.org/get-started/previous-versions/

### symlink to move cache to another location

`mklink /J C:\users\.cache\huggingface E:\boyd_cache\huggingface`

### ffmpeg install for the text to speech [currently using bark]

https://www.wikihow.com/Install-FFmpeg-on-Windows to download and install
then `ffmpeg` and `ffmpeg-python` added to the deps

## tracking the model tests:

- flan was for translation. works well but base model is not good.
- phi is for text gen. not great output.
- dolly was better but also not good output.
- zephyr gave excellent output, so now I need to learn to adjust it to my needs.

## snapshot of resource usage:

1. Whisper Speech to Text

- Result: `So, is the Christian Bible in less than 50 words? [was supposed to be 'Summarise the Christian Bible in less than 50 words'`
- Time taken: `STT finished in 0.0471 minutes`

2. Zephyr LLM

- Result: `No, the Christian Bible is not in less than 50 words. Aa... And even if it was, it's way too big for just 50 words! The Christian Bible has two main parts: the Old Testament and the New Testament. The Old Testament has 39 books, and the New Testament has 27 books. Togeth...er, there are 66 books and some extra things like prophecies, letters, list of books, and things people kept saying about the books. If you put them all together, it would be more than 50 words!`
- Time taken: `LLM Finished in 2.9646 minutes`

3. Bark Text to Speech

- Result: <audio src="bark_out_20240106_022056.wav" controls title="Bark TTS result"></audio>
- Time taken: `TTS Finished in 4.83 minutes`

4. Resource usage
   ![Screenshots showing laptop baseline and load numbers for CPU, GPU, and RAM](/images/start.jpg "Baseline to Under-Load statistics")
   ![Screenshots showing laptop baseline and load numbers for CPU, GPU, and RAM](/images/end.jpg " Under-Load to Baseline statistics")

## thoughts thus far / todo list:

- ✅all experiments have been on commandline. If I load the model into gradio, then maybe it will not need to cold boot so I'll get faster subsequent responses?
  - sorted this out. the model now remains in VRAM until the gradio is closed and server is terminated (Ctrl+C in terminal).
- 📝I need to find a way to read entire paragraphs with bark. it's currently not completing ALL the input I give to it.
  - bark is currently the weakest link in the chain. takes ages and hallucinates the for a mere two sentences. need to look for work-around or replacement.
  - found this [espnet repo] (https://github.com/espnet/espnet/) that might be a better alternative to bark. haven't yet tested it and might need to return this laptop before I get to. We'll see.
- ✅then I need to add speech to text so that we can speak the query rather than type it. [working on this next] [done, added whisper-base which is super fast and accurate!]
- ✅also need to add actual versions of deps in requirements.txt because the deps whack-a-mole game is NOT fun!
- ✅also want to screenshot the system on idle and then running one, two, and three models to see resource usage. That'll be fun, I think.
- given the above point, I will need to modularise model imports so that I can test each independently without having to comment out code. This will also help me not have 3 models hogging all the VRAM when I'm not using them.
  - [x] done this for just the LLM. The concept works but I'll need to rework my gradio so that all models are swappale without causing exceptions.
  - [x] done it for all. Now just need to activate the textbox inputs when mic-recorded audio is not available.
- [ ] finally [the real goal]:
  - [ ] adding speech-to-text that listens to Luganda and transcibes (then translates) it well. try Mozilla-based models or the Meta-nllb one?
  - [ ] adding text-to-speech after the LLM has responded. TTS should respond back in Luganda! try [MMS from fairseq]? (https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
