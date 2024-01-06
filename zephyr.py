from transformers import AutoModelForCausalLM, AutoTokenizer
import time
# for whisper stt
from transformers import pipeline
# for bark tts
from transformers import AutoProcessor, BarkModel
import scipy 

llm_tokenizer = None
llm_model = None

def loadLLM():
    print('loading LLM works')
    global llm_tokenizer 
    global llm_model

    llm_tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
    llm_model = AutoModelForCausalLM.from_pretrained(
        'stabilityai/stablelm-zephyr-3b',
        trust_remote_code=True,
        device_map="auto"
    )

def loadTTS():
    print('loading TTS works')
def loadSTT():
    print('loading STT works')

def runLLM(input_query):
    print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runLLM     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')

    # query = 'List 3 synonyms for the word "tiny"'
    # query = 'did Jesus weep? give proof.'
    # query = 'what verses in the Bible support predestination?'
    query = 'list all the attributes of the Christian God, with one sentence explanation and supporting Bible verses for each attribute.'
    query = input_query

    prompt = [{'role': 'user', 'content': query}]
    inputs = llm_tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt'
    )

    tic = time.perf_counter()

    tokens = llm_model.generate(
        inputs.to(llm_model.device),
        max_new_tokens=1024,
        temperature=0.8,
        do_sample=True
    )

    toc = time.perf_counter()

    # result = llm_tokenizer.decode(tokens[0], skip_special_tokens=False)
    result = llm_tokenizer.decode(tokens[0], skip_special_tokens=True)

    print(f"████████████████ LLM Finished in {(toc - tic)/60:0.4f} minutes ████████████████")

    print(f"################  result, {result}")
    # regex
    import re
    # pattern = re.compile(r'<\|(.*?)\|>(.*)')
    pattern = re.compile(r'<\|assistant\|>')
    matches = pattern.split(result)
    print(f"#################  matches {matches}")
    result = matches[1]

    print(f"################ cleaned result: { result}")

    return result


#####################################################################
#####################     whisper stt      ##########################
#####################################################################

whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")

def runSTT(audio):
    print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runSTT     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')
    tic = time.perf_counter()
    result = whisper_model(audio)["text"]
    toc = time.perf_counter()
    print(f"################ stt result: { result}")
    print(f"████████████████ STT finished in {(toc - tic)/60:0.4f} minutes ████████████████")
    return result


#####################################################################
#####################     bark tts      #############################
#####################################################################
# bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
# bark_model = BarkModel.from_pretrained("suno/bark-small")
# voice_preset = "v2/en_speaker_6"

# def runTTS(input_query):
#     print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runTTS     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')
#     inputs = bark_processor(input_query, voice_preset=voice_preset)
#     tic = time.perf_counter()
#     audio_array = bark_model.generate(**inputs)
#     toc = time.perf_counter()
#     print(f"████████████████ TTS generated in {(toc - tic)/60:0.4f} minutes ████████████████")
#     audio_array = audio_array.cpu().numpy.squeeze()
#     sample_rate = bark_model.generation_config.sample_rate
#     scipy.io.wavfile.write("test.wav", rate=sample_rate, data=audio_array)
#     tuk = time.perf_counter()
#     print(f"████████████████ TTS Finished in {(tuk - toc)/60:0.4f} minutes ████████████████")
#     return 'test.wav'

from transformers import AutoProcessor, BarkModel
from datetime import datetime
import scipy
import os
import torch
import time

# Start the clock
start_time = time.time()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Using CPU")

# Settings (If you need them)
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# Load processor and model
bark_processor = AutoProcessor.from_pretrained("suno/bark")
bark_model = BarkModel.from_pretrained("suno/bark")

# Move model to the device (CPU or GPU)
bark_model.to(device)

voice_preset = "v2/en_speaker_6"

def runTTS2(input_query):
    print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runTTS2     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')

    # Process text input
    inputs = bark_processor(input_query if len(input_query)>2 else "The James Webb Space Telescope has captured stunning images of the Whirlpool spiral galaxy, located 27 million light-years away from Earth.", voice_preset=voice_preset)
    # Move inputs to the same device as the model
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    # add attention mask here
    attention_mask = inputs["attention_mask"]
    tic = time.perf_counter()

    # Generate audio
    audio_array = bark_model.generate(input_ids=inputs["input_ids"], 
    attention_mask=attention_mask)
    audio_array = audio_array.cpu().numpy().squeeze()

    toc = time.perf_counter()
    print(f"████████████████ TTS generated in {(toc - tic)/60:0.4f} minutes ████████████████")

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string: YYYYMMDD_HHMMSS
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    # Use the timestamp as part of the filename
    filename = f"bark_out_{timestamp_str}.wav"

    # Save as WAV file
    sample_rate = bark_model.generation_config.sample_rate
    scipy.io.wavfile.write(filename, rate=sample_rate, data=audio_array)

    # Stop the clock and print the elapsed time
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"The process took {elapsed_time:.2f} minutes.")

    tuk = time.perf_counter()
    print(f"████████████████ TTS2 Finished in {(tuk - tic)/60:0.4f} minutes ████████████████")

    return filename


# def runCombined(ui_input):
#     print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runCombined     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')
#     to_llm = runSTT(ui_input)
#     to_tts = runLLM(to_llm)
#     # to_tts=ui_input
#     result = runTTS2(to_tts)
#     # return 'bark_out_20240103_201332.wav'
#     return result

def runCombined(mic, models):
    print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runCombined     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')
    if len(models)>0:
        for item in models:
            loadSTT() if item == 'STT' else loadLLM() if item == 'LLM' else loadTTS() if item == 'TTS' else None

    to_llm = runSTT(mic)
    to_tts = runLLM(to_llm)
    # to_tts=ui_input
    result = runTTS2(to_tts)
    # return 'bark_out_20240103_201332.wav'
    return result

#####################################################################
#####################     gradio      ###############################
#####################################################################

import gradio as gr
in_mic = gr.Audio(sources=["microphone"],type="filepath")
in_models = gr.CheckboxGroup(["STT", "LLM", "TTS"], label="Models", info="Which model to load?")
iface = gr.Interface(fn=runCombined,inputs=[in_mic,in_models],outputs="audio",live=False)
iface.launch(debug=True)