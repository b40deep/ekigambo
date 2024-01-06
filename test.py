print ('it works!')

from transformers import pipeline

# model = pipeline("text-generation", model="microsoft/phi-2")

# res = model("i'm so glad this disease is going to kill me")

# print(res)

LLM = False

def loadLLM(val):
    LLM = val
    print("#######################")
    print(LLM)
    print("#######################")

def runCombined(mic, models):
    global loadLLM
    print('╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦     runCombined     ╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦╦')
    print(models)
    if len( models)>0:
        for item in models:
            print('////////////////////// STT'+item) if item == 'STT' else print('////////////////////// LLM'+item) if item == 'LLM' else print('////////////////////// TTS'+item) if item == 'TTS' else None

    else:
        loadLLM(True)
    # loadLLM = True if models[0]
    # to_llm = runSTT(ui_input)
    # to_tts = runLLM(to_llm)
    # to_tts=ui_input
    # result = runTTS2(to_tts)
    # return 'bark_out_20240103_201332.wav'
    result = "bark_out_20240103_201332.wav"
    return result

#####################################################################
#####################     gradio      ###############################
#####################################################################

import gradio as gr
in_mic = gr.Audio(sources=["microphone"],type="filepath")
in_models = gr.CheckboxGroup(["STT", "LLM", "TTS"], label="Models", info="Which model to load?")
iface = gr.Interface(fn=runCombined,inputs=[in_mic, in_models],outputs="audio",live=False)
iface.launch(debug=True)