import time
import ivy
from graph_compiler import compile
from transpiler.transpiler import transpile
import numpy as np
import torch
import jax
import transformers
from transformers import AutoModel
from optimum.utils import DummyAudioInputGenerator, DummyVisionInputGenerator, NormalizedVisionConfig
import traceback
from colorama import Fore, Style
import csv

vision_input_tasks = ["image-to-text", "visual-question-answering", "document-question-answering", "depth-estimation", "image-classification", "object-detection", "image-segmentation", "image-to-image", "unconditional-image-generation", "video-classification", "zero-shot-image-classification"]
audio_input_tasks = ["automatic-speech-recognition", "audio-to-audio", "audio-classification", "voice-activity-detection"]

# Model Preparation Functions
def build_model(ckpt):
    try:
        return AutoModel.from_pretrained(ckpt)
    except Exception:
        raise Exception("Model Building Failed")

def create_inputs(model, task, batch_size):
    try:
        if task in vision_input_tasks:
            inputs = {}
            inputs[model.main_input_name] = DummyVisionInputGenerator(task=task, normalized_config=NormalizedVisionConfig(model.config), random_batch_size_range=(batch_size, batch_size+1)).generate(input_name=model.main_input_name,framework= "pt")
            return inputs 
        elif task in audio_input_tasks:
            inputs = {}
            inputs[model.main_input_name] = DummyAudioInputGenerator(task=task, normalized_config=model.config, random_batch_size_range=(batch_size, batch_size+1)).generate(input_name=model.main_input_name,framework= "pt")
            return inputs
        else:
            id = model.main_input_name
            expanded_dummy = model._expand_inputs_for_generation(expand_size=batch_size, input_ids=model.dummy_inputs.get(id))
            return {id: expanded_dummy[0]}

    except Exception as e:
        raise Exception(f"Creating Inputs Failed {e}")

# Compiling Functions
def compile_model(model, model_input):
    try: 
        compiled_graph = compile(model, kwargs=model_input)
        return compiled_graph
    except Exception as e:
        print(Fore.RED + f"{model.name_or_path} Compilation Failed." + Fore.RESET)
        print("Exception: ", e)
        print("Traceback: ", traceback.format_exc())
        raise Exception("Compilation Failed")

def transpile_model(compiled_graph, model_input, to):
    try:
        transpiled_graph = transpile(compiled_graph, source=backend, to=to, kwargs=model_input)
        return transpiled_graph
    except Exception as e:
        print(Fore.RED + f"{model.name_or_path} Transpilation to {to} failed." + Fore.RESET)
        print("Exception: ", e)
        print("Traceback: ", traceback.format_exc())
        raise Exception("Transpilation Failed")
    
def get_original_results(model, model_input):
    try:
        orig = model(**model_input)
        return orig
    except Exception as e:
        raise Exception(f"Original Model Failed. Type: {(type(e).__name__)}")
    

if __name__ == "__main__":
    ivy.set_torch_backend()
    jax.config.update('jax_enable_x64', True)
    backend = "torch"
    framework = "pytorch"
    batch_size = 1
    task="video-classification"
    ckpt = "microsoft/xclip-base-patch32"

    print(Fore.YELLOW + f"Getting model and inputs for {ckpt.upper()}" + Fore.RESET)
    try:
        model = build_model(ckpt)

        model_input = create_inputs(model, task, batch_size=batch_size)

        print("Compiling...")
        compiled_graph = compile_model(model, model_input)

        print("Transpiling to Torch...")
        transpiled_graph_torch = transpile_model(compiled_graph, model_input, to="torch")

        print("Transpiling to Jax...")
        transpiled_graph_jax = transpile_model(compiled_graph, model_input, to="haiku")
        
    except Exception as e:
        print(e) 
