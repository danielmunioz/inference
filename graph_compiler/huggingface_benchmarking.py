import time
import ivy
from time import perf_counter
from graph_compiler import compile
from transpiler.transpiler import transpile
import numpy as np
import torch
import jax
import transformers
from huggingface_hub import HfApi, list_files_info
from transformers import AutoModel
from optimum.utils import DummyAudioInputGenerator, DummyVisionInputGenerator, NormalizedVisionConfig
import traceback
from colorama import Fore, Style
import csv
import math
import pandas as pd
import re
import os
import argparse



vision_input_tasks = ["vision", "image-to-text", "visual-question-answering", "document-question-answering", "depth-estimation", "image-classification", "object-detection", "image-segmentation", "image-to-image", "unconditional-image-generation", "video-classification", "zero-shot-image-classification"]
audio_input_tasks = ["audio", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "voice-activity-detection"]

# Model Preparation Functions
def task_filtering(models_batch, mode="equal"):
    # Counting the number of models to test per sub-task type according to a desired weighting scheme
    # Default mode "equal" weights all tasks equally, "frequency" mode weights the tasks according to their frequency ratios in HF
    
    model_counts = {}
    with open("/workspaces/graph-compiler/hf_models_testing/HF_torch_models.txt", 'r') as f:
        for line in f:
            items = line.strip().split(',')
            if len(items) > 0:
                first_item = items[0]
                if first_item in list(model_counts.keys()):
                    model_counts[first_item] += 1 if mode=="frequency" else 0
                else:
                    model_counts[first_item] = 1

    total = sum(model_counts.values())
    for key in model_counts:
        model_counts[key] *= (models_batch/total)
        if isinstance(model_counts[key], float):
            model_counts[key] = math.ceil(model_counts[key])

    return model_counts

def models_sample(models_batch, mode="equal"):
    model_counts = task_filtering(models_batch=models_batch, mode=mode)

    with open('/workspaces/graph-compiler/hf_models_testing/HF_torch_models.txt', 'r') as source_file:
        reader = csv.reader(source_file)
        rows = [row for row in reader]

    selected_rows = []
    for key, count in model_counts.items():
        rows_with_key = [row for row in rows if row[0] == key]
        selected_rows.extend(rows_with_key[:count])

    with open(f'/workspaces/graph-compiler/hf_models_testing/{mode}_sorted_hf_torch_models.txt', 'w') as sorted_file:
        writer = csv.writer(sorted_file)
        writer.writerows(selected_rows)

def exclude_tested_models(file):
    # Fetching Previously Tested Models To Avoid Repetition
    with open(file) as f:
        f.readline()
        previously_tested = [(row.split(",")[0], row.split(",")[1]) for row in f]
        f.close()
        return previously_tested

def set_experiment(file_models, file_results):
    previously_tested = exclude_tested_models(file_results)
    with open(file_models) as f:
        f.readline()
        for line in f:
            row = line.split(",")
            model, batch_size, task = row[0], row[1], row[2].split("\n")[0]
            if (model, batch_size) not in previously_tested:
                f.close()
                return model, int(batch_size), task
        f.close()
    
def get_model_info(ckpt):
    
    #hf_api = HfApi()
    #info = hf_api.model_info(repo_id=ckpt)
    #auto_model_class = info.transformersInfo.get("auto_model")
    #dataset = info.cardData.get("model-index")[0].get("results")[0].get("dataset")
    #architecture = info.config.get("architectures")
    #downloads = info.downloads
    #gated = info.gated
    files_info = list_files_info(ckpt)
    size = max([info.size for info in files_info])
    
    return size

def build_model(model_name):
    try:
        return AutoModel.from_pretrained(model_name)
    except Exception as e:
        raise Exception("Model Building Failed {(type(e).__name__)}")

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
        raise Exception(f"Compilation Failed. Type: {(type(e).__name__)}")
    
def transpile_model(compiled_graph, model_input, to):
    try:
        transpiled_graph = transpile(compiled_graph, source=backend, to=to, kwargs=model_input)
        return transpiled_graph
    except Exception as e:
        raise Exception(f"Transpilation to {to} Failed. Type: {(type(e).__name__)}")
    
def get_original_results(model, model_input):
    try:
        orig = model(**model_input)
        return orig
    except Exception as e:
        raise Exception(f"Original Model Failed. Type: {(type(e).__name__)}")

def log_benchmark_results(graph, times, results):
    unique, counts = np.unique(results, return_counts=True)
    return [len(graph._functions), np.mean(times), np.std(times), np.asarray((unique, counts)).T]

def log_original_results(times):
    return [np.mean(times), np.std(times)]

def check_passing(orig, comparable, to):
    output_class = orig.__class__
    class_params = dir(output_class)

    check = torch.allclose if to == "torch" else np.allclose
    if "last_hidden_state" in class_params:
        last_hidden_state = orig.last_hidden_state if to == "torch" else orig.last_hidden_state.detach().numpy()
        all_close = check(last_hidden_state, comparable.last_hidden_state, atol=1e-4)

    elif "end_logits" in class_params:
        all_close = check(orig.end_logits, comparable.end_logits)

    elif "logits" in class_params:
        all_close = check(orig.logits, comparable.logits)

    elif "predicted_depth" in class_params:
        all_close = check(orig.predicted_depth, comparable.predicted_depth)

    elif "sequences" in class_params:
        all_close = check(orig.sequences, comparable.sequences)

    elif "spectrogram" in class_params:
        all_close = check(orig.spectrogram, comparable.spectrogram)

    else:
        all_close = check(orig.encoder_last_hidden_state, comparable.encoder_last_hidden_state)

    if all_close:
        return 1
    else:
        return 0

def check_passing_jit(orig, comparable, to):
    output_class = orig.__class__
    class_params = dir(output_class)

    check = torch.allclose if to == "torch" else np.allclose
    if "last_hidden_state" in class_params:
        last_hidden_state = orig.last_hidden_state if to == "torch" else orig.last_hidden_state.detach().numpy()
        all_close = check(last_hidden_state, comparable, atol=1e-4)

    elif "end_logits" in class_params:
        all_close = check(orig.end_logits, comparable)

    elif "logits" in class_params:
        all_close = check(orig.logits, comparable)

    elif "predicted_depth" in class_params:
        all_close = check(orig.predicted_depth, comparable)

    elif "sequences" in class_params:
        all_close = check(orig.sequences, comparable)

    elif "spectrogram" in class_params:
        all_close = check(orig.spectrogram, comparable)

    else:
        all_close = check(orig.encoder_last_hidden_state, comparable)

    if all_close:
        return 1
    else:
        return 0

def save_benchmark_results(results, file):
    with open(file, 'a', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(results)
        file.close()

def benchmarking_iterations(
        mode,
        model, 
        compiled_graph,   
        transpiled_graph_torch, 
        transpiled_graph_jax, 
        orig,
        n_iterations
):
    compiled_times, compiled_results, transpiled_times_torch, transpiled_results_torch, transpiled_times_jax, transpiled_results_jax, orig_times = (np.zeros(shape=n_iterations) for _ in range(7))
    

    jix_graph, jax_inputs = create_jit_graph(transpiled_graph_jax, model_input)
    try: 
        for i in range(n_iterations):
            start =  time.time()
            _ = comp_model(**model_input)
            end = time.time()  - start
            orig_times[i] = end

            start =  time.time()
            comp = compiled_graph(**model_input)
            end = time.time()  - start
            compiled_times[i] = end
            compiled_results[i] = check_passing(orig, comp, "torch")

            start =  time.time()
            transp_torch = transpiled_graph_torch(**model_input)
            end = time.time() - start
            transpiled_times_torch[i] = end
            transpiled_results_torch[i] = check_passing(orig, transp_torch, "torch")

            start =  time.time()
            transp_jax = jix_graph(jax_inputs).block_until_ready
            end = time.time() - start
            transpiled_times_jax[i] = end
            transpiled_results_jax[i] = check_passing_jit(orig, transp_jax, "jax")


        comp_results = log_benchmark_results(compiled_graph, compiled_times, compiled_results) 
        torch_results = log_benchmark_results(transpiled_graph_torch, transpiled_times_torch, transpiled_results_torch) 
        jax_results = log_benchmark_results(transpiled_graph_jax, transpiled_times_jax, transpiled_results_jax) 
        orig_results = log_original_results(orig_times) 

        save_benchmark_results(mode, task, model.name_or_path, batch_size, orig_results, comp_results, torch_results, jax_results)
        delete_things_from_memory(model, compiled_graph, transpiled_graph_torch, transpiled_graph_jax, orig, jix_graph)
    
    except Exception as e:
        raise Exception(f"Benchmarking Failed. Error: {e}")

def create_jit_graph(transpiled_graph, model_input):
    try:
        jit_inputs = {}
        for key, value in model_input.items(): 
                jit_inputs[key] = jax.numpy.array(value.cpu().numpy())
        def fn(x):
            return transpiled_graph(**x).last_hidden_state
        jix_graph = jax.jit(fn)
        return jix_graph, jit_inputs
    except Exception as e:
        print(e)
        raise Exception(f"JIT Compilation Failed.{e}")

def save_terminated_models(mode, task, model_name, batch_size):
    with open(f'hf_models_testing/{mode}_sorted_terminated_models.txt', 'a', newline='\n') as file:
        writer = csv.writer(file)
        terminated = [task, model_name] + [batch_size]
        writer.writerow(terminated)
        file.close()

def remove_terminated_models(mode, task, model_name, batch_size):

    filename = f"/workspaces/graph-compiler/hf_models_testing/{mode}_sorted_terminated_models.txt"

    with open(filename, "r") as f:
        temp = []
        reader = csv.reader(f)
        for row in reader:
            chain = ','.join(item for item in row)
            if chain != f"{task},{model_name},{batch_size}":
                temp.append(chain)
        f.close()

    os.remove(filename)

    with open(filename, "w") as f:
        f.writelines([item + '\n' for item in temp])

def save_failed_models(mode, task, model_name, batch_size, timing):
    file = "frequency_sorted_failed_models" if mode == "frequency" else "equal_sorted_failed_models"
    print(Fore.BLUE+f"{timing}. Check {mode}_sorted_failed_models.txt for more details"+Fore.RESET)
    with open(f'hf_models_testing/{mode}_sorted_failed_models.txt', 'a', newline='\n') as file:
        writer = csv.writer(file)
        results = [task, model_name] + [batch_size] + [timing]
        writer.writerow(results)
        file.close()

def delete_things_from_memory(model, compiled_graph, transpiled_graph_torch, transpiled_graph_jax, orig, jix_graph):
    torch.cuda.empty_cache()
    model = None
    compiled_graph = None
    transpiled_graph_torch = None
    transpiled_graph_jax = None
    orig = None
    jix_graph = None
    del model, compiled_graph, transpiled_graph_jax, transpiled_graph_torch, orig, jix_graph

def benchmark_for_paper(batch_size, model):
    task, ckpt, batch_size = set_experiment(large_batch_size=batch_size, mode=mode)

    save_terminated_models(mode=mode, task=task, model_name=ckpt, batch_size=batch_size)

    print(Fore.YELLOW + f"Getting model, inputs and batch size for {ckpt.upper()}" + Fore.RESET)
    try:
        model = build_model(ckpt)

        model_input = create_inputs(model, task, batch_size=batch_size)

        print("Compiling...")
        compiled_graph = compile_model(model, model_input)

        print("Transpiling to Torch...")
        transpiled_graph_torch = transpile_model(compiled_graph, model_input, to="torch")

        print("Transpiling to Jax...")
        transpiled_graph_jax = transpile_model(compiled_graph, model_input, to="haiku")

        print("Benchmarking...")
        orig = model(**model_input)
        comp_model = torch.compile(model)
        
        benchmarking_iterations(
            mode,
            comp_model, 
            compiled_graph,
            transpiled_graph_torch,
            transpiled_graph_jax, 
            orig, 
            n_iterations=10
        )

        remove_terminated_models(mode, task, ckpt, batch_size)

    except Exception as timing:
        save_failed_models(mode, task, ckpt, batch_size, timing)
        remove_terminated_models(mode, task, ckpt, batch_size)

def benchmark_for_release(model_file, result_file):
    model_name, batch_size, task = set_experiment(model_file, result_file)
    print(Fore.YELLOW + f"Testing model {model_name}" + Fore.RESET)
    try:
        model = build_model(model_name)

        model_input = create_inputs(model, task, batch_size=batch_size)

        #torch_compiled = torch.compile(model.__call__)
        #torch_compiled(**model_input)

        #torch_times = []
        #for _ in range(10):
        #    s = perf_counter()
        #    torch_compiled(**model_input)
        #    torch_times.append(perf_counter() - s)
        #torch_avg = sum(torch_times) / len(torch_times)

        graph = transpile(model.__call__, source="torch", to="jax", kwargs=model_input)

        def fn(x):
            return graph(**x).last_hidden_state
        
        jit_inputs = {}
        for key, value in model_input.items(): 
                jit_inputs[key] = jax.numpy.array(value.cpu().numpy())
        jitted = jax.jit(fn)
        jitted(jit_inputs)

        jax_times = []
        for _ in range(10):
            s = perf_counter()
            jitted(jit_inputs).block_until_ready()
            jax_times.append(perf_counter() - s)
        jax_avg = sum(jax_times) / len(jax_times)
        
        #speed = torch_avg / jax_avg

        #print(Fore.GREEN +f"Result {speed}"+ Fore.RESET)
        print(Fore.GREEN +f"Result {jax_avg}"+ Fore.RESET)
        #model,batch,result,speed
        #results = [model_name, batch_size,"Success",speed]
        results = [model_name, batch_size,"Success",jax_avg]
        save_benchmark_results(results, result_file)


    except Exception as e:
        with open(results_file, 'a', newline='\n') as file:
            writer = csv.writer(file)
            #model,batch,result,speed
            results = [model_name, batch_size,"Failed",0.0]
            writer.writerow(results)
            file.close()

if __name__ == "__main__":
    ivy.set_torch_backend()
    jax.config.update('jax_enable_x64', True)
    backend = "torch"
    framework = "pytorch"                         

    parser = argparse.ArgumentParser(description='Benchmarking hf models')
    parser.add_argument('--demos', action='store_true',
                        help='run the demos models for the release')
    args = parser.parse_args()

    if args.demos:
        results_file = "/workspaces/graph-compiler/hf_models_testing/demos_results.txt"
        models_file = "/workspaces/graph-compiler/hf_models_testing/demos_models.txt"
        benchmark_for_release(models_file, results_file)
    else:
        raise NotImplementedError("Please use the --demos flag")
    
    