import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import openai
import sys
import pickle

from transformers import BitsAndBytesConfig
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] ='2'

quantization_config = BitsAndBytesConfig(
    
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    
)

n_stochastic_samples = 20

path_to_context_file = '/home/prasoon/snap/main/mtp/context_qa_eval.txt'
context = open(path_to_context_file, 'r').read()

device = sys.argv[1]
path_to_model_directory = '/home/models/'
path_to_model_configs = '/home/prasoon/snap/main/mtp/llm-science-miscommunication/model_configs'

for model_name in ['Llama-2-7b-hf']:

    print(model_name)
    model_path = path_to_model_directory + model_name + '/'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config = quantization_config, low_cpu_mem_usage = True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, low_cpu_mem_usage = True).to(device)

    df = pd.read_csv('/home/prasoon/snap/main/mtp/data/data.csv')
    master_list = []
    
    for index, row in tqdm(df.iterrows()):
        
        current_question = row['question']
        current_answer = row['answer']
        encoder_input_str = context + 'Question: ' + current_question + '\n'+ 'Answer: '
        expected_answer = row['answer']
        
        outputs_dict = {}
        
        inputs = tokenizer(encoder_input_str, return_tensors="pt").input_ids.to(device)
        main_generation_config = GenerationConfig.from_pretrained(path_to_model_configs, "main_generation_config.json")
        output = model.generate(
            inputs,
            generation_config = main_generation_config
        )
        
        generated_token_ids = output.sequences[0].tolist()
        input_length = len(inputs.tolist()[0])
        context_tokens = inputs[0][:input_length].tolist()
        generated_tokens_ids = generated_token_ids[input_length:]
        context_output = tokenizer.decode(context_tokens, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
        
        outputs_dict['main'] = {'generated_text': generated_text}
        
        for index in range(n_stochastic_samples):
            
            inputs = tokenizer(encoder_input_str, return_tensors="pt").input_ids.to(device)
            stochastic_generation_config = GenerationConfig.from_pretrained(path_to_model_configs, "stochastic_generation_config.json")

            output = model.generate(
                inputs,
                generation_config = stochastic_generation_config
            )
            
            generated_token_ids = output.sequences[0].tolist()
            input_length = len(inputs.tolist()[0])
            context_tokens = inputs[0][:input_length].tolist()
            generated_tokens_ids = generated_token_ids[input_length:]
            context_output = tokenizer.decode(context_tokens, skip_special_tokens=True)
            generated_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
        
            outputs_dict['stochastic_'+str(index)] = {'generated_text' : generated_text}   
            
        master_list.append(outputs_dict)


    path_to_save = '/home/prasoon/snap/main/mtp/results/evaluations/'
    
    with open(path_to_save + model_name.replace('/', '-').lower() + '.pkl', 'wb') as f:
        pickle.dump(master_list, f)
        
    del model
    
    torch.cuda.empty_cache()