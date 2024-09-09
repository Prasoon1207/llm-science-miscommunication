import os
import sys
import ast
import math
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import openai
import time
import sys

from transformers import BitsAndBytesConfig
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] ='1'

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


model_name = sys.argv[1]
device = sys.argv[2]

df = pd.read_csv('data/data_10.csv')
os.environ['AZURE_OPENAI_ENDPOINT'] = "https://kglm.openai.azure.com/"
os.environ['AZURE_OPENAI_KEY'] = "060db6b6c3ff468ca2215e0ef75b9cc1"

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

# models = openai.Model.list()
# for model in models["data"]:
#     print(model["id"])

deployment_name = 'kglm-text-davinci-003'
model_name = deployment_name.replace('_', '.')

def create_demonstrations(subset_df, n_shots):
    context = open('context_qa.txt', 'r').read()
    example_indices = df.sample(n = n_shots).index
    for index in example_indices:
        row = df.loc[index]
        context += 'Question: ' + row['question'] + '\n' + 'Answer: ' + str(row['answer']) + '\n'
    return (context, example_indices)

context = create_demonstrations(df, 4)[0]
print(df.shape)
print(context)

def generate_gpt3_response(user_text, print_output = False):
    
    completions = openai.Completion.create(
        engine='kglm-text-davinci-003', 
        temperature=0.8,            
        prompt=user_text,          
        max_tokens=1000,             
        n=1,                        
        stop=None,        
    )
    
    if print_output:
        print(completions)

    return completions.choices[0].text

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = quantization_config)


predicted_answers_list = []

for index, row in tqdm(df.iterrows()):
    
    current_question = row['question']
    current_answer = row['answer']
    encoder_input_str = context + 'Question: ' + current_question + '\n'+ 'Answer: '
    expected_answer = row['answer']
    
    # inputs = tokenizer(encoder_input_str, return_tensors="pt").input_ids.to(device)
    # outputs = model.generate(
    #     inputs,
    #     max_new_tokens = 4,
    #     return_dict_in_generate = True, 
    #     output_scores=True,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    
    # model.reset_mhsa_hidden_states()
    # model.reset_up_projections()
    
    # generated_token_ids = outputs.sequences[0].tolist()
    # input_length = len(inputs.tolist()[0])
    
    # context_tokens = inputs[0][:input_length].tolist()
    # generated_tokens_ids = generated_token_ids[input_length:]
    
    # context_output = tokenizer.decode(context_tokens, skip_special_tokens=True)
    # generated_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
    # generated_tokens = tokenizer.convert_ids_to_tokens(generated_tokens_ids)
    

    # exact_answer = ''
    # if(':' in generated_tokens):
    #     exact_answer = generated_tokens[generated_tokens.index(':') + 1]
    # print(exact_answer)
    # predicted_answers_list[index] = exact_answer
    # print(generated_text)
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # print(encoder_input_str)
    # print(generated_tokens[0])
    # predicted_answers_list.append(generated_tokens[0])
    # print(generated_tokens[0])


    while(True):
        try:
            output = generate_gpt3_response(encoder_input_str)
            break
        except:
            time.sleep(3)
            continue
        
    exact_answer = output[-1]
    predicted_answers_list.append(exact_answer) # -> for gpt models
    print(exact_answer)
        


if __name__ == "__main__":
    
    
    df.to_csv('unverified_data/v3/closed.csv', index = False)
    
    
# df[model_name.replace('/', '_') + '_few_shot4_run5'] = predicted_answers_list

