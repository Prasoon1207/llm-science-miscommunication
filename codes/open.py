import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import argparse


def create_demonstrations(df, n_shots):
    context = open('/home/prasoon/snap/main/mtp/llm-science-miscommunication/analysis/prompts/context_qa.txt', 'r').read()
    example_indices = df.sample(n = n_shots).index
    for index in example_indices:
        row = df.loc[index]
        context += 'Question: ' + row['question'] + '\n' + 'Answer: ' + str(row['answer']) + '\n'
    return (context, example_indices)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input_file", help="path of data", required=True)
    parser.add_argument("-m", "--model_name", help="name of the model as per huggingface (Ex. 'meta-llama/Llama-2-7b-hf')", required=True)
    parser.add_argument("-q", "--required_quantization", help="True/False if required quantization", required=True)
    parser.add_argument("-d", "--device", help='available gpu device', required=True)
    
    args = parser.parse_args()
    
    input_file = args.input_file
    model_name = args.model_name
    required_quantization = args.required_quantization
    device = args.device



    if(required_quantization):
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )


    model_name = sys.argv[1]
    device = sys.argv[2]
    df = pd.read_csv(input_file)




    context = create_demonstrations(df, 4)[0]
    print(context)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if(required_quantization):
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = 'torch.float16')


    predicted_answers_list = []

    for index, row in tqdm(df.iterrows()):
        
        current_question = row['question']
        current_answer = row['answer']
        encoder_input_str = context + 'Question: ' + current_question + '\n'+ 'Answer: '
        expected_answer = row['answer']
        
        inputs = tokenizer(encoder_input_str, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            inputs,
            max_new_tokens = 4,
            return_dict_in_generate = True, 
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        model.reset_mhsa_hidden_states()
        model.reset_up_projections()
        
        generated_token_ids = outputs.sequences[0].tolist()
        input_length = len(inputs.tolist()[0])
        
        context_tokens = inputs[0][:input_length].tolist()
        generated_tokens_ids = generated_token_ids[input_length:]
        
        context_output = tokenizer.decode(context_tokens, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_tokens_ids)
        

        exact_answer = ''
        if(':' in generated_tokens):
            exact_answer = generated_tokens[generated_tokens.index(':') + 1]
        predicted_answers_list.append(generated_tokens[0])
        print(generated_tokens[0])