import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import sys
import pickle
import argparse

from transformers import BitsAndBytesConfig



quantization_config = BitsAndBytesConfig(
        
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        
)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="path of data", required=True)
    parser.add_argument("-m", "--model_name", help="name of the model as per huggingface (Ex. 'meta-llama/Llama-2-7b-hf')", required=True)
    parser.add_argument("-q", "--required_quantization", help="True/False if required quantization", required=True)
    parser.add_argument("-d", "--device_number", help='available gpu device', required=True)
    parser.add_argument("-o", "--output_directory", help='available gpu device', required=True)
    
    

    args = parser.parse_args()

    input_file = args.input_file
    model_name = args.model_name
    required_quantization = args.required_quantization
    device_number = args.device_number
    output_directory = args.output_directory
    
    device = 'cuda'
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = device_number
    
    n_stochastic_samples = 20

    path_to_context_file = '/home/prasoon/snap/main/mtp/llm-science-miscommunication/analysis/prompts/context_qa_eval.txt'
    context = open(path_to_context_file, 'r').read()

    path_to_model_configs = '/home/prasoon/snap/main/mtp/llm-science-miscommunication/model_configs'
    path_to_model_directory = '/home/models/' # <- add path to the directory containing models to be benchmarked
    print(model_name)
    model_path = path_to_model_directory + model_name + '/'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if(required_quantization): model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config = quantization_config, low_cpu_mem_usage = True)
    else: model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, low_cpu_mem_usage = True).to(device)

    df = pd.read_csv(input_file)
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


    
    with open(output_directory + '/' + model_name.replace('/', '-').lower() + '.pkl', 'wb') as f:
        pickle.dump(master_list, f)
        
    del model
    torch.cuda.empty_cache()
