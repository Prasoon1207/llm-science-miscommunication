from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import torch
import torch.nn.functional as F
import openai
import time
import os
from nltk import ngrams
from collections import defaultdict
from os import listdir
import openai
import os
from nltk import ngrams
from collections import defaultdict
import sys
import argparse


nlp = spacy.load("en_core_web_sm")

def find_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
    

def sentence_slit(passage):
    global nlp
    doc = nlp(passage)
    sentences = [sent.text for sent in doc.sents]
    return sentences


# create appropriate client here
def generate_gpt_response(prompt, number_of_tokens, model_name, temperature = 0.0):
    
    completion = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens = number_of_tokens,
        temperature = temperature
    )
    
    return completion.choices[0].message

def score_mapper_prompt_response(response):
    response = response.lower().strip('\n').strip()
    if(response == 'yes'): return 0.0
    elif(response == 'no'): return 1.0
    else: return 0.5
    
def prompt_scorer(data):
    context = open('/home/prasoon/snap/main/mtp/llm-science-miscommunication/analysis/prompts/self-check-prompt.txt', 'r').read()
    response_wise_result = {}
    
    for index, row in tqdm(data.iterrows()):

        response_wise_result[index] = []
        
        main_response = row['main_response']
        stochastic_responses = []
        for stochastic_index in range(number_of_stochastic_responses):
            stochastic_responses.append(row['stochastic_response_'+str(stochastic_index+1)])
        try:
            main_response_sentences = sentence_slit(main_response)
            stochastic_responses_sentences = [sentence_slit(stochastic_response) for stochastic_response in stochastic_responses]
        except: 
            continue
        
        for main_response_sentence_index in range(len(main_response_sentences)):
            main_response_sentence = main_response_sentences[main_response_sentence_index]
            score = 0
            for stochastic_response_index in range(number_of_stochastic_responses):
                stochastic_response = stochastic_responses[stochastic_response_index]
                prompt = context.replace('<CONTEXT>', stochastic_response).replace('<SENTENCE>', main_response_sentence)

                while(True):
                    try:
                        output = generate_gpt_response(prompt, 4).content
                        print(output, stochastic_response_index, end = " ")
                        time.sleep(2)
                        break
                    except:
                        print("sleeping...")
                        time.sleep(5)
                        continue
                    
                score += score_mapper_prompt_response(output)
            response_wise_result[index].append(score/number_of_stochastic_responses)

    return response_wise_result



def nli_scorer(data):
    
    model_nli_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer_nli = AutoTokenizer.from_pretrained(model_nli_name)
    model_nli = AutoModelForSequenceClassification.from_pretrained(model_nli_name).to('cuda')


    
    response_wise_result = {}
    for index, row in tqdm(data.iterrows()):
        
        response_wise_result[index] = []
        main_response = row['main_response']
        stochastic_responses = []
        for stochastic_index in range(number_of_stochastic_responses):
            stochastic_responses.append(row['stochastic_response_'+str(stochastic_index+1)])
        try:
            main_response_sentences = sentence_slit(main_response)
            stochastic_responses_sentences = [sentence_slit(stochastic_response) for stochastic_response in stochastic_responses]
        except:
            continue
        
        
        try:
            for main_response_sentence_index in range(len(main_response_sentences)):
                main_response_sentence = main_response_sentences[main_response_sentence_index]
                premise = main_response_sentence
                score = 0
                for stochastic_response_index in range(number_of_stochastic_responses):
                    
                    stochastic_response = stochastic_responses[stochastic_response_index]
                    hypothesis = stochastic_response
                    
                    input = tokenizer_nli(premise, hypothesis, return_tensors="pt")
                    output = model_nli(input["input_ids"].to('cuda'))
                    label_names = ["entailment", "neutral", "contradiction"]
                    prediction = {name: float(pred) for pred, name in zip(output["logits"][0].cpu(), label_names)}
                    p_contradict = np.exp(prediction["contradiction"])/(np.exp(prediction["entailment"]) + np.exp(prediction["contradiction"]))
                    score += (p_contradict)
                    
                score = score / number_of_stochastic_responses
                response_wise_result[index].append(score)
        except:
            continue  
    return response_wise_result

def bert_scorer(data, bert_model):
    
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("sentence-transformers/{}".format(bert_model)).to('cuda') # run with 
    
    response_wise_result = {}
    for index, row in tqdm(data.iterrows()):
        response_wise_result[index] = []
        main_response = row['main_response']
        stochastic_responses = []
        for stochastic_index in range(number_of_stochastic_responses):
            stochastic_responses.append(row['stochastic_response_'+str(stochastic_index+1)])
        try:
            main_response_sentences = sentence_slit(main_response)
            stochastic_responses_sentences = [sentence_slit(stochastic_response) for stochastic_response in stochastic_responses]
        except: 
            continue
        
        cls_main_response_embeds = model.encode(main_response_sentences)
            
        stochastic_responses_embeds = []
        for stochastic_response_index in range(number_of_stochastic_responses):
            stochastic_response_sentences = stochastic_responses_sentences[stochastic_response_index]
            stochastic_responses_embeds.append(model.encode(stochastic_responses_sentences[stochastic_response_index]))
            
        
        score_main_response_sentence = [None for index in range(len(main_response_sentences))]
        for main_response_sentence_index in range(len(main_response_sentences)):
            storer = []
            for stochastic_response_index in range(len(stochastic_responses_sentences)):
                
                dot_scores = util.dot_score(stochastic_responses_embeds[stochastic_response_index], cls_main_response_embeds[main_response_sentence_index])
                current_max_score = torch.max(torch.flatten(dot_scores)).item()                
                storer.append(current_max_score)
            score_main_response_sentence[main_response_sentence_index]  = (1 - sum(storer)/len(storer))
        response_wise_result[index] = score_main_response_sentence
        
    return response_wise_result






number_of_stochastic_responses = 10
# for closed models, we used 10 stochastic responses instead of 20 due to limited monetary resources


if(__name__ == '__main__'):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory", help="path of directory where .csv files for closed models are present", required=True)
    parser.add_argument("-d", "--device_number", help='available gpu device', required=True)
    parser.add_argument("-o", "--output_directory", help='path of directory where results would be saved', required=True)
    parser.add_argument("-md", "--method_metric", help="choice of SelfCheckGPT metricl choose one from {'nli', 'prompt_scorer', 'bert_score'}", required=True)
    
    

    args = parser.parse_args()

    input_directory = args.input_directory
    device_number = args.device_number
    output_directory = args.output_directory
    method_metric = args.method_metric
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = device_number


    if not os.path.exists(output_directory + '/' + method_metric + '/'):
        os.makedirs(output_directory + '/' + method_metric + '/')

    for filename in find_filenames(input_directory):
        model_name = filename[:-4]
        print("Currently processing...", model_name)
        if(method_metric == 'prompt_scorer'):
            # output = prompt_scorer(pd.read_csv(input_directory + '/' + filename))
            # with open(output_directory + '/' + method_metric + '/' + model_name + '.pkl', 'wb') as fp:
            #     pickle.dump({"check": 1}, fp)
            print()
        if(method_metric == 'nli_scorer'):
            # output = nli_scorer(pd.read_csv(input_directory + '/' + filename))
            # with open(output_directory + '/' + method_metric + '/' + model_name + '.pkl', 'wb') as fp:
            #     pickle.dump({"check": 1}, fp)
            print()
        if(method_metric == 'bert_scorer'):
            models= ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
            for model in models:
                output = bert_scorer(pd.read_csv(input_directory + '/' + filename), model)
                with open(output_directory + '/' + method_metric + '/' + model_name + '.pkl', 'wb') as fp:
                    pickle.dump({"check": 1}, fp)
            print()
        