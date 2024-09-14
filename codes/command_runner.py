import subprocess
model_list = ['Meta-Llama-3-70B-Instruct', 'Meta-Llama-3-70B', 'Llama-2-70b-hf', 'Llama-2-70b-chat-hf', 'Mixtral-8x7B-Instruct-v0.1']
for model in model_list:
        result = subprocess.run('python3 open.py --input_file /home/prasoon/snap/main/mtp/llm-science-miscommunication/data/data.csv --model_name {} --required_quantization True --device_number 2 --output_directory /home/prasoon/snap/main/mtp/llm-science-miscommunication/results/open'.format(model), shell = True)