# Can LLMs Replace Neil DeGrasse Tyson: Evaluating Reliability of LLMs as Science Communicators
This is the official repository for our <a href = 'https://openreview.net/forum?id=Eqpnq1sC43'> Can LLMs Replace Neil DeGrasse Tyson?: Evaluating the Reliability of LLMs as Science Communicators </a> (Prasoon Bajpai, Subhabrata Dutta, Niladri Chatterjee and Tanmoy Chakraborty, 2024)<br>
In this work, we conduct a large-scale probing of 14 open-source language models (Llama-2, Llama-3, Mistral) and 3 GPT models (text-davinci-003, GPT-3.5-Turbo, GPT-4-Turbo) using our new dataset **SCiPS-QA**, a collection of 742 boolean scientific problems grounded on complex scientific objects. The objective is to stress test and unveil the limitations of popular language models in communicating (and verifying) extremely complex scientific ideas.
## Contact and Citations
Please contact the first author regarding any queries about the paper or code. If you find our code or paper useful, please cite the paper:
```
@article{bajpai2024llm_science_miscommunication ,
  title={Can LLMs Replace Neil DeGrasse Tyson: Evaluating Reliability of LLMs as Science Communicators},
  author={Prasoon Bajpai, Subhabrata Dutta, Niladri Chatterjee, Tanmoy Chakraborty},
  journal={arXiv preprint},
  year={2024}
}
```

## Content

1. [Installation](#installation)
2. [SCiPS-QA](#scipsqa)
3. [Benchmark](#benchmark)

### Installation

#### Via `requirements.txt` (using `pip`)
To install the required dependencies using pip, you can run the following command:

```bash
pip install -r requirements.txt
```

#### Via `environment.yml` (using `conda`)
To create a conda environment with the required dependencies, use the following command:

```bash
conda env create -f environment.yml
```

### SCiPS-QA
We construct a boolean QA dataset with questions from subjects: Physics, Chemistry, Mathematics, Astronomy, Theoretical Computer Science, Biology and Economics. We augment this collection of 742 questions with golden reasoning, explaining the answer in the 'gold_reason' field. For every explanation requiring citation of study or some additional reference material, we provide the concerned hyperlinks to resources in the 'gold_reason_reference' field. Moreover, for open questions, we provide the latest timestamp (in MM/YYYY) the authors cross-checked the questions to be open in the 'time_open' field. An example entry **SCiPS-QA** is provided below:
```
{question: Do the homotopy groups of O(∞) contain a period-2 phenomenon with respect to dimension? , answer: A, time_open: NA, gold_reason: The groups $$ \pi_0 $$ and $$ \pi_1 $$ are both isomorphic to $$ \mathbb{Z}_2 $$, demonstrating a period-2 phenomenon in these dimensions., gold_reason_reference: NA}
```
### Benchmark

You can reproduce the benchmark results by running _open.py_ file in the 'codes/' directory as follows:
```
python3 open.py --input_file <directory_path>/data.csv --model_name Llama-2-7b-hf --required_quantization False --device_number 2 --output_directory <directory_path>

```
* _model_name_: We loaded our download models present in a directory '/home/models' and saved with name _model_name_
* _required_quantization_: We ran Llama-2-7b-hf on <a href = 'https://huggingface.co/blog/4bit-transformers-bitsandbytes'>4-bit</a> quantization. The exact _quantization_config_ is present in _open.py_
* _device_number_: PCI_BUS_ORDER of the CUDA device we ran our processes on. (We ran all our experiments on a single A100 GPU).
