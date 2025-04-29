# LLM-Finetuning-PedCardio
The repository hosts the scripts and Jupyter notebooks used in my master thesis "Finetuning Large-Language-Models Based on a Pediatric Cardiology Guideline on Congenital Heart Diseases"

Generally, the code provided in this repo is intended to be used on the MMH HPC and has to be modfied to be able to run for your special use cases.

In total you require 4 tokens (a crude way to use them is to replace the following keys with your API tokens).

- "YOUR_HOME_DIR/": Absolute path where this repo is cloned on the HPC. probably looks something like "/hpc/project/ag-xxxxxx/username/"
- "HUGGINGFACE TOKEN HERE": Your Hugginggface API token for downloading gated models like Llama 3.1
- "YOUR OPENAI TOKEN": Your OpenAI token which will be used in evaluating the models. Be warned: Evaluation, especially the Faithfulness metric, is not cheap using the default gpt-4o model!
- "YOUR DEEPEVAL TOKEN": Your Deepeval/ConfidentAi dashboard token. Used during evaluation for online eval storage and nice visualization.

This repo will probably not be updated or curated in any way. Feel free to fork and make improvements on your own.

DISCLAIMER:
The code provided in this repo should only serve as an initial reference for future projects and should be modified and tested before deploying on the HPC or any other platform!
Thoroughly test the code you take from this repo as it can result in very high computational load especially for the DeepSeek 651B model!

## Python Files
Files starting with "gen_ds_deepseek_651b" are launching the dataset generation using DeepSeeks R1 651B model.
- gen_ds_deepseek_651b_qta_llamacpp.py: Uses partial GPU acceleration with 12 offloaded layers. WARNING: HPC admins might not like this as it has a high resource consumption
- gen_ds_deepseek_651b_qta.py: Infers DeepSeek R1 using the OLLAMA application.
- gen_ds_deepseek_651b_qta_llamacpp_cpu.py: Infers DeepSeek R1 using the llamacpp engine and only using CPU resources

generate_questions_70b_quant.py starts the dataset generation using the LLama 3.1 70B model. This was used to generate the basic dataset of my thesis.

## SBATCH Files

These files are intended to be run as a batched background job on the HPC.
Do so by typing "sbatch sbatch_x.sbatch" in the leine CLI to start the script.
The required resources are allocated automatically if available.

Typically the start the python file with a similar naming structure.
E.g. "sbatch_gen_ds_deepseek651b_qta_llamacpp_cpu.sbatch" starts the python file "gen_ds_deepseek_651b_qta_llamacpp_cpu.py"

sbatch_run_ollama.sbatch runs a OLLAMA instance on a separate node using a 40GB Nvidia A100 GPU.
Make sure you have the required python packages installed. The pip commands should be included in the sbatch files.

## TXT Files
These files contain the prompts used during dataset generation.
- prompt_gen_questions.txt contains the prompt for the basic dataset with the Llama 3.1 70B model
- prompt_with_examples_no_ref_context.txt contains the prompt for the advanced dataset using DeepSeek R1 651B

## Gradio-UI

The Gradio webui built in this thesis can be found here. It can be started either as a Jupyter notebook or a Python file.
Before running take a look at the python requirements as it contains the required packages to run.
You probably have to start another job where OLLAMA is running with enough resources to infer the models.
Set the hostname in the files accordingly!

## Jupyter Notebooks
This folder contains a collection of Jupyter Notebooks used for testing, converting and evaluation.

- deepeval_create_testcases_ollama.ipynb: Infers the finetuned models using OLLAMA on a given dataset to generate the actual output of these models used for human evaluation
- deepeval_create_testcases_ollama.ipynb: Infers a model (usedModel) through OLLAMA on a given dataset (usedDataset) to generate the actual_output of a model. In my thesis this was used in Chapter 6.2 (before RAFT augmentation)
- deepeval_eval_correctness.ipynb: Calculates the correctness metrics for a given model using the deepeval library
- deepeval_eval_faithfullness.ipynb: Calculates the faithfulness metrics for a given model using the deepeval library
- eval_generate_meditron_outs_manual.ipynb: manually infers the MEDITRON model to generate the actual outputs for evaluation as inference through OLLAMA was too unreliable
- evaluate_models_BERTScore.ipynb: Generates the BERTScore of the to be evaluated models using Deepeval
- prepare_raft_advancedDS.ipynb: Processes the initially generated advanced dataset towards the RAFT training recipe. Thesis Chapter 6.3
- prepare_raft_basicDS.ipynb: Processes the initially generated basic dataset towards the RAFT training recipe. Thesis Chapter 6.3
- rag_inference_exported.ipynb: testing script for inferring a trained LORA adapter merged with the base model with the RAG system
- rag_inference_lora.ipynb: testing script for inferring a trained LORA adapter which is not yet merged with the base model with the RAG system
- rag_ollama_inference_exported.ipynb: testing script which infers a merged finetuned model through OLLAMA using the RAG system

## LLama Factory

This folder contains the training configurations used to finetuned the models Llama 3.1 8b, 70b, MEDITRON 7b and Qwen 2.5 32b Q4 with DeepSeek Distillation.
Merge the content of this folder with the local clone of the LLamaFactory repo for the filepaths to match.

The sbatch files should automatically start the finetuning.
The suffix "deepseekDS" indicates the advanced dataset finetuning, while "basicDS" implies the basic dataset.

export_scripts and training_scripts contains the configs for the advanced dataset finetune while export_scripts_basic and training_scripts_basic contains the configs for the basic dataset finetune.