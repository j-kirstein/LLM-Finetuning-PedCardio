#!/bin/bash
#SBATCH --job-name=sft-llama8b-deepseek
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --output=sbatch_output/%x_-%j.txt  # %j (job_id) %x(job_name)
#SBATCH --partition=leinegpu  # only for GPU jobs       
#SBATCH --gres=gpu:a100-80g:1              # only for GPU jobs     

# node the job ran on + empty line
echo "Job ran on:" $(hostname)
echo "" 

ml load Python/3.9.6-GCCcore-11.2.0
ml load cuDNN/8.4.1.50-CUDA-11.7.0

export HUGGINGFACEHUB_API_TOKEN=HUGGINGFACE TOKEN HERE
export HF_TOKEN=HUGGINGFACE TOKEN HERE
export HF_HOME=YOUR_HOME_DIR/.cache/huggingface/
export TRANSFOMERS_CACHE=YOUR_HOME_DIR/.cache/huggingface/


cd YOUR_HOME_DIR/LLaMA-Factory

pip install -e ".[metrics]"

llamafactory-cli train training_scripts/deepseek_noCOT_lora_llama-8b.yaml &
llamafactory-cli train training_scripts/deepseek_COT_lora_llama-8b.yaml


