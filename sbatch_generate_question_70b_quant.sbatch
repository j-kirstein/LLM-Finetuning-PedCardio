#!/bin/bash
#SBATCH --job-name=llama70b-quant
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00          
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --output=sbatch_output/%x_-%j.txt  # %j (job_id) %x(job_name)
#SBATCH --partition=leinegpu  # only for GPU jobs       
#SBATCH --gres=gpu:a100-40g:1              # only for GPU jobs                

# node the job ran on + empty line
echo "Job ran on:" $(hostname)
echo "" 

ml load CUDA/11.7.0
ml load Python/3.9.6-GCCcore-11.2.0

pip install lm-format-enforcer
pip install llama_index
pip install llama-index-embeddings-huggingface
pip install datasets --upgrade
pip install llama-index-core llama-index-readers-docling llama-index-node-parser-docling llama-index-embeddings-huggingface llama-index-llms-huggingface-api llama-index-vector-stores-milvus llama-index-readers-file python-dotenv
pip install backports.tarfile
pip install docling --upgrade
pip install llama-index-llms-huggingface
pip install llama-index
pip install bitsandbytes
pip install --upgrade transformers autoawq accelerate
pip install --no-build-isolation auto-gptq
pip install optimum
pip install urllib3
pip install deepeval
pip install autoawq

cd YOUR_HOME_DIR/

python generate_questions_70b_quant.py

wait

echo ""
echo "Job finished" 

