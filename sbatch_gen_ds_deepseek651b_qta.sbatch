#!/bin/bash
#SBATCH --job-name=deepseek651b-gen-questions
#SBATCH --cpus-per-task=256
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2000G
#SBATCH --output=sbatch_output/%x_-%j.txt  # %j (job_id) %x(job_name)             

# node the job ran on + empty line
echo "Job ran on:" $(hostname)
echo "" 

ml load Python/3.9.6-GCCcore-11.2.0

#pip install lm-format-enforcer
#pip install llama_index
#pip install llama-index-embeddings-huggingface
#pip install datasets --upgrade
#pip install llama-index-core llama-index-readers-docling llama-index-node-parser-docling llama-index-embeddings-huggingface llama-index-llms-huggingface-api llama-index-vector-stores-milvus llama-index-readers-file python-dotenv
#pip install backports.tarfile
#pip install docling --upgrade
#pip install llama-index-llms-huggingface
#pip install llama-index
#pip install bitsandbytes
#pip install --upgrade transformers autoawq accelerate
#pip install --no-build-isolation auto-gptq
#pip install optimum
#pip install urllib3
#pip install deepeval

export OLLAMA_MODELS=YOUR_HOME_DIR/ollama_install/models
export LD_LIBRARY_PATH=YOUR_HOME_DIR/ollama_install/usr/lib:$LD_LIBRARY_PATH
export PATH=YOUR_HOME_DIR/ollama_install/usr/bin:$PATH
export OLLAMA_HOST=172.24.120.177

#enable print statetemnts in logs
export PYTHONUNBUFFERED=1

cd YOUR_HOME_DIR/

ollama serve &
python gen_ds_deepseek_651b_qta.py

wait

echo ""
echo "Job finished" 

