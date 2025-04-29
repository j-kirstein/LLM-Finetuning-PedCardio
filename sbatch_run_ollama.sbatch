#!/bin/bash
#SBATCH --job-name=ollama-server
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --output=sbatch_output/%x_-%j.txt  # %j (job_id) %x(job_name)       
#SBATCH --partition=leinegpu  # only for GPU jobs       
#SBATCH --gres=gpu:a100-40g:1      

# node the job ran on + empty line
echo "Job ran on:" $(hostname)
echo "" 

ml load Python/3.9.6-GCCcore-11.2.0
ml load make/4.4.1-GCCcore-13.3.0
ml load cuDNN/8.4.1.50-CUDA-11.7.0

export OLLAMA_MODELS=YOUR_HOME_DIR/ollama_install/models
export LD_LIBRARY_PATH=YOUR_HOME_DIR/ollama_install/usr/lib:$LD_LIBRARY_PATH
export PATH=YOUR_HOME_DIR/ollama_install/usr/bin:$PATH
export OLLAMA_HOST=$(hostname)

#enable print statetemnts in logs
export PYTHONUNBUFFERED=1

cd YOUR_HOME_DIR/

ollama serve

wait

echo ""
echo "Job finished" 

