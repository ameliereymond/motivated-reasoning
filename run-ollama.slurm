#!/bin/bash

#SBATCH --account=clmbr
#SBATCH --job-name=run-ollama
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH -o /mmfs1//gscratch/clmbr/amelie/projects/motivated-reasoning/run_%x_%j.out

cd /gscratch/clmbr/amelie/projects/motivated-reasoning

conda init bash
source ~/.bashrc
conda activate misinfo

./start-ollama.sh

python run_ollama.py baseline_cot
python run_ollama.py democrat_cot
python run_ollama.py republican_cot
python run_ollama.py democrat_accuracy
python run_ollama.py republican_accuracy
