import os
import stat
from pathlib import Path

PARTITION = "ckpt-g2"

VARIANTS = [
    "baseline_cot",
    "democrat_cot",
    "republican_cot",
    "religious_cot",
    "atheist_cot",
    "high_school_cot",
    "college_cot", 
    "woman_cot",
    "man_cot",
    "baseline_accuracy",
    "democrat_accuracy",
    "republican_accuracy",
    "religious_accuracy",
    "atheist_accuracy",
    "high_school_accuracy",
    "college_accuracy",
    "woman_accuracy",
    "man_accuracy"

]

# Maps model name to where it should be evaluated
MODELS = {
    "llama3.1": "ollama",
    "llama2": "ollama",
    "mistral": "ollama",
    "wizardlm2": "ollama",
    "gpt-3.5-turbo-0125": "openai",
    "gpt-4-0613": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai"
}

slurm_def = lambda variant, model, evaluator, port: f"""#!/bin/bash

#SBATCH --account=clmbr
#SBATCH --job-name=run-ollama-{variant}-{model}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH -o /mmfs1//gscratch/clmbr/amelie/projects/motivated-reasoning/%x_%j.out

cd /gscratch/clmbr/amelie/projects/motivated-reasoning

conda init bash
source ~/.bashrc
conda activate misinfo

echo "Starting ollama on port {port}"
export no_proxy=127.0.0.1,localhost
export OLLAMA_MODELS=/gscratch/clmbr/amelie/.cache/ollama/models
OLLAMA_HOST=127.0.0.1:{port} ollama serve > ollama-{variant}-{model}.log 2>&1 &

echo "Sleeping for 10 seconds to let the ollama server start"
sleep 10

echo "Running scripts"
python run-ollama.py {variant} {model} {evaluator} http://127.0.0.1:{port}
"""

def chmodx(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

def main():
    os.makedirs("slurm", exist_ok=True)

    with open("slurm/submit_all.sh", "w") as sh:
        sh.write("#!/bin/bash\n")

        port = 11434

        for variant in VARIANTS:
            for (model, evaluator) in MODELS.items():
                path = Path(f"slurm/{variant}-{model}.slurm")
                
                print(f"Writing {path}")
                with open(path, "w") as f:
                    contents = slurm_def(variant, model, evaluator, port)
                    f.write(contents)
                
                sh.write(f"sbatch {path.absolute()}\n")
                port += 1

    chmodx("slurm/submit_all.sh")

if __name__ == "__main__":
    main()