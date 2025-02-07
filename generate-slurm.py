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

class Model:
    def __init__(self, name: str, evaluator: str, needs_gpu: bool):
        self.name = name
        self.evaluator = evaluator
        self.needs_gpu = needs_gpu

    def generate_slurm(self, variant, port):
        if self.needs_gpu:
            gpu_header = "#SBATCH --gpus-per-node=1"
        else:
            gpu_header = ""

        return f"""#!/bin/bash

#SBATCH --account=clmbr
#SBATCH --job-name=run-ollama-{variant}-{self.name}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
{gpu_header}
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
OLLAMA_HOST=127.0.0.1:{port} ollama serve > ollama-{variant}-{self.name}.log 2>&1 &

echo "Sleeping for 10 seconds to let the ollama server start"
sleep 10

echo "Running scripts"
python run-ollama.py {variant} {self.name} {self.evaluator} http://127.0.0.1:{port}
"""


# Maps model name to where it should be evaluated
MODELS = [
    Model(name="llama3.1",  evaluator="ollama", needs_gpu=True),
    Model(name="llama3.1",  evaluator="ollama", needs_gpu=True),
    Model(name="llama2",    evaluator="ollama", needs_gpu=True),
    Model(name="mistral",   evaluator="ollama", needs_gpu=True),
    Model(name="wizardlm2", evaluator="ollama", needs_gpu=True),
    Model(name="gpt-3.5-turbo-0125", evaluator="openai", needs_gpu=False),
    Model(name="gpt-4-0613",  evaluator="openai", needs_gpu=False),
    Model(name="gpt-4o",      evaluator="openai", needs_gpu=False),
    Model(name="gpt-4o-mini", evaluator="openai", needs_gpu=False)
]

def chmodx(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

def main():
    os.makedirs("slurm", exist_ok=True)

    with open("slurm/submit_all.sh", "w") as sh:
        sh.write("#!/bin/bash\n")

        port = 11434

        for variant in VARIANTS:
            for model in MODELS:
                path = Path(f"slurm/{variant}-{model.name}.slurm")
                
                print(f"Writing {path}")
                with open(path, "w") as f:
                    contents = model.generate_slurm(variant, port)
                    f.write(contents)
                
                sh.write(f"sbatch {path.absolute()}\n")
                port += 1

    chmodx("slurm/submit_all.sh")

if __name__ == "__main__":
    main()