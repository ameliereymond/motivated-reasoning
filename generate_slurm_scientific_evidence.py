import os
import stat
from pathlib import Path

PARTITION = "ckpt-g2"

# Only include the variants that match our evidence evaluation script
VARIANTS = [
    "baseline_cot",
    "democrat_cot",
    "republican_cot",
    "baseline_accuracy",
    "democrat_accuracy",
    "republican_accuracy"
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

        # Modified to use evidence_evaluation.py instead of run-ollama.py
        return f"""#!/bin/bash

#SBATCH --account=clmbr
#SBATCH --job-name=evidence-eval-{variant}-{self.name}
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

# Only start Ollama server if we're using Ollama
{f'''echo "Starting ollama on port {port}"
export no_proxy=127.0.0.1,localhost
export OLLAMA_MODELS=/gscratch/clmbr/amelie/.cache/ollama/models
OLLAMA_HOST=127.0.0.1:{port} ollama serve > ollama-{variant}-{self.name}.log 2>&1 &

echo "Sleeping for 10 seconds to let the ollama server start"
sleep 10''' if self.evaluator == "ollama" else ""}

echo "Running evidence evaluation"
python scientific_evidence.py {variant} {self.name} {self.evaluator} {f'http://127.0.0.1:{port}' if self.evaluator == "ollama" else ""}
"""

MODELS = [
    Model(name="llama3.1",  evaluator="ollama", needs_gpu=True),
    Model(name="llama2",    evaluator="ollama", needs_gpu=True),
    Model(name="mistral",   evaluator="ollama", needs_gpu=True),
    Model(name="wizardlm2", evaluator="ollama", needs_gpu=True),
    #Model(name="gpt-3.5-turbo-0125", evaluator="openai", needs_gpu=False),
    #Model(name="gpt-4-0613",  evaluator="openai", needs_gpu=False),
    #Model(name="gpt-4o",      evaluator="openai", needs_gpu=False),
    #Model(name="gpt-4o-mini", evaluator="openai", needs_gpu=False)
]

def chmodx(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)

def main():
    os.makedirs("slurm", exist_ok=True)

    with open("slurm/submit_all_evidence.sh", "w") as sh:
        sh.write("#!/bin/bash\n")

        port = 11434

        for variant in VARIANTS:
            for model in MODELS:
                path = Path(f"slurm/evidence-{variant}-{model.name}.slurm")
                
                print(f"Writing {path}")
                with open(path, "w") as f:
                    contents = model.generate_slurm(variant, port)
                    f.write(contents)
                
                sh.write(f"sbatch {path.absolute()}\n")
                # Only increment port for Ollama models
                if model.evaluator == "ollama":
                    port += 1

    chmodx("slurm/submit_all_evidence.sh")

if __name__ == "__main__":
    main()