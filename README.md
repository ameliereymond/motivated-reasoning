# motivated-reasoning

## Running locally

```bash
./start-ollama.sh
export no_proxy=127.0.0.1,localhost

python run-ollama.py baseline_cot llama2
```

## Running on slurm

```bash
python generate_slurm.py
./slurm/submit_all.sh
```