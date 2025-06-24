# Persona-Assigned LLMs Exhibit Human-Like Motivated Reasoning 

This repository contains the code, data, and evaluation scripts for our XXX 2025 paper: 

** Persona-Assigned LLMs Exhibit Human-Like Motivated Reasoning ** 

Bibtex:

```
todo
```

Please reach out to attr[AT]uw.edu and sadash [AT]uw.edu  with any questions. 

## Running the code

### Prerequisites

Make sure you have Python 3.13 installed. Then create and activate a virtual environment, and install the dependencies with `pip`.

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install pandas numpy ollama openai python-dotenv
```

### Running

The project can run experiments on locally served models, using [Ollama](https://github.com/ollama/ollama), or on OpenAI models, by calling their HTTP API directly. Depending on the type of model you want to run your experiment on, follow the following instructions.

#### Running on OpenAI models

Create a file named `.env` at the root of the project, and paste your [OpenAI API key](https://platform.openai.com/api-keys).

```bash
OPENAI_KEY=sk-pasteYourKeyHere
```

Then, run the experiment script with the following parameters:

```bash
python run-ollama.py [variant] [model_name] openai
```

Where:

- `[variant]` is the persona prompt variant. For a full list of implemented prompt variants, see [prompt variants](#prompt-variants)
- `[model_name]` is the [OpenAI model name](https://platform.openai.com/docs/models), e.g.`gpt-3.5-turbo-0125`, `gpt-4-0613`, `gpt-4o`, `gpt-4o-mini`, etc.

#### Running locally with Ollama 

First, start Ollama in the background:


```bash
./start-ollama.sh 
export no_proxy=127.0.0.1,localhost
```

Then, run the experiment:

```bash
python run-ollama.py baseline_cot llama2 ollama
```

Where:

- `[variant]` is the persona prompt variant. For a full list of implemented prompt variants, see [prompt variants](#prompt-variants)
- `[model_name]` is the [Ollama model name](https://ollama.com/library?sort=newest), e.g.`llama3.1`, `llama2`, `mistral`, `wizardlm2`, etc.

### Prompt variants

| Prompt variant         | Mitigation type  | Persona                        |
| -----------------------|------------------|--------------------------------|
| `baseline`             | No mitigation    | None                           |
| `democrat`             | No mitigation    | Democrat                       |
| `republican`           | No mitigation    | Republican                     |
| `religious`            | No mitigation    | Religious person               |
| `atheist`              | No mitigation    | Atheist                        |
| `high_school`          | No mitigation    | Highest education: high-school |
| `college`              | No mitigation    | Highest education: college     |
| `woman`                | No mitigation    | Woman                          |
| `man`                  | No mitigation    | Man                            |
| `baseline_cot`         | Chain-of-thought | None                           |
| `democrat_cot`         | Chain-of-thought | Democrat                       |
| `republican_cot`       | Chain-of-thought | Republican                     |
| `religious_cot`        | Chain-of-thought | Religious person               |
| `atheist_cot`          | Chain-of-thought | Atheist                        |
| `high_school_cot`      | Chain-of-thought | Highest education: high-school |
| `college_cot`          | Chain-of-thought | Highest education: college     |
| `woman_cot`            | Chain-of-thought | Woman                          |
| `man_cot`              | Chain-of-thought | Man                            |
| `baseline_accuracy`    | Accuracy         | None                           |
| `democrat_accuracy`    | Accuracy         | Democrat                       |
| `republican_accuracy`  | Accuracy         | Republican                     |
| `religious_accuracy`   | Accuracy         | Religious person               |
| `atheist_accuracy`     | Accuracy         | Atheist                        |
| `high_school_accuracy` | Accuracy         | Highest education: high-school |
| `college_accuracy`     | Accuracy         | Highest education: college     |
| `woman_accuracy`       | Accuracy         | Woman                          |
| `man_accuracy`         | Accuracy         | Man                            |


## Experiment design 
Instruction templates are defined at the bottom of run-ollama.py. For example: 
INSTRUCTIONS_PERSONA_ACCURACY_TEMPLATE = "Take the role of a {} who has a skeptical attitude and strives for accuracy..."


## Repository structure 