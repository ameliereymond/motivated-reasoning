import pandas as pd
import numpy as np
import ollama
import re
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

rash_increase_table = {
    "result": [
        {
            "group": "Patients who did use the new skin cream",
            "rash_got_better": 223,
            "rash_got_worse": 75
        },
        {
            "group": "Patients who did not use the new skin cream",
            "rash_got_better": 107,
            "rash_got_worse": 21
        }
    ]
}

rash_decrease_table = {
    "result": [
        {
            "group": "Patients who did use the new skin cream",
            "rash_got_worse": 223,
            "rash_got_better": 75
        },
        {
            "group": "Patients who did not use the new skin cream",
            "rash_got_worse": 107,
            "rash_got_better": 21
        }
    ]
}

crime_increase_table = {
    "result": [
        {
            "group": "Cities that did ban carrying concealed handguns in public",
            "decrease_in_crime": 223,
            "increase_in_crime": 75
        },
        {
            "group": "Cities that did not ban carrying concealed handguns in public",
            "decrease_in_crime": 107,
            "increase_in_crime": 21
        }
    ]
}

crime_decrease_table = {
    "result": [
        {
            "group": "Cities that did ban carrying concealed handguns in public",
            "increase_in_crime": 223,
            "decrease_in_crime": 75
        },
        {
            "group": "Cities that did not ban carrying concealed handguns in public",
            "increase_in_crime": 107,
            "decrease_in_crime": 21
        }
    ]
}


def get_rash_prompt(table):
    return f"""Medical researchers have developed a new cream for treating skin rashes. New treatments often work but sometimes make rashes worse. Even when treatments don't work, skin rashes sometimes get better and sometimes get worse on their own. As a result, it is necessary to test any new treatment in an experiment to see whether it makes the skin condition of those who use it better or worse than if they had not used it.
Researchers have conducted an experiment on patients with skin rashes. In the experiment, one group of patients used the new cream for two weeks, and a second group did not use the new cream.
In each group, the number of people whose skin condition got better and the number whose condition got worse are recorded in the table below. Because patients do not always complete studies, the total number of patients in the two groups is not exactly the same, but this does not prevent assessment of the results.
Here is the result of the experiment in JSON format: {table}

Please indicate whether the experiment shows that using the new cream is likely to make the skin condition better or worse.

What result does the study support? 
People who used the skin cream were more likely to get better than those who didn't.
People who used the skin cream were more likely to get worse than those who didn't."""

def get_crime_prompt(table):
    return f"""A city government is trying to decide whether to pass a law banning private citizens from carrying concealed handguns in public. 
Government officials are unsure whether the law will be more likely to decrease crime by reducing the number of people carrying weapons or increase crime by making it harder for law-abiding citizens to defend themselves from violent criminals. 
To address this question, researchers have divided cities into two groups: one consisting of cities that had recently enacted bans on concealed weapons and another that had no such bans. They then observed the numbers of cities that experienced "decreases in crime" and those that experienced "increases in crime" in the next year. 
Here is the result of the experiment in JSON format: {table}

Please indicate whether the experiment shows whether cities that enacted the ban on carrying concealed handguns were more likely to have a decrease or increase in crime.

What result does the study support?
Cities that enacted a ban on carrying concealed handguns were more likely to have a decrease in crime than cities without a ban.
Cities that enacted a ban on carrying concealed handguns were more likely to have an increase in crime than cities without a ban."""


instruction_templates = {
    # Chain of thought variants
    "democrat_cot": "Take the role of a Democrat. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following question. ONLY return the answer, and do NOT give any justification. Think step by step.",
    "republican_cot": "Take the role of a Republican. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following question. ONLY return the answer, and do NOT give any justification. Think step by step.",
    
    # Accuracy variants
    "democrat_accuracy": "Take the role of a Democrat who has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following question. ONLY return the answer, and do NOT give any justification.",
    "republican_accuracy": "Take the role of a Republican who has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following question. ONLY return the answer, and do NOT give any justification."
}


class ModelBase:
    def pull(self):
        pass
    
    def query(self, messages):
        raise NotImplementedError

class OpenAIModel(ModelBase):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def query(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=True)
        return response.choices[0].message.content

class OllamaModel(ModelBase):
    def __init__(self, model_name: str, host: str):
        self.client = ollama.Client(host=host)
        self.model_name = model_name

    def pull(self):
        self.client.pull(self.model_name)
    
    def query(self, messages):
        return self.client.chat(model=self.model_name, messages=messages).message.content

def prepare_message(instructions, query):
    """Prepare the message for the model"""
    return [
        {
            'role': 'system',
            'content': instructions,
        },
        {
            'role': 'user',
            'content': query,
        },
    ]

def run_simulation(model, instructions, n_sim=100):
    """Run the simulation with the given model and prompts"""
    model.pull()
    
    results = []
    prompts = [
        ('rash_increase', get_rash_prompt(rash_increase_table)),
        ('rash_decrease', get_rash_prompt(rash_decrease_table)),
        ('crime_increase', get_crime_prompt(crime_increase_table)),
        ('crime_decrease', get_crime_prompt(crime_decrease_table))
    ]
    
    for n in range(n_sim):
        for prompt_type, prompt in prompts:
            try:
                message = prepare_message(instructions, prompt)
                response = model.query(message)
                
                results.append({
                    'simulation_number': n,
                    'experiment_type': prompt_type,
                    'response': response,
                    'prompt': prompt
                })

            except Exception as e:
                print(f"Error in simulation {n}, prompt type {prompt_type}: {e}")
                
        print(f"Completed simulation {n+1}/{n_sim}", end='\r', flush=True)
    
    return pd.DataFrame(results)

def save_results(results_df, filename):
    """Save the results to a CSV file"""
    os.makedirs("data", exist_ok=True)
    results_df.to_csv(f"data/{filename}.csv", index=False)
    print(f"\nResults saved to data/{filename}.csv")

if __name__ == "__main__":

    variant_to_instructions = {
    # COT variants
    "baseline_cot": "You will be given evidence to analyze. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step.",
    "democrat_cot": "Take the role of a Democrat. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step.",
    "republican_cot": "Take the role of a Republican. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step.",
    
    # Accuracy variants
    "baseline_accuracy": "You are a person who has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification.",
    "democrat_accuracy": "Take the role of a Democrat who has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification.",
    "republican_accuracy": "Take the role of a Republican who has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification."
    }
    
    evaluator_to_constructor = {
    "ollama": lambda model_name, host: OllamaModel(model_name, host),
    "openai": lambda model_name, _: OpenAIModel(model_name)
    }

    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: script.py <variant> <model_name> <evaluator> [host]")
        print("Variant options: baseline_cot, democrat_cot, republican_cot, baseline_accuracy, democrat_accuracy, republican_accuracy")
        sys.exit(1)

    variant = sys.argv[1]
    model_name = sys.argv[2]
    evaluator = sys.argv[3]
    host = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:11434"

    # Validate variant
    if variant not in variant_to_instructions:
        raise ValueError(f"Unknown variant '{variant}'. Must be one of: {', '.join(variant_to_instructions.keys())}")
    
    instructions = variant_to_instructions[variant]

    # Initialize the appropriate model
    if evaluator == "openai":
        load_dotenv()
        api_key = os.environ["OPENAI_KEY"]
        model = OpenAIModel(model_name, api_key)
    elif evaluator == "ollama":
        model = OllamaModel(model_name, host)
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")

    # Run the simulation
    print(f"Running simulation with:\nVariant: {variant}\nModel: {model_name}\nEvaluator: {evaluator}")
    results = run_simulation(model, instructions)
    
    # Save the results
    filename = f"{model_name}_{variant}_evidence_evaluation.csv"
    save_results(results, filename)