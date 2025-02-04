import pandas as pd
import numpy as np
import ollama
import re
import os
import sys

OLLAMA_HOST = "http://127.0.0.1:11434"

real_or_fake = [
    "Fake", "Fake", "Fake", "Fake", "Fake", "Fake", "Fake", "Fake", "Fake", "Fake",
    "Real", "Real", "Real", "Real", "Real", "Real", "Real", "Real", "Real", "Real"
]

headlines = [
    "Government Officials Have Manipulated Stock Prices to Hide Scandals",
    "The Corporate Media Is Controlled by the Military-industrial Complex: The Major Oil Companies Own the Media and Control Their Agenda",
    "New Study: Left-Wingers Are More Likely to Lie to Get a Higher Salary",
    "The Government Is Manipulating the Public's Perception of Genetic Engineering in Order to Make People More Accepting of Such Techniques",
    "Left-Wing Extremism Causes 'More Damage' to World Than Terrorism, Says UN Report",
    "Certain Vaccines Are Loaded with Dangerous Chemicals and Toxins",
    "New Study: Clear Relationship Between Eye Color and Intelligence",
    "The Government Is Knowingly Spreading Disease Through the Airwaves and Food Supply",
    "Ebola Virus 'Caused by US Nuclear Weapons Testing', New Study Says",
    "Government Officials Have Illegally Manipulated the Weather to Cause Devastating Storms",
    "Attitudes Toward EU Are Largely Positive, Both Within Europe and Outside It",
    "One-in-Three Worldwide Lack Confidence in NGOs",
    "Reflecting a Demographic Shift, 109 US Counties Have Become Majority Nonwhite Since 2000",
    "International Relations Experts and US Public Agree: America Is Less Respected Globally",
    "Hyatt Will Remove Small Bottles from Hotel Bathrooms by 2021",
    "Morocco’s King Appoints Committee Chief to Fight Poverty and Inequality",
    "Republicans Divided in Views of Trump’s Conduct, Democrats Are Broadly Critical",
    "Democrats More Supportive than Republicans of Federal Spending for Scientific Research",
    "Global Warming Age Gap: Younger Americans Most Worried",
    "US Support for Legal Marijuana Steady in Past Year"
]


mist = pd.DataFrame({
    "ground_truth": real_or_fake,
    "headlines": headlines
})

def prepare_pa_message(headline, instructions):
    message_pa = [
        {
            'role': 'system',
            'content': instructions,
        },
        {
            'role': 'user',
            'content': "News Headline: {}".format(headline),
        },
        {
            'role': 'user',
            'content': "To the best of your knowledge, how accurate is the claim in the above headline?  1 being 'not at all' and 6 being 'very'",
        }
    ]
    return message_pa

def prepare_conf_message(message_pa, system_message):
    message_conf = [
        {
            'role': 'assistant',
            'content': system_message,
        },
        {
            'role': 'user',
            'content': "How confident are you in your judgment?”, 1 being 'not at all' and 7 being 'very'"
        }

    ]
    return message_pa + message_conf

def run_simulation(model, instructions, mist, n_sim):
    client = ollama.Client(host=OLLAMA_HOST)
    client.pull(model)
    
    llm_answers_pa = []
    llm_answers_conf = []
    pid = []

    for n in range(n_sim):
        for i in range(mist.shape[0]):
            try:
                pa_message = prepare_pa_message(mist['headlines'].iloc[i], instructions)
                assistant_message_pa = client.chat(model=model, messages=pa_message)['message']['content']
                conf_message = prepare_conf_message(pa_message, assistant_message_pa)
                assistant_message_conf = client.chat(model=model, messages=conf_message)['message']['content']
                llm_answers_pa.append(assistant_message_pa)
                llm_answers_conf.append(assistant_message_conf)
                pid.append(n)

            except Exception as e:
                print(e)
                
        print("{}/{}".format(n,n_sim), end = '\r', flush = True)
                
    return llm_answers_pa, llm_answers_conf, pid

def run_model(model_name, instructions, csv_name):
    n_sim = 100
    model_pa, model_conf, model_pid = run_simulation(model_name, instructions, mist, n_sim)
    mist_replicated = pd.concat([mist] * n_sim, ignore_index = True)
    model_df = pd.DataFrame(zip(model_pa, model_conf, model_pid), columns = ['perceived_accuracy', 'confidence', 'pid'])
    model_df = pd.concat([mist_replicated, model_df], axis = 1)
    model_df.to_csv(csv_name)
    return model_df

if __name__ == "__main__":

    # COT instructions 
    INSTRUCTIONS_BASELINE_COT = "You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step."
    INSTRUCTIONS_PERSONA_COT_TEMPLATE = "Take the role of a {}. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step."
    INSTRUCTIONS_PERSONA_EDUCATION_COT_TEMPLATE = "Take the role of a person whose highest level of education is {}. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step."
    INSTRUCTIONS_PERSONA_GENDER_COT_TEMPLATE = "You are a {}. Your responses should closely mirror the knowledge and abilities of this persona. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step."


    # political 
    INSTRUCTIONS_PERSONA_DEMOCRAT_COT = INSTRUCTIONS_PERSONA_COT_TEMPLATE.format("Democrat")
    INSTRUCTIONS_PERSONA_REPUBLICAN_COT = INSTRUCTIONS_PERSONA_COT_TEMPLATE.format("Republican")

    # religious
    INSTRUCTIONS_PERSONA_RELIGIOUS_COT = INSTRUCTIONS_PERSONA_COT_TEMPLATE.format("a religious person")
    INSTRUCTIONS_PERSONA_ATHEIST_COT = INSTRUCTIONS_PERSONA_COT_TEMPLATE.format("an atheist")

    # education
    INSTRUCTIONS_PERSONA_HIGH_SCHOOL_COT = INSTRUCTIONS_PERSONA_EDUCATION_COT_TEMPLATE.format("high school diploma or GED")
    INSTRUCTIONS_PERSONA_COLLEGE_COT = INSTRUCTIONS_PERSONA_EDUCATION_COT_TEMPLATE.format("Bachelor's or above")

    # gender
    INSTRUCTIONS_PERSONA_WOMAN_COT_TEMPLATE = INSTRUCTIONS_PERSONA_GENDER_COT_TEMPLATE.format("man")
    INSTRUCTIONS_PERSONA_MAN_COT_TEMPLATE = INSTRUCTIONS_PERSONA_GENDER_COT_TEMPLATE.format("woman")


    # Accuracy instructions 
    INSTRUCTIONS_BASELINE_ACCURACY = "You are a person who has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification."
    INSTRUCTIONS_PERSONA_ACCURACY_TEMPLATE = "Take the role of a {} who has skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification."
    INSTRUCTIONS_PERSONA_EDUCATION_ACCURACY_TEMPLATE = "Take the role of a person whose highest level of education is {} and has a skeptical attitude and strives for accuracy. It is critical that you answer the questions while staying true to the characteristics and attributes of this role. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification."
    INSTRUCTIONS_PERSONA_GENDER_ACCURACY_TEMPLATE = "You are a {} who has a skeptical attitude and strives for accuracy. Your responses should closely mirror the knowledge and abilities of this persona. You will be given a news headline. Please answer the following questions about it. ONLY return the answer, and do NOT give any justification. Think step by step."

    # political 
    INSTRUCTIONS_PERSONA_DEMOCRAT_ACCURACY = INSTRUCTIONS_PERSONA_ACCURACY_TEMPLATE.format("Democrat")
    INSTRUCTIONS_PERSONA_REPUBLICAN_ACCURACY = INSTRUCTIONS_PERSONA_ACCURACY_TEMPLATE.format("Republican")

    # religious
    INSTRUCTIONS_PERSONA_RELIGIOUS_ACCURACY = INSTRUCTIONS_PERSONA_ACCURACY_TEMPLATE.format("a religious person")
    INSTRUCTIONS_PERSONA_ATHEIST_ACCURACY = INSTRUCTIONS_PERSONA_ACCURACY_TEMPLATE.format("an atheist")

    # education 
    INSTRUCTIONS_PERSONA_HIGH_SCHOOL_ACCURACY = INSTRUCTIONS_PERSONA_EDUCATION_ACCURACY_TEMPLATE.format("high school diploma or GED")
    INSTRUCTIONS_PERSONA_COLLEGE_ACCURACY = INSTRUCTIONS_PERSONA_EDUCATION_ACCURACY_TEMPLATE.format("Bachelor's or above")

    # gender
    INSTRUCTIONS_PERSONA_WOMAN_ACCURACY = INSTRUCTIONS_PERSONA_GENDER_ACCURACY_TEMPLATE.format("man")
    INSTRUCTIONS_PERSONA_MAN_ACCURACY = INSTRUCTIONS_PERSONA_GENDER_ACCURACY_TEMPLATE.format("woman")


    variant_to_instructions = {
        # Chain of thought 
        "baseline_cot": INSTRUCTIONS_BASELINE_COT,
        "democrat_cot": INSTRUCTIONS_PERSONA_DEMOCRAT_COT,
        "republican_cot": INSTRUCTIONS_PERSONA_REPUBLICAN_COT, 
        "religious_cot": INSTRUCTIONS_PERSONA_RELIGIOUS_COT, 
        "atheist_cot": INSTRUCTIONS_PERSONA_ATHEIST_COT, 
        "high_school_cot": INSTRUCTIONS_PERSONA_HIGH_SCHOOL_COT,
        "college_cot": INSTRUCTIONS_PERSONA_COLLEGE_COT,
        "woman_cot": INSTRUCTIONS_PERSONA_WOMAN_COT_TEMPLATE,
        "man_cot": INSTRUCTIONS_PERSONA_MAN_COT_TEMPLATE, 

        # Accuracy 
        "baseline_accuracy": INSTRUCTIONS_BASELINE_ACCURACY,
        "democrat_accuracy": INSTRUCTIONS_PERSONA_DEMOCRAT_ACCURACY, 
        "republican_accuracy": INSTRUCTIONS_PERSONA_REPUBLICAN_ACCURACY, 
        "religious_accuracy": INSTRUCTIONS_PERSONA_RELIGIOUS_ACCURACY,
        "atheist_accuracy": INSTRUCTIONS_PERSONA_ATHEIST_ACCURACY,
        "high_school_accuracy": INSTRUCTIONS_PERSONA_HIGH_SCHOOL_ACCURACY,
        "college_accuracy": INSTRUCTIONS_PERSONA_COLLEGE_ACCURACY,
        "woman_accuracy": INSTRUCTIONS_PERSONA_WOMAN_ACCURACY, 
        "man_accuracy": INSTRUCTIONS_PERSONA_MAN_ACCURACY


    }

    variant = sys.argv[1]
    model = sys.argv[2]

    if len(sys.argv) == 4:
        OLLAMA_HOST = sys.argv[3]
    
    print(f"RUNNING WITH VARIANT: {variant}, MODEL: {model}, OLLAMA HOST: {OLLAMA_HOST}")

    if variant not in variant_to_instructions:
        raise Exception(f"Variant '{variant}' not known")

    instructions = variant_to_instructions[variant]

    os.makedirs("data", exist_ok=True)

    run_model(model, instructions, f"data/{model}_{variant}_MIST.csv")