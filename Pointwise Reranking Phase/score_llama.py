import os
import openai
import sys
from openai import OpenAI
import re
import time
import json
import pprint 
import statistics

"""
Load Dataset
"""
DATASET = "/data/nq-dev-dataset.json"
PASSAGES_LLAMA = sys.argv[1]
PASSAGES_OPENAI = sys.argv[2]
MODE = int(sys.argv[3]) #0 is normal, 1 is genbyqa, 2 is long
TYPE = int(sys.argv[4]) #0 is TRUE, 1 is FALSE
SCORES_RESULTS = sys.argv[5]
with open(DATASET, "r") as f:
    qa_pairs = json.load(f)
with open(PASSAGES_LLAMA, 'r') as f:
    llama_passages = json.load(f)
with open(PASSAGES_OPENAI, 'r') as f:
    openai_passages = json.load(f)

EVALUATION_PROMPT = "You are a single-number rater. Answer with a single number and please don't give me any explanation. You will be given a question and a passage. On a scale of 1 to 5, please evaluate how suitable the passage is for answering the question. The higher the score, the more suitable the passage is for providing answers. A passage that adequately answers a question should be rated 3. A passage that is not at all suitable for answering questions should be rated 1. Passages that provide clear, accurate, and comprehensive answers can be rated 5. Your reply format: \"###<number>***\"."
PASSAGE_NUM = 1000
ASK_TIME = 5
cnt = 0
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ask_llama(message, sys_prompt="You are a helpful assistant."):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "" # Your API base here

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="TheBloke/Llama-2-70B-chat-GPTQ",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": message},
        ]
    )
    return chat_response.choices[0].message.content.strip()

def get_score(passage: str) -> [float, str]:
    trials = 5
    for i in range(trials):
        result = ask_llama(passage, EVALUATION_PROMPT)
        try:
            score = float(result[result.find("###") + 3:result.find("***")])
            return [score, result]
        except:
            continue
    return [None, result]

if MODE == 0:
    results = {
        "scores_human" : [],
        "scores_openai" : [],
        "scores_llama" : []
    }
else:
    results = {
        "scores_openai" : [],
        "scores_llama" : []
    }

TRIALS = 3

for i in range(PASSAGE_NUM):
    eprint(i)
    if i > 0:
        with open(SCORES_RESULTS, 'r') as f:
            results = json.load(f)
    
    qa_pair = qa_pairs[i]
    question = qa_pair["question"]
    qa_pair_prompt = "Question: " + question + "\nPassage: "
    
    if TYPE == 0:
        human_passage = qa_pair_prompt + openai_passages[i]['human_true']
        openai_passage = qa_pair_prompt + openai_passages[i]['ai_true']
        llama_passage = qa_pair_prompt + llama_passages[i]['ai_true']
    else:
        human_passage = qa_pair_prompt + openai_passages[i]['human_false']
        openai_passage = qa_pair_prompt + openai_passages[i]['ai_false']
        llama_passage = qa_pair_prompt + llama_passages[i]['ai_false']
        
    human_score, openai_score, llama_score = get_score(human_passage), get_score(openai_passage), get_score(llama_passage)
    
    if human_score[0] == None:
        for j in range(TRIALS):
            print("repeat")
            human_score = get_score(human_passage)
            if human_score[0] != None:
                break
    if openai_score[0] == None:
        for j in range(TRIALS):
            openai_score = get_score(openai_passage)
            if openai_score[0] != None:
                break
    if llama_score[0] == None:
        for j in range(TRIALS):
            llama_score = get_score(llama_passage)
            if llama_score[0] != None:
                break
            
    if MODE == 0:
        results["scores_human"].append(human_score)
    results["scores_openai"].append(openai_score)
    results["scores_llama"].append(llama_score)
    
    print(f"[{i}] {human_score[0]} {openai_score[0]} {llama_score[0]}")
    
    with open(SCORES_RESULTS, 'w') as f:
        json.dump(results, f, indent=4)

# print(results)
with open(SCORES_RESULTS, 'r') as f:
    results = json.load(f)
new_res = {
    "scores_human" : [],
    "scores_openai" : [],
    "scores_llama" : []
}
for i in range(PASSAGE_NUM):
    if MODE == 0:
        if(results["scores_human"][i][0] != None):
            new_res["scores_human"].append(results["scores_human"][i][0])
    if(results["scores_openai"][i][0] != None):
        new_res["scores_openai"].append(results["scores_openai"][i][0])
    if(results["scores_llama"][i][0] != None):
        new_res["scores_llama"].append(results["scores_llama"][i][0])

print(new_res)
print("********* Begin Evaluation **********")
if MODE == 0:
    print("Human:", sum(new_res["scores_human"]) / len(new_res["scores_human"]), len(new_res["scores_human"]))
    print(statistics.stdev(new_res["scores_human"]))
    table = [0, 0, 0, 0, 0, 0]
    for i in new_res["scores_human"]:
        table[int(i)] += 1
    print(table)
print("OpenAI:", sum(new_res["scores_openai"]) / len(new_res["scores_openai"]), len(new_res["scores_openai"]))
print(statistics.stdev(new_res["scores_openai"]))
table = [0, 0, 0, 0, 0, 0]
for i in new_res["scores_openai"]:
    table[int(i)] += 1
print(table)
print("Llama:", sum(new_res["scores_llama"]) / len(new_res["scores_llama"]), len(new_res["scores_llama"]))
print(statistics.stdev(new_res["scores_llama"]))
table = [0, 0, 0, 0, 0, 0]
for i in new_res["scores_llama"]:
    table[int(i)] += 1
print(table)