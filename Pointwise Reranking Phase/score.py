import argparse

parser = argparse.ArgumentParser(description="Using LLM to evaluate passages")

parser.add_argument('--LLM', type=str, required=True, help='Specify the LLM to use, e.g., ChatGPT')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--result_path', type=str, required=True, help='Path to save the results')
parser.add_argument('--setting', type=str, required=True, help='Type of the task, e.g., RAG')
parser.add_argument('--factuality', type=str, required=True, help='True passages or False passages')

args = parser.parse_args()

import openai
import sys
import time
import json
import statistics

with open(args.dataset_path, "r") as f:
  qa_pairs = json.load(f)
  
ASK_TIME = 5
cnt = 0
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def ask_gpt3(user_input: str):
    OPENAI_API_KEY = ""
    openai.api_key = OPENAI_API_KEY
    for _ in range(ASK_TIME):
      try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        global cnt
        time.sleep(0.3)
        cnt += 1
        print(completion.choices[0].message["content"])
        return completion.choices[0].message["content"]
      except Exception as e:
        print(e)
        continue
    return "fail to generate answer"

def ask_llama(message, sys_prompt="You are a helpful assistant."):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = ""

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

if args.setting == 'Normal':
    EVALUATION_PROMPT = "You are a single-number rater. Answer with a single number. You will be given a passage. On a scale of 1 to 5, please evaluate this passage. The higher the score, the better the passage. Consider writing style and factuality. A normal passage should have a score of 3. A passage with flaws should be rated 1. Excellent passages can be rated 5. Format: ###<number>***."
elif args.setting == 'RAG':
    EVALUATION_PROMPT = "You are a single-number rater. Answer with a single number. You will be given a question and a passage. On a scale of 1 to 5, please evaluate how suitable the passage is for answering the question. The higher the score, the more suitable the passage is for providing answers. A passage that adequately answers a question should be rated 3. A passage that is not at all suitable for answering questions should be rated 1. Passages that provide clear, accurate, and comprehensive answers can be rated 5. Your reply format: \"###<number>***\"."
PASSAGE_NUM = 1000

def get_score(passage: str) -> [float, str]:
    trials = 5
    for i in range(trials):
        if args.LLM == 'GPT':
            result = ask_gpt3(EVALUATION_PROMPT + '\n' + passage)
        elif args.LLM == 'Llama':
            result = ask_llama(passage, EVALUATION_PROMPT)
        try:
            score = float(result[result.find("###") + 3:result.find("***")])
            return [score, result]
        except:
            continue
    return [None, result]

results = {
    "scores_human" : [],
    "scores_openai" : [],
    "scores_llama" : []
}

TRIALS = 3

for i in range(PASSAGE_NUM):
    eprint(i)
    if i > 0:
        with open(args.result_path, 'r') as f:
            results = json.load(f)
    
    qa_pair = qa_pairs[i]
    
    if args.setting == 'RAG':
        question = qa_pair['question']
        qa_pair_prompt = "Question: " + question + "\nPassage: "
    elif args.setting == 'Normal':
        qa_pair_prompt = "Passage: "
    
    if args.factuality == 'true':
        human_passage = qa_pair_prompt + qa_pair['context']
        openai_passage = qa_pair_prompt + qa_pair['gpt_true']
        llama_passage = qa_pair_prompt + qa_pair['llama_true']
    elif args.factuality == 'false':
        if args.LLM == 'ChatGPT':
            false_ans = qa_pair['gpt_false_ans']
        elif args.LLM == 'Llama':
            false_ans = qa_pair['llama_false_ans']
        answer = qa_pair['answer']
        human_passage = qa_pair_prompt + qa_pair['context'].replace(answer, false_ans)
        openai_passage = qa_pair_prompt + qa_pair['gpt_true'].replace(answer, false_ans)
        llama_passage = qa_pair_prompt + qa_pair['llama_true'].replace(answer, false_ans)
        
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
    results["scores_human"].append(human_score)
    results["scores_openai"].append(openai_score)
    results["scores_llama"].append(llama_score)
    print(f"[{i}] {human_score[0]} {openai_score[0]} {llama_score[0]}")
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=4)

with open(args.result_path, 'r') as f:
    results = json.load(f)
    
new_res = {
    "scores_human" : [],
    "scores_openai" : [],
    "scores_llama" : []
}
for i in range(PASSAGE_NUM):
    if(results["scores_human"][i][0] != None):
        new_res["scores_human"].append(results["scores_human"][i][0])
    if(results["scores_openai"][i][0] != None):
        new_res["scores_openai"].append(results["scores_openai"][i][0])
    if(results["scores_llama"][i][0] != None):
        new_res["scores_llama"].append(results["scores_llama"][i][0])

print(new_res)
print("********* Begin Evaluation **********")
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