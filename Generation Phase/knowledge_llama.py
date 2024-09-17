import os
os.system("pip install openai==1.9.0") # ensure correct version of openai for ChatGPT inference

import openai
from openai import OpenAI
import re
import time
import json

TIMES = 1000
OPENAI_API_BASE = "substitute llama openai api base here"
ASK_TIME = 10

def ask_llama(message, sys_prompt="You are a helpful assistant."):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = OPENAI_API_BASE

    for _ in range(ASK_TIME):
      try:
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
      except Exception as e:
          print(e)
          continue
    return "fail to generate answer"

KNOWLEDGE_CHECK = "results/nq_knowledge_scale_of_5_llama.json"

import time
def has_knowledge(truth: str, ai_answer: str, debug = False, record = []) -> bool:
  """Uses bag of words and substring comparison to check if LLM has knowledge about a certain question.
  Note that this is a symmetric function.

  Args:
    truth: ground truth provided by SQuAD Dataest
    ai_answer: LLM's answer to the given question
    debug: whether to print debug message
    record: whether to record LLM's answer and ground truth (shallow copy)

  Example:
    has_knowledge("A, B, and C", "C, B, D, A, and p")
    truth_set: {'', 'B', 'C', 'A', 'and'}
    answer_set: {'', 'D', 'B', 'C', 'A', 'p', 'and'}
    => TRUE
  """
  truth_set = set(re.split("\s|(?<!\d)[,.](?!\d)", truth))
  answer_set = set(re.split("\s|(?<!\d)[,.](?!\d)", ai_answer))
  if debug:
    print("ground_truth:", truth)
    print("AI_answer:", ai_answer)
    print("truth_set:", list(truth_set))
    print("answer_set:", list(answer_set))
  record.append({"ground_truth":truth, "AI_answer":ai_answer, "truth_set":list(truth_set), "answer_set":list(answer_set)})
  return truth_set.issubset(answer_set) or answer_set.issubset(truth_set) or truth in ai_answer or ai_answer in truth


with open('data/nq-both-true.json', 'r') as f:
  qa_pairs = json.load(f)

ai_answers = []
knowledge_level = []

for i in range(TIMES):
  cnt = 0
  for trials in range(5):
    qa_pair = qa_pairs[i]
    ai_answer = ask_llama(qa_pair['question'] + " Short answer with less than 5 words:")
    # time.sleep(0.1)
    print(ai_answer)
    cnt += has_knowledge(qa_pair['answer'].lower(), ai_answer.lower(), False, ai_answers)
  knowledge_level.append(cnt)
  print(f'[{i}] knowledge_level: {cnt}')

print(len(knowledge_level))
distribution = [0] * 6
for i in knowledge_level:
  distribution[i] += 1
print(distribution)

results = {"distribution":distribution, "knowledge_level":knowledge_level, "ai_answers":ai_answers}
with open(KNOWLEDGE_CHECK, 'w') as fd:
  json.dump(results, fd, indent=4)