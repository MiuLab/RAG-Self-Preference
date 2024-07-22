#!/usr/bin/env python
# coding: utf-8

import os
import sys
os.system("pip install openai==0.28") # ensure correct version of openai for ChatGPT inference
import openai
import re
import time
import json
import pprint # pretty print
from prettytable import PrettyTable as pt

OPENAI_API_KEY = "substitute openai api key here"
PREFERENCE_PROMPT = "Please answer the question based on the following two passages. The answer must be retrieved from one of the passages. Only 1 passage! Cannot refer to both.\n"
FORMAT_PROMPT = "Please answer with the following format: Answer: <short answer> Answer retrieved from which passage: <choose either 1 or 2>. No additional explanation needed. Don't write anything else.\n"
FAIL_PROMPT = "Fail to generate a passage that includes the answer string."

PASSAGES_LLAMA = sys.argv[1] # llama
PASSAGES_OPENAI = sys.argv[2] # openai
RESULTDIR_1 = sys.argv[3]
RESULTDIR_2 = sys.argv[4]
TIMES_4 = int(sys.argv[5]) # input 0 if don't want to inference chatgpt vs. llama
TIMES_5 = int(sys.argv[6]) # input 0 if don't want to inference chatgpt vs. human
KNOWLEDGE_CHECK = "data/nq_knowledge_scale_of_5_1000_openai.json"
ASK_TIME = 5

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ask_gpt3(user_input: str):
    openai.api_key = OPENAI_API_KEY
    for _ in range(ASK_TIME):
      try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        time.sleep(0.3)
        return completion.choices[0].message["content"]
      except Exception as e:
        eprint(e)
        continue
    return "fail to generate answer"

with open(PASSAGES_LLAMA, 'r') as f:
    llama_passages = json.load(f)
with open(PASSAGES_OPENAI, 'r') as f:
    openai_passages = json.load(f)
with open(KNOWLEDGE_CHECK, 'r') as f:
    knowledge = json.load(f)
    knowledge_levels = knowledge['knowledge_level']
    distribution = knowledge['distribution']

num_level = len(distribution)
mp = {True : "TRUE", False : "FALSE"}

def check_bag_of_words(truth: str, ai_answer: str, debug = False, record = []) -> bool:
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
  truth_set = set(re.split(r'\s|(?<!\d)[,.;](?!\d)', truth))
  answer_set = set(re.split(r'\s|(?<!\d)[,.;](?!\d)', ai_answer))
  if debug:
    print("ground_truth:", truth)
    print("AI_answer:", ai_answer)
    print("truth_set:", truth_set)
    print("answer_set:", answer_set)
    record.append({"ground_truth":truth, "AI_answer":ai_answer, "truth_set":truth_set, "answer_set":answer_set})
  return truth_set.issubset(answer_set) or answer_set.issubset(truth_set) or truth in ai_answer or ai_answer in truth

def reading_comprehension(
    question: str, true_answer: str, first_passage: str, second_passage: str,
    first_answer: str, second_answer: str, results_list: list, debug = False
  ) -> bool:
  """Asks ChatGPT to answer a question based on two passages and explain which passage(s) it referred to.

  Note that a new item would be appended into the results_list only if it is
  done with {trials} times.

  Args:
    question: The question to be answered.
    true_answer: Standard answer to the question in SQuAD dataset.
    first_passage and second_passage: The two passages for reading comprehension.
    first_answer: the answer string in the first passage.
    second_answer: the answer string in the second passage.
    debug: A bool indicating whether to print debug message.
    result: A list for storing ChatGPT's answers. This parameter should be given.

  Returns:
    correctness: Bool that indicates whether ChatGPT answers the question correctly. This is checked through the check_correctness() function.
    from_first: whether the ai answer is extracted from the first passage
    from_second: whether the ai answer is extracted from the second passage
    preference: Integer that indicates which passage ChatGPT refers to. Its value should be either 0 or 1.
    Returns [-1, -1] if ChatGPT returns answer with wrong format too many times.
  """
  trials = 5
  correctness, preference = bool(), int()
  ai_answer, passage_referred_to = str(), str()
  while trials:
    if first_passage == FAIL_PROMPT or second_passage == FAIL_PROMPT:
      break
    trials -= 1
    prompt = "Question: " + question + "\nPassage 1:\n" + first_passage + "\nPassage 2:\n" + second_passage
    sys_prompt = PREFERENCE_PROMPT + "\n" + FORMAT_PROMPT
    result = ask_gpt3(sys_prompt + ' ' + prompt)
    if result.find("Answer: ") == -1 or result.find("Answer retrieved from") == -1:
      continue # wrong format
    pos1 = result.find("Answer: ") + len("Answer: ")
    pos2 = result.find("Answer retrieved from") + len("Answer retrieved from")

    ai_answer = result[pos1:result.find("Answer retrieved from")]
    passage_referred_to = result[pos2:]
    correctness = check_bag_of_words(ai_answer, true_answer, False)
    if first_answer == second_answer:
      # check preference according to AI's claim
      one = passage_referred_to.find("1") != -1 # 1 in preference
      two = passage_referred_to.find("2") != -1 # 2 in preference
    else:
      # check preference according to answer string
      one = check_bag_of_words(ai_answer, first_answer, False)
      two = check_bag_of_words(ai_answer, second_answer, False)
    if int(one) + int(two) != 1:
        continue # wrong format. AI is forced to pick only one.
    if one:
      preference = 0
    else:
      preference = 1

    results_list.append({
      "question" : question,
      "true_answer" : true_answer,
      "passage_1" : first_passage,
      "passage_2" : second_passage,
      "ai_answer" : ai_answer,
      "correctness" : correctness,
      "preference" : preference # 0-based
    })
    return True
  results_list.append(None)
  return False # reading comprehension didn't complete with {trials} times

def print_result_summary(num_level: int, results: list, title: str,
                         self_preferred_only: bool):
  """Given the summary of reading comprehension, print out pretty table.

  Args:
    num_level: Number of knowledge levels.
    results: The summary to be printed out.
    title: The title of the table.
  """
  table = pt()
  table.title = title

  fields = ["Level \ ", "Correct", "Prefer 1", "Prefer 2", "Total"]
  table.field_names = fields

  for level in range(num_level):
    table.add_row([level] + results[level])

  if self_preferred_only:
    summary_row = ["Llama TRUE Total Preferred"]
  else:
    summary_row = ["Total"]

  for col in range(len(fields)-1):
    summary_row.append(sum([results[level][col] for level in range(num_level)]))
  table.add_row(summary_row)

  print(table)

def compare_preference(results_A, results_B):
  """Compares the preferences of AI after swapping the orders of two passages."""
  # Organize 2 x 2 grid
  pref = [[0, 0], [0, 0]]
  for result_a, result_b in zip(results_A, results_B):
    if result_a and result_b:
      pref[result_a['preference']][result_b['preference']] += 1

  # Print results
  print("llama(0) -> chatgpt(0) :", pref[0][0])
  print("llama(0) -> llama(1) :", pref[0][1])
  print("chatgpt(1) -> chatgpt(0) :", pref[1][0])
  print("chatgpt(1) -> llama(1) :", pref[1][1])

def summarize(results: list, EXPECTED_LENGTH: int) -> list:
  sum_results = []
  preference_results = []

  for _ in range(num_level):
    sum_results.append([0, 0, 0, 0])
    preference_results.append([0, 0, 0, 0])

  KNOWLEDGE_LEVEL = int()

  for i in range(EXPECTED_LENGTH):
    result = results[i]
    if not result:
      continue # result == None, this index is skipped
    KNOWLEDGE_LEVEL = knowledge_levels[i]
    if result['correctness']:
      sum_results[KNOWLEDGE_LEVEL][0] += 1
    sum_results[KNOWLEDGE_LEVEL][result['preference']+1] += 1
    sum_results[KNOWLEDGE_LEVEL][3] += 1
  return sum_results

def LLM_vs_LLM_pairwise_passage_reading_comprehension(passage_1_reference: dict, passage_1_correctness: bool, passage_2_reference: dict, passage_2_correctness: bool):
  results = []
  failed_index = []

  for i in range(TIMES_4):
    if i % 100 == 0:
      eprint(f"Completed: {i}/{TIMES_4}")
    if passage_1_correctness == True:  
      passage_1 = passage_1_reference[i]['ai_true']
      answer_1 = passage_1_reference[i]['true_answer']
    else:
      passage_1 = passage_1_reference[i]['ai_false']
      answer_1 = passage_1_reference[i]['false_answer']
    if passage_2_correctness == True:
      passage_2 = passage_2_reference[i]['ai_true']
      answer_2 = passage_2_reference[i]['true_answer']
    else:
      passage_2 = passage_2_reference[i]['ai_false']
      answer_2 = passage_2_reference[i]['false_answer']
    done = reading_comprehension(
        llama_passages[i]['question'],
        llama_passages[i]['true_answer'],
        passage_1,
        passage_2,
        answer_1,
        answer_2,
        results,
        True
      )

  return results

def LLM_vs_LLM_exp(EXPECTED_LENGTH: int, RESULT_DIR: str, num: str, passage_1_correctness: bool, passage_2_correctness: bool):
  # Order 1
  eprint(f"========= Start of Exp. {num}.a: ChatGPT {mp[passage_1_correctness]} vs. Llama {mp[passage_2_correctness]} =========")
  results_A = LLM_vs_LLM_pairwise_passage_reading_comprehension(openai_passages, passage_1_correctness, llama_passages, passage_2_correctness)
  result_file = os.path.join(RESULT_DIR, f"AI_preference_ChatGPT_{mp[passage_1_correctness]}_Llama_{mp[passage_2_correctness]}.json")
  with open(result_file, "w") as f:
    json.dump(results_A, f, indent=4)
  sum_results = summarize(results_A, EXPECTED_LENGTH)
  table_title = f"Exp. {num}.a Llama {mp[passage_1_correctness]} vs. ChatGPT {mp[passage_2_correctness]}"
  print_result_summary(num_level, sum_results, table_title, False)
  eprint(f"Exp. {num}.a completed")

  # Order 2
  eprint(f"========= Start of Exp. {num}.a: Llama {mp[passage_2_correctness]} vs. ChatGPT {mp[passage_1_correctness]} =========")
  results_B = LLM_vs_LLM_pairwise_passage_reading_comprehension(llama_passages, passage_2_correctness, openai_passages, passage_1_correctness)
  result_file = os.path.join(RESULT_DIR, f"AI_preference_Llama_{mp[passage_2_correctness]}_ChatGPT_{mp[passage_1_correctness]}.json")
  with open(result_file, "w") as f:
    json.dump(results_B, f, indent=4)
  sum_results = summarize(results_B, EXPECTED_LENGTH)
  table_title = f"Exp. {num}.b ChatGPT {mp[passage_2_correctness]} vs. Llama {mp[passage_1_correctness]}"
  print_result_summary(num_level, sum_results, table_title, False)
  eprint(f"Exp. {num}.b completed")

  # Comparing results from two orders
  compare_preference(results_A, results_B)

def LLM_vs_LLM(EXPECTED_LENGTH: int, RESULT_DIR: str):
  os.mkdir(RESULT_DIR)
  LLM_vs_LLM_exp(EXPECTED_LENGTH, RESULT_DIR, "4.1", True, True)
  LLM_vs_LLM_exp(EXPECTED_LENGTH, RESULT_DIR, "4.2", True, False) # ChatGPT (self) True, Llama False
  LLM_vs_LLM_exp(EXPECTED_LENGTH, RESULT_DIR, "4.3", False, True)
  LLM_vs_LLM_exp(EXPECTED_LENGTH, RESULT_DIR, "4.4", False, False)

def LLM_vs_Human_pairwise_passage_reading_comprehension(passage_1_reference: dict, passage_1_correctness: bool, passage_2_reference: dict, passage_2_correctness: bool):
  results = []
  failed_index = []

  for i in range(TIMES_5):
    if i % 100 == 0:
      eprint(f"Completed: {i}/{TIMES_5}")
    if passage_1_correctness == True:  
      passage_1 = passage_1_reference[i]['human_true']
      answer_1 = passage_1_reference[i]['true_answer']
    else:
      passage_1 = passage_1_reference[i]['human_false']
      answer_1 = passage_1_reference[i]['false_answer']
    if passage_2_correctness == True:
      passage_2 = passage_2_reference[i]['ai_true']
      answer_2 = passage_2_reference[i]['true_answer']
    else:
      passage_2 = passage_2_reference[i]['ai_false']
      answer_2 = passage_2_reference[i]['false_answer']
    done = reading_comprehension(
        llama_passages[i]['question'],
        llama_passages[i]['true_answer'],
        passage_1,
        passage_2,
        answer_1,
        answer_2,
        results,
        True
      )

  return results

def LLM_vs_Human_exp(EXPECTED_LENGTH: str, RESULT_DIR: str, num: str, passage_1_correctness: bool, passage_2_correctness: bool):
  # Use other LLM's answers for human passages

  # Order 1
  eprint(f"========= Start of Exp. {num}.a: Human {mp[passage_1_correctness]} vs. ChatGPT {mp[passage_2_correctness]} =========")
  results_A = LLM_vs_LLM_pairwise_passage_reading_comprehension(llama_passages, passage_1_correctness, openai_passages, passage_2_correctness)
  result_file = os.path.join(RESULT_DIR, f"AI_preference_Human_{mp[passage_1_correctness]}_ChatGPT_{mp[passage_2_correctness]}.json")
  with open(result_file, "w") as f:
    json.dump(results_A, f, indent=4)
  sum_results = summarize(results_A, EXPECTED_LENGTH)
  table_title = f"Exp. {num}.a Human {mp[passage_1_correctness]} vs. ChatGPT {mp[passage_2_correctness]}"
  print_result_summary(num_level, sum_results, table_title, False)
  eprint(f"Exp. {num}.a Completed")

  # Order 2
  eprint(f"========= Start of Exp. {num}.a: ChatGPT {mp[passage_2_correctness]} vs. Human {mp[passage_1_correctness]} =========")
  results_B = LLM_vs_LLM_pairwise_passage_reading_comprehension(openai_passages, passage_2_correctness, llama_passages, passage_1_correctness)
  result_file = os.path.join(RESULT_DIR, f"AI_preference_ChatGPT_{mp[passage_2_correctness]}_Human_{mp[passage_1_correctness]}.json")
  with open(result_file, "w") as f:
    json.dump(results_B, f, indent=4)
  sum_results = summarize(results_B, EXPECTED_LENGTH)
  table_title = f"Exp. {num}.b ChatGPT {mp[passage_2_correctness]} vs. Human {mp[passage_1_correctness]}"
  print_result_summary(num_level, sum_results, table_title, False)
  eprint(f"Exp. {num}.b Completed")
  compare_preference(results_A, results_B)

def LLM_vs_Human(EXPECTED_LENGTH: str, RESULT_DIR: str):
  os.mkdir(RESULT_DIR)
  LLM_vs_Human_exp(EXPECTED_LENGTH, RESULT_DIR, "5.1", True, True)
  LLM_vs_Human_exp(EXPECTED_LENGTH, RESULT_DIR, "5.2", False, True) # Human False, ChatGPT (self) True
  LLM_vs_Human_exp(EXPECTED_LENGTH, RESULT_DIR, "5.3", True, False) 
  LLM_vs_Human_exp(EXPECTED_LENGTH, RESULT_DIR, "5.4", False, False) 

if TIMES_4 > 0:
  LLM_vs_LLM(TIMES_4, RESULTDIR_1)

if TIMES_5 > 0:
  LLM_vs_Human(TIMES_5, RESULTDIR_2)

# TODO:
#   1. Add arg parser, optional args, and main function
#   2. Add docstrings for each function
#   3. README / sample script
#   4. Tables on paper