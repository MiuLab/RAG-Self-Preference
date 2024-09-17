python Llama_inference.py \
        data/marco-writing-style.json \
        results/marco_knowledge_scale_of_5_llama.json \
        results/0914_marco_llama_normal_qa/ \
        results/useless/ \
        100 0 \
        > results/1_0914_marco_llama_normal_qa_results.txt
python ChatGPT_inference.py \
        data/marco-writing-style.json \
        results/marco_knowledge_scale_of_5_openai.json \
        results/0914_marco_chatgpt_normal_qa/ \
        results/useless/ \
        100 0 \
        > results/2_0914_marco_chatgpt_normal_qa_results.txt
python Llama_inference.py \
        data/nq-writing-style.json \
        results/nq_knowledge_scale_of_5_llama.json \
        results/0914_nq_llama_normal_qa/ \
        results/useless/ \
        100 0 \
        > results/3_0914_nq_llama_normal_qa_results.txt
python ChatGPT_inference.py \
        data/nq-writing-style.json \
        results/nq_knowledge_scale_of_5_openai.json \
        results/0913_nq_chatgpt_normal_qa/ \
        results/useless/ \
        100 0 \
        > results/4_0914_nq_chatgpt_normal_qa_results.txt

python Llama_inference.py \
        data/marco-writing-style.json \
        results/marco_knowledge_scale_of_5_llama.json \
        results/0914_marco_llama_normal_long/ \
        results/useless/ \
        100 0 \
        > results/5_0914_marco_llama_normal_long_results.txt
python ChatGPT_inference.py \
        data/marco-writing-style.json \
        results/marco_knowledge_scale_of_5_openai.json \
        results/0914_marco_chatgpt_normal_long/ \
        results/useless/ \
        100 0 \
        > results/6_0914_marco_chatgpt_normal_long_results.txt
python Llama_inference.py \
        data/nq-writing-style.json \
        results/nq_knowledge_scale_of_5_llama.json \
        results/0914_nq_llama_normal_long/ \
        results/useless/ \
        100 0 \
        > results/7_0914_nq_llama_normal_long_results.txt
python ChatGPT_inference.py \
        data/nq-writing-style.json \
        results/nq_knowledge_scale_of_5_openai.json \
        results/0914_nq_chatgpt_normal_long/ \
        results/useless/ \
        100 0 \
        > results/8_0914_nq_chatgpt_normal_long_results.txt