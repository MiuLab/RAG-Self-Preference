python Llama_inference.py \
        data/marco-both-true.json \
        results/marco_knowledge_scale_of_5_llama.json \
        results/0913_marco_llama_vs_chatgpt/ \
        results/0913_marco_llama_vs_human/ \
        1000 1000 \
        > results/0913_marco_llama_chatgpt_results.txt
python ChatGPT_inference.py \
        data/marco-both-true.json \
        results/marco_knowledge_scale_of_5_openai.json \
        results/0913_marco_chatgpt_vs_llama/ \
        results/0913_marco_chatgpt_vs_human/ \
        1000 1000 \
        > results/0913_marco_chatgpt_llama_results.txt
python ChatGPT_inference.py \
        data/nq-both-true.json \
        results/nq_knowledge_scale_of_5_openai.json \
        results/0913_nq_chatgpt_vs_llama/ \
        results/0913_nq_chatgpt_vs_human/ \
        0 1000 \
        > results/0913_nq_chatgpt_llama_results.txt
python Llama_inference.py \
        data/nq-both-true.json \
        results/nq_knowledge_scale_of_5_llama.json \
        results/0913_nq_llama_vs_chatgpt/ \
        results/0913_nq_llama_vs_human/ \
        1000 1000 \
        > results/0913_nq_llama_chatgpt_results.txt