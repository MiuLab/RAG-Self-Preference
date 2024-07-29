python Llama_inference.py \
        ../data/reading_comprehension_data_llama_1000.json \
        ../data/reading_comprehension_data_openai_1000.json \
        results/0721_llama_vs_chatgpt/ \
        results/0721_llama_vs_human/ \
        2 2 \
        > results/0721_llama_results.txt
python ChatGPT_inference.py \
        ../data/reading_comprehension_data_llama_1000.json \
        ../data/reading_comprehension_data_openai_1000.json \
        results/0721_chatgpt_vs_llama/ \
        results/0721_chatgpt_vs_human/ \
        2 2 \
        > results/0721_chatgpt_results.txt