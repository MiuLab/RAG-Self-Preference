python Llama_inference.py \
        data/reading_comprehension_data_llama_1000.json \
        data/reading_comprehension_data_openai_1000.json \
        0 0 \
        results/score_llama_normal.txt
python ChatGPT_inference.py \
        data/reading_comprehension_data_llama_1000.json \
        data/reading_comprehension_data_openai_1000.json \
        0 0 \
        results/score_chatgpt_normal.txt