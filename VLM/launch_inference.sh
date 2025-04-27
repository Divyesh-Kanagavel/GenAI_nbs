#!/bin/zsh

MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
PROMPT="the robot in this picture is "
IMAGE_FILE_PATH="/Users/divyeshkanagavel/Desktop/repos/GenAI_nbs/VLM/53628d_0d6b37296abd4e49838b942eeec49d85~mv2.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.75
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --do_sample "$DO_SAMPLE" \
    --only_cpu "$ONLY_CPU"