


# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname "$script_dir")

# Set default parameters
device=0
save_dir='example/results'
model_dir="pretrained_models/Spark-TTS-0.5B"
text="Experience the future of AI-generated speech. Setting a new standard for open-source text-to-speech synthesis."
prompt_text="Enjoy the smooth and natural voice. This program is proudly sponsored by AI-driven speech technology."
prompt_speech_path="example/prompt_audio.wav"

# Change directory to the root directory
cd "$root_dir" || exit

source sparktts/utils/parse_options.sh

# Run inference
python -m cli.inference \
    --text "${text}" \
    --device "${device}" \
    --save_dir "${save_dir}" \
    --model_dir "${model_dir}" \
    --prompt_text "${prompt_text}" \
    --prompt_speech_path "${prompt_speech_path}"
    
    