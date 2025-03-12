python run.py --model-id "openai/gpt-4o" "What is the capital of China?"

python run.py --model-id "Qwen/QwQ-32B" --provider="hf" question="What is the capital of China?"

python run.py --model-id "Qwen/QwQ-32B" --provider="localhost" question="What is the capital of China?"

python run_gaia.py --run-name="gaia-level1-qwq-32b" --model-id "Qwen/QwQ-32B" --provider="localhost"
python run_gaia.py --run-name="gaia-level-1" --model-id "Qwen/Qwen2.5-72B-Instruct" --provider="localhost"