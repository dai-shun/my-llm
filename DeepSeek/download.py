from modelscope import snapshot_download
model_dir = snapshot_download('deepseek-ai/deepseek-llm-7b-chat', cache_dir='./', revision='master')