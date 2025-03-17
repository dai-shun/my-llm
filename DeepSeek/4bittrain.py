from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
# 将JSON文件转换为CSV文件
df = pd.read_json('../../data/huanhuan.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained('../../model/deepseek-ai/deepseek-llm-7b-chat/', use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Assistant: {example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id

tokenizer.decode(tokenized_id[0]['input_ids'])

tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

import torch

model = AutoModelForCausalLM.from_pretrained(
        '/root/model/deepseek-ai/deepseek-llm-7b-chat/', 
        trust_remote_code=True, 
        torch_dtype=torch.half, 
        device_map="auto",
        low_cpu_mem_usage=True,   # 是否使用低CPU内存
        load_in_4bit=True,  # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
        bnb_4bit_compute_dtype=torch.half,  # 4位精度计算的数据类型。这里设置为torch.half，表示使用半精度浮点数。
        bnb_4bit_quant_type="nf4", # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。
        bnb_4bit_use_double_quant=True  # 是否使用双精度量化。如果设置为True，则使用双精度量化。
    )
model.generation_config = GenerationConfig.from_pretrained('/root/model/deepseek-ai/deepseek-llm-7b-chat/')
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
model.dtype

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config

model = get_peft_model(model, config)
config

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./output/DeepSeek",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit"  # 优化器类型
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

text = "小姐，别的秀女都在求中选，唯有咱们小姐想被撂牌子，菩萨一定记得真真儿的——"
inputs = tokenizer(f"User: {text}\n\n", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
