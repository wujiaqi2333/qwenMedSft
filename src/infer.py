'''
    测试对比三个模型对同一个问题的回答
    问题在代码里写死
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def predict_single(model_path, messages, is_lora=False, base_model_path=None):
    # 清理缓存
    torch.cuda.empty_cache()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    # 加载模型
    if is_lora and base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    # 预测函数
    def predict(messages, model, tokenizer):
        device = next(model.parameters()).device

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    response = predict(messages, model, tokenizer)

    # 清理模型释放内存
    del model
    torch.cuda.empty_cache()

    return response


# 测试数据
test_texts = {
    'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}

messages = [
    {"role": "system", "content": test_texts['instruction']},
    {"role": "user", "content": test_texts['input']}
]

# 分别测试不同模型
print("=== 原始模型 ===")
response1 = predict_single("../Qwen3-0.6B", messages)
print(response1)

print("\n=== LoRA微调模型 ===")
response2 = predict_single(
    "../output/Qwen3-0.6B/checkpoint-1082",
    messages,
    is_lora=True,
    base_model_path="./Qwen3-0.6B"
)
print(response2)

print("\n=== 全参数微调模型 ===")
response3 = predict_single("../output/Qwen3-param/checkpoint-1082", messages)
print(response3)