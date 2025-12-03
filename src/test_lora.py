'''
    已经训练好的全参数模型对某个问题的回答
    支持交互式输入问题
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# 添加SwanLab导入
import swanlab


def predict(messages, model, tokenizer):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 初始化SwanLab，明确指定文本类型的图表
swanlab.init(
    experiment_name="medical_qa_interactive",
    description="医学专家模型交互式问答",
    config={
        "model_path": "./output/Qwen3-param/checkpoint-1082",
        "system_instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    }
)

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("../Qwen3-0.6B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../Qwen3-0.6B", device_map="auto", torch_dtype=torch.bfloat16)

# 加载lora模型
model = PeftModel.from_pretrained(model, model_id="./output/Qwen3-0.6B/checkpoint-1082")


# 固定的系统指令
instruction = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

print("医学专家问答系统已启动！")
print("输入 '退出' 或 'quit' 来结束对话\n")

# 交互式对话循环
conversation_count = 0
while True:
    # 获取用户输入
    user_input = input("请输入您的问题: ").strip()

    # 检查退出条件
    if user_input.lower() in ['退出', 'quit', 'exit']:
        print("感谢使用，再见！")
        break

    if not user_input:
        print("问题不能为空，请重新输入。")
        continue

    # 构建消息
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{user_input}"}
    ]

    print("\n思考中...")

    # 获取模型回答
    response = predict(messages, model, tokenizer)

    # 输出回答
    print(f"\n医学专家回答: {response}\n")
    print("-" * 50)

    # 记录到SwanLab - 使用文本格式记录
    conversation_count += 1
    swanlab.log({
        "conversation_id": conversation_count,
        "user_question": swanlab.Text(user_input),
        "model_response": swanlab.Text(response)
    })

# 完成记录
swanlab.finish()