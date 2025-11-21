'''
    对已经训练好的Lora模型和原来对比
'''
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

import jieba
import re
from tqdm import tqdm
import swanlab


class ModelComparator:
    def __init__(self, base_model_path, finetuned_model_path, val_data_path):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.val_data_path = val_data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, use_fast=False, trust_remote_code=True
        )

        # 加载基础模型
        print("加载基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map="auto", torch_dtype=torch.bfloat16
        )

        # 加载微调后的模型
        print("加载微调后的模型...")
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.finetuned_model = PeftModel.from_pretrained(
            self.finetuned_model, model_id=finetuned_model_path
        )

        # 加载验证数据
        self.val_data = self._load_validation_data()

    def _load_validation_data(self):
        """加载验证集数据"""
        with open(self.val_data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data

    def _extract_think_and_answer(self, text):
        """从文本中提取思考过程和答案"""
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""

        # 移除思考部分获取答案
        answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        answer = re.sub(r'^\s*\n?', '', answer)  # 移除开头的空白和换行

        return think, answer

    def calculate_perplexity(self, model, texts):
        """计算困惑度"""
        perplexities = []
        model.eval()

        for text in tqdm(texts, desc="计算困惑度"):
            # 编码文本
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())

        return np.mean(perplexities)

    def calculate_bleu(self, references, candidates):
        """计算BLEU分数"""
        smoothie = SmoothingFunction().method4
        bleu_scores = []

        for ref, cand in zip(references, candidates):
            # 使用jieba进行中文分词
            ref_tokens = list(jieba.cut(ref))
            cand_tokens = list(jieba.cut(cand))

            # 计算BLEU分数
            bleu = sentence_bleu([ref_tokens], cand_tokens,
                                 smoothing_function=smoothie,
                                 weights=(0.25, 0.25, 0.25, 0.25))
            bleu_scores.append(bleu)

        return np.mean(bleu_scores)

    def calculate_rouge(self, references, candidates):
        """计算ROUGE分数"""
        rouge = Rouge()
        try:
            scores = rouge.get_scores(candidates, references, avg=True)
            return scores
        except:
            # 如果出现错误，返回默认值
            return {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-l': {'f': 0, 'p': 0, 'r': 0}}

    def calculate_f1_precision_recall(self, references, candidates):
        """基于词汇重叠计算F1、精确率和召回率"""
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for ref, cand in zip(references, candidates):
            # 分词
            ref_words = set(jieba.cut(ref))
            cand_words = set(jieba.cut(cand))

            # 计算交集
            common_words = ref_words & cand_words

            if len(cand_words) == 0:
                precision = 0
            else:
                precision = len(common_words) / len(cand_words)

            if len(ref_words) == 0:
                recall = 0
            else:
                recall = len(common_words) / len(ref_words)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return (np.mean(f1_scores), np.mean(precision_scores),
                np.mean(recall_scores))

    def generate_response(self, model, question):
        """生成模型回答"""
        PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def evaluate_models(self, sample_size=50):
        """全面评估两个模型"""
        print(f"开始评估，使用 {sample_size} 个样本...")

        # 随机选择样本
        if sample_size > len(self.val_data):
            sample_size = len(self.val_data)

        indices = np.random.choice(len(self.val_data), sample_size, replace=False)
        sample_data = [self.val_data[i] for i in indices]

        # 存储结果
        base_results = []
        finetuned_results = []
        references = []

        # 生成回答
        for i, item in enumerate(tqdm(sample_data, desc="生成模型回答")):
            question = item["question"]
            reference_think = item.get("think", "")
            reference_answer = item.get("answer", "")
            reference_full = f"<think>{reference_think}</think>\n{reference_answer}"

            references.append(reference_full)

            # 基础模型生成
            base_response = self.generate_response(self.base_model, question)
            base_results.append(base_response)

            # 微调模型生成
            finetuned_response = self.generate_response(self.finetuned_model, question)
            finetuned_results.append(finetuned_response)

            # 每10个样本清理一次GPU缓存
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 计算定量指标
        print("计算定量指标...")

        # 困惑度
        base_perplexity = self.calculate_perplexity(self.base_model, base_results)
        finetuned_perplexity = self.calculate_perplexity(self.finetuned_model, finetuned_results)

        # BLEU分数
        base_bleu = self.calculate_bleu(references, base_results)
        finetuned_bleu = self.calculate_bleu(references, finetuned_results)

        # ROUGE分数
        base_rouge = self.calculate_rouge(references, base_results)
        finetuned_rouge = self.calculate_rouge(references, finetuned_results)

        # F1、精确率、召回率
        base_f1, base_precision, base_recall = self.calculate_f1_precision_recall(references, base_results)
        finetuned_f1, finetuned_precision, finetuned_recall = self.calculate_f1_precision_recall(references,
                                                                                                 finetuned_results)

        # 准备定性对比结果
        qualitative_results = []
        for i in range(min(5, len(sample_data))):  # 选择前5个进行详细展示
            qualitative_results.append({
                "question": sample_data[i]["question"],
                "reference": references[i],
                "base_model": base_results[i],
                "finetuned_model": finetuned_results[i]
            })

        # 汇总结果
        results = {
            "quantitative": {
                "base_model": {
                    "perplexity": base_perplexity,
                    "bleu": base_bleu,
                    "rouge_1_f": base_rouge['rouge-1']['f'],
                    "rouge_1_p": base_rouge['rouge-1']['p'],
                    "rouge_1_r": base_rouge['rouge-1']['r'],
                    "rouge_2_f": base_rouge['rouge-2']['f'],
                    "rouge_2_p": base_rouge['rouge-2']['p'],
                    "rouge_2_r": base_rouge['rouge-2']['r'],
                    "rouge_l_f": base_rouge['rouge-l']['f'],
                    "rouge_l_p": base_rouge['rouge-l']['p'],
                    "rouge_l_r": base_rouge['rouge-l']['r'],
                    "f1": base_f1,
                    "precision": base_precision,
                    "recall": base_recall
                },
                "finetuned_model": {
                    "perplexity": finetuned_perplexity,
                    "bleu": finetuned_bleu,
                    "rouge_1_f": finetuned_rouge['rouge-1']['f'],
                    "rouge_1_p": finetuned_rouge['rouge-1']['p'],
                    "rouge_1_r": finetuned_rouge['rouge-1']['r'],
                    "rouge_2_f": finetuned_rouge['rouge-2']['f'],
                    "rouge_2_p": finetuned_rouge['rouge-2']['p'],
                    "rouge_2_r": finetuned_rouge['rouge-2']['r'],
                    "rouge_l_f": finetuned_rouge['rouge-l']['f'],
                    "rouge_l_p": finetuned_rouge['rouge-l']['p'],
                    "rouge_l_r": finetuned_rouge['rouge-l']['r'],
                    "f1": finetuned_f1,
                    "precision": finetuned_precision,
                    "recall": finetuned_recall
                }
            },
            "qualitative": qualitative_results
        }

        return results

    def print_comparison_results(self, results):
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("模型对比结果")
        print("=" * 80)

        base_metrics = results["quantitative"]["base_model"]
        finetuned_metrics = results["quantitative"]["finetuned_model"]

        # 打印定量指标对比
        print("\n定量指标对比:")
        print("-" * 60)
        print(f"{'指标':<20} {'基础模型':<15} {'微调模型':<15} {'提升':<10}")
        print("-" * 60)

        metrics_to_display = {
            "困惑度": ("perplexity", True),  # 越低越好
            "BLEU": ("bleu", False),  # 越高越好
            "ROUGE-1 F1": ("rouge_1_f", False),
            "ROUGE-2 F1": ("rouge_2_f", False),
            "ROUGE-L F1": ("rouge_l_f", False),
            "词汇F1": ("f1", False),
            "精确率": ("precision", False),
            "召回率": ("recall", False)
        }

        for metric_name, (metric_key, lower_is_better) in metrics_to_display.items():
            base_val = base_metrics[metric_key]
            finetuned_val = finetuned_metrics[metric_key]

            if lower_is_better:
                improvement = base_val - finetuned_val  # 负值表示变差
                improvement_str = f"{improvement:+.4f}"
            else:
                improvement = finetuned_val - base_val
                improvement_str = f"{improvement:+.4f}"

            print(f"{metric_name:<20} {base_val:<15.4f} {finetuned_val:<15.4f} {improvement_str:<10}")

        # 打印定性对比
        print("\n定性对比 (前5个样本):")
        print("-" * 80)

        for i, item in enumerate(results["qualitative"]):
            print(f"\n样本 {i + 1}:")
            print(f"问题: {item['question']}")
            print(f"参考回答: {item['reference'][:200]}...")
            print(f"基础模型: {item['base_model'][:200]}...")
            print(f"微调模型: {item['finetuned_model'][:200]}...")
            print("-" * 40)

    def log_to_swanlab(self, results):
        """将结果记录到SwanLab"""
        base_metrics = results["quantitative"]["base_model"]
        finetuned_metrics = results["quantitative"]["finetuned_model"]

        # 记录主要指标
        swanlab.log({
            "base_perplexity": base_metrics["perplexity"],
            "finetuned_perplexity": finetuned_metrics["perplexity"],
            "base_bleu": base_metrics["bleu"],
            "finetuned_bleu": finetuned_metrics["bleu"],
            "base_f1": base_metrics["f1"],
            "finetuned_f1": finetuned_metrics["f1"],
            "base_rouge_1_f": base_metrics["rouge_1_f"],
            "finetuned_rouge_1_f": finetuned_metrics["rouge_1_f"],
        })

        # 记录定性对比结果
        qualitative_texts = []
        for i, item in enumerate(results["qualitative"]):
            text = f"""
样本 {i + 1}:
问题: {item['question']}
参考回答: {item['reference']}
基础模型: {item['base_model']}
微调模型: {item['finetuned_model']}
{'=' * 50}
"""
            qualitative_texts.append(swanlab.Text(text))

        swanlab.log({"qualitative_comparison": qualitative_texts})


def main():
    """主函数"""
    # 配置参数
    BASE_MODEL_PATH = "./Qwen3-0.6B"
    FINETUNED_MODEL_PATH = "./output/Qwen3-0.6B/checkpoint-1082"
    VAL_DATA_PATH = "val.jsonl"
    SAMPLE_SIZE = 50  # 评估使用的样本数量


    # 初始化比较器
    comparator = ModelComparator(BASE_MODEL_PATH, FINETUNED_MODEL_PATH, VAL_DATA_PATH)

    # 初始化SwanLab（必须在log之前调用）
    swanlab.init(
        project="Model-Comparison",
        experiment_name="Qwen3-Medical-SFT",
        config={
            "base_model": BASE_MODEL_PATH,
            "finetuned_model": FINETUNED_MODEL_PATH,
            "sample_size": SAMPLE_SIZE,
            "val_data": VAL_DATA_PATH
        }
    )
    # 进行评估
    results = comparator.evaluate_models(sample_size=SAMPLE_SIZE)

    # 打印结果
    comparator.print_comparison_results(results)


    # 记录到SwanLab
    try:
        comparator.log_to_swanlab(results)
        print("\n结果已记录到SwanLab")
    except Exception as e:
        print(f"\nSwanLab记录失败: {e}")

    # 保存详细结果到文件
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: model_comparison_results.json")


if __name__ == "__main__":
    main()