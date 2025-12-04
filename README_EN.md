# Qwen3 Fine-tuning Practice: Medical R1 Reasoning Style Chat

[中文](README.md)

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

- **Base Model**: [Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B/summary)
- **Fine-tuned Model**: [Qwen3-1.7b-Medical-R1-sft](https://modelscope.cn/models/testUser/Qwen3-1.7b-Medical-R1-sft/summary)
- **Dataset**: [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- **SwanLab**: [qwen3-sft-medical](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/runs/agps0dkifth5l1xytcdyk/chart)
- **Fine-tuning Methods**: Full Parameter Fine-tuning, LoRA Fine-tuning
- **Reasoning Style**: R1 Reasoning Style
- **Computing Requirements**:
  - **Full Parameter Fine-tuning**: 8GB GPU Memory
  - **LoRA Fine-tuning**: 6GB GPU Memory

> To reduce GPU memory requirements, the Qwen3-0.6B model is used.

## 📊 Model Evaluation Results

### Quantitative Evaluation Comparison

#### Full Parameter Fine-tuning vs Base Model
| Evaluation Metric | Base Model | Full Parameter Fine-tuned Model | Improvement |
|---------|---------|--------------|---------|
| **Perplexity** | 1.8568 | 1.8441 | **Reduced by 0.0127** |
| **BLEU Score** | 0.1154 | 0.1869 | **Improved by 61.87%** |
| **ROUGE-1 F1** | 0.0563 | 0.0952 | **Improved by 68.92%** |
| **Vocabulary F1** | 0.4487 | 0.5427 | **Improved by 20.94%** |
| **Precision** | 0.4921 | 0.5922 | **Improved by 20.33%** |
| **Recall** | 0.4174 | 0.5094 | **Improved by 22.04%** |

#### LoRA Fine-tuning vs Base Model
| Evaluation Metric | Base Model | LoRA Fine-tuned Model | Improvement |
|---------|---------|-------------|---------|
| **Perplexity** | 1.9060 | 2.0242 | Increased by 6.20% |
| **BLEU Score** | 0.1235 | 0.1539 | **Improved by 24.63%** |
| **ROUGE-1 F1** | 0.0538 | 0.0847 | **Improved by 57.43%** |
| **Vocabulary F1** | 0.4470 | 0.4902 | **Improved by 9.67%** |
| **Precision** | 0.4995 | 0.5940 | **Improved by 18.92%** |
| **Recall** | 0.4091 | 0.4314 | **Improved by 5.45%** |

> **Key Finding**: Full parameter fine-tuning outperforms LoRA fine-tuning in almost all evaluation metrics, with significant improvements in text generation quality (BLEU, ROUGE).

### Qualitative Evaluation Examples

#### Example 1: Role of Endoscopic Ultrasound in Gastric Cancer Diagnosis
**Question**: Doctor, I heard that endoscopic ultrasound is very helpful in diagnosing gastric cancer. What specific role does it play?

**Base Model Answer**:
- Contains thinking process: Thought about the basic functions of endoscopic ultrasound, but some descriptions are not accurate
- Answer structure: Lists 4 functions including enhancing diagnostic accuracy, localization and biopsy
- Accuracy: Some descriptions have confusion with medical concepts

**Full Parameter Fine-tuned Model Answer**:
- Thinking process: More professional medical reasoning, considers specific applications of endoscopic ultrasound in gastric cancer staging
- Answer focus: Accurately describes key roles of endoscopic ultrasound in judging tumor invasion depth, evaluating lymph node metastasis
- Professionalism: Uses correct medical terminology, explanations are more consistent with clinical practice

#### Example 2: Examination Methods for Bronchiectasis
**Question**: Doctor, I've been feeling difficulty breathing recently, sometimes coughing up yellow phlegm. I'd like to know what examination methods are available for bronchiectasis?

**Base Model Answer**:
- Misdiagnosed bronchiectasis as bronchial asthma
- Inaccurate listing of examination methods, includes irrelevant ECG examination
- Lacks detailed explanation of key diagnostic methods like sputum examination

**Full Parameter Fine-tuned Model Answer**:
- Correctly identified bronchiectasis and related symptoms
- Comprehensively listed correct methods including imaging examinations (chest X-ray, high-resolution CT), laboratory tests, microbiological examinations
- Detailed explanation of the clinical significance and diagnostic value of each examination

## Project Structure

```
qwenMedSft/
├── dataset/                    # Dataset directory
│   ├── train.jsonl
│   └── val.jsonl
├── output/                    # Trained model output
│   ├── Qwen3-0.6B/            # LoRA fine-tuned model
│   │   ├── checkpoint-400
│   │   ├── checkpoint-800
│   │   └── checkpoint-1082
│   ├── Qwen3-param/           # Full parameter fine-tuned model
│   │   ├── checkpoint-400
│   │   ├── checkpoint-800
│   │   └── checkpoint-1082
├── Qwen3-0.6B/               # Base model
├── results/                  # Experiment result files
│   ├── model_comparison_results.json          # LoRA fine-tuning comparison results
│   └── full_param_model_comparison_results.json  # Full parameter fine-tuning comparison results
├── src/                      # Source code
│   ├── data.py               # Data loading and splitting
│   ├── train_lora.py         # LoRA fine-tuning training
│   ├── train.py              # Full parameter fine-tuning training
│   ├── model_compare.py      # LoRA model comparison evaluation
│   ├── compare_param.py      # Full parameter model comparison evaluation
│   └── infer.py              # Three-model comparison inference
├── scripts/
│   └── run.sh                # Experiment running script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Environment Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Automatically completes dataset download, preprocessing, validation set splitting, generating `train.jsonl` and `val.jsonl` files:

```bash
python src/data.py
```

## Training

### Full Parameter Fine-tuning
```bash
python src/train.py
```

### LoRA Fine-tuning
```bash
python src/train_lora.py
```

### SwanLab Training Logs
[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

Comparison of two fine-tuning methods. Tests show full parameter fine-tuning performs better than LoRA:

![](./readme_images/charts.png)

## Model Evaluation

### LoRA Fine-tuned Model Evaluation
```bash
python src/model_compare.py
```
Results saved to: `results/model_comparison_results.json`

### Full Parameter Fine-tuned Model Evaluation
```bash
python src/compare_param.py
```
Results saved to: `results/full_param_model_comparison_results.json`

### Evaluation Metrics Explanation
1. **Perplexity**: Measures model prediction capability, lower is better
2. **BLEU Score**: Evaluates similarity between generated text and reference text
3. **ROUGE Score**: Measures recall-oriented text similarity
4. **F1/Precision/Recall**: Vocabulary overlap-based text matching evaluation

## Inference Testing

### Three-Model Comparison Inference
```bash
python src/infer.py
```

### LoRA Fine-tuned Model Single Inference
```bash
python src/test_lora.py
```

### Full Parameter Fine-tuned Model Single Inference
```bash
python src/test_param.py
```

## Training Effect Example

**Question**: Doctor, I've been having stomach discomfort recently. I heard there are several anti-ulcer drugs for treatment. Could you详细介绍这些药物的分类、作用机制以及它们是如何影响胃黏膜的保护与损伤平衡的？

**Model Answer (with thinking process)**:
```xml
<think>
用户问的是抗溃疡药物的分类、作用机制，以及它们如何影响胃黏膜的保护和损伤平衡。
首先，我需要回忆抗溃疡药物的主要分类。主要分为几大类：抑酸药、胃黏膜保护剂、促胃动力药和抗幽门螺杆菌药物。
抑酸药包括H2受体拮抗剂和质子泵抑制剂，作用机制是抑制胃酸分泌。胃黏膜保护剂如硫糖铝、铋剂，通过形成保护层或促进黏液分泌来保护黏膜。促胃动力药通过增强胃蠕动减少反流。抗幽门螺杆菌药物通过抑制细菌生长和减少胃酸分泌来治疗溃疡。
这些药物通过不同机制平衡胃黏膜的保护和损伤：抑酸药减少损伤因素，保护剂增强防御，促胃动力药减少反流损伤，抗幽门螺杆菌药物消除病原体。
</think>

当然可以。抗溃疡药物主要分为四类：抑酸药、胃黏膜保护剂、促胃动力药和抗幽门螺杆菌药物。
1. **抑酸药**：包括H₂受体拮抗剂和质子泵抑制剂，通过抑制胃酸分泌减少对胃黏膜的化学损伤。
2. **胃黏膜保护剂**：如硫糖铝和胶体果胶铋，在胃黏膜表面形成保护膜，隔离胃酸和胃蛋白酶的侵蚀。
3. **促胃动力药**：如多潘立酮，增强胃排空，减少胃内食物和胃酸的反流时间。
4. **抗幽门螺杆菌药物**：通常采用抗生素联合质子泵抑制剂，根除幽门螺杆菌感染，消除主要病因。
这些药物共同作用，通过减少攻击因素（胃酸、幽门螺杆菌）和增强防御因素（黏膜屏障、胃排空）来恢复胃黏膜的保护与损伤平衡。
```

## Technical Points

### R1 Reasoning Style
- Model generates `<think>` thinking process before answering
- Thinking process demonstrates model's reasoning logic and knowledge retrieval process
- Final answer is based on summary and refinement of the thinking process

### Fine-tuning Strategy Comparison
1. **Full Parameter Fine-tuning**:
   - Updates all model parameters
   - Higher training cost but better results
   - More suitable for professional domain knowledge learning

2. **LoRA Fine-tuning**:
   - Only trains low-rank adapters
   - High training efficiency, low memory requirements
   - Suitable for resource-constrained scenarios

### Evaluation Methods
- Quantitative evaluation using 50 random samples
- Qualitative analysis including 5 detailed samples
- Comprehensive multiple NLP evaluation metrics

## Related Tools

- [swanlab](https://github.com/SwanHubX/SwanLab): Open-source, modern-designed deep learning training tracking and visualization tool
- [transformers](https://github.com/huggingface/transformers): HuggingFace's library containing pre-trained models for text, computer vision, audio, video and multimodal tasks, for inference and training
- [peft](https://github.com/huggingface/peft): Library for efficient fine-tuning of large language models

## Conclusion

This project demonstrates the effectiveness of fine-tuning the Qwen3 model in the medical domain. By comparing full parameter fine-tuning and LoRA fine-tuning methods, we found:
1. Full parameter fine-tuning performs better in professional medical Q&A
2. R1 reasoning style effectively improves answer professionalism and explainability
3. Even with the limited 0.6B parameter scale, the model can still learn professional medical knowledge

Future work could explore fine-tuning larger parameter-scale models and training with more diverse medical datasets.