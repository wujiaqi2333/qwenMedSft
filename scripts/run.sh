#!/bin/bash

# ============================================
# Qwen3 医疗大模型微调实验 - 自动化运行脚本
# 功能：全流程自动化执行数据准备、训练、评估与推理
# 作者：AI助手
# 日期：2025-12-04
# ============================================

# 设置全局变量和路径
BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)  # 项目根目录
SRC_DIR="$BASE_DIR/src"
DATASET_DIR="$BASE_DIR/dataset"
OUTPUT_DIR="$BASE_DIR/output"
RESULTS_DIR="$BASE_DIR/results"
SCRIPTS_DIR="$BASE_DIR/scripts"
BASE_MODEL_DIR="$BASE_DIR/Qwen3-0.6B"

# 训练参数配置（可根据需要修改）
# 注意：全参数微调需要更多显存，LoRA微调更节省资源[citation:1]
LORA_CHECKPOINT="checkpoint-1082"      # LoRA模型检查点
FULL_PARAM_CHECKPOINT="checkpoint-1082" # 全参数模型检查点
SAMPLE_SIZE=50                          # 评估样本数量[citation:6]
TRAIN_EPOCHS=3                          # 训练轮数

# 颜色定义，用于终端输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 检查命令执行状态
check_status() {
    if [ $? -eq 0 ]; then
        log_success "$1"
    else
        log_error "$2"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo -e "${GREEN}Qwen3医疗大模型微调实验运行脚本${NC}"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --all              执行完整流程（数据准备->训练->评估->推理）"
    echo "  --data-only        仅准备数据"
    echo "  --train-only       仅进行训练（全参数和LoRA）"
    echo "  --train-lora       仅进行LoRA微调训练"
    echo "  --train-full       仅进行全参数微调训练"
    echo "  --eval-only        仅进行评估（需要已训练好的模型）"
    echo "  --infer-only       仅进行推理对比"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --all            # 执行完整实验流程"
    echo "  $0 --train-only     # 仅训练模型"
    echo "  $0 --eval-only      # 仅评估已有模型"
    echo ""
    echo "注意：全参数微调需要约8GB显存，LoRA微调需要约6GB显存[citation:6]"
}

# 环境检查函数
check_environment() {
    log_info "开始环境检查..."

    # 检查Python环境
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装，请先安装Python3"
        exit 1
    fi

    # 检查项目目录结构
    local required_dirs=("$SRC_DIR" "$DATASET_DIR" "$OUTPUT_DIR" "$RESULTS_DIR")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_warning "目录不存在: $dir，正在创建..."
            mkdir -p "$dir"
        fi
    done

    # 检查基础模型是否存在
    if [ ! -d "$BASE_MODEL_DIR" ]; then
        log_error "基础模型目录不存在: $BASE_MODEL_DIR"
        log_error "请先将Qwen3-0.6B模型下载到该目录"
        exit 1
    fi

    # 检查Python依赖
    if [ -f "$BASE_DIR/requirements.txt" ]; then
        log_info "检查Python依赖包..."
        pip install -r "$BASE_DIR/requirements.txt" > /dev/null 2>&1
        check_status "Python依赖检查完成" "Python依赖安装失败"
    else
        log_warning "未找到requirements.txt文件，跳过依赖检查"
    fi

    log_success "环境检查完成"
}

# 数据准备函数
prepare_data() {
    log_info "开始数据准备..."

    # 检查数据脚本是否存在
    if [ ! -f "$SRC_DIR/data.py" ]; then
        log_error "数据准备脚本不存在: $SRC_DIR/data.py"
        exit 1
    fi

    # 执行数据准备脚本
    cd "$BASE_DIR" || exit 1
    python3 "$SRC_DIR/data.py"
    check_status "数据准备完成" "数据准备失败"

    # 检查生成的数据文件
    if [ -f "$DATASET_DIR/train.jsonl" ] && [ -f "$DATASET_DIR/val.jsonl" ]; then
        local train_count=$(wc -l < "$DATASET_DIR/train.jsonl")
        local val_count=$(wc -l < "$DATASET_DIR/val.jsonl")
        log_success "数据集分割完成：训练集 $train_count 条，验证集 $val_count 条"
    else
        log_error "数据文件生成失败"
        exit 1
    fi
}

# LoRA微调训练函数
train_lora() {
    log_info "开始LoRA微调训练..."

    # 检查训练脚本是否存在
    if [ ! -f "$SRC_DIR/train_lora.py" ]; then
        log_error "LoRA训练脚本不存在: $SRC_DIR/train_lora.py"
        exit 1
    fi

    # 创建输出目录
    local lora_output_dir="$OUTPUT_DIR/Qwen3-0.6B"
    mkdir -p "$lora_output_dir"

    # 执行LoRA训练
    cd "$BASE_DIR" || exit 1
    log_info "运行命令: python3 $SRC_DIR/train_lora.py"
    python3 "$SRC_DIR/train_lora.py"
    check_status "LoRA微调训练完成" "LoRA微调训练失败"

    # 检查训练结果
    if [ -d "$lora_output_dir" ]; then
        local checkpoint_count=$(find "$lora_output_dir" -name "checkpoint-*" -type d | wc -l)
        log_success "LoRA训练完成，生成 $checkpoint_count 个检查点"
    fi
}

# 全参数微调训练函数
train_full_param() {
    log_info "开始全参数微调训练..."

    # 检查训练脚本是否存在
    if [ ! -f "$SRC_DIR/train.py" ]; then
        log_error "全参数训练脚本不存在: $SRC_DIR/train.py"
        exit 1
    fi

    # 创建输出目录
    local full_param_output_dir="$OUTPUT_DIR/Qwen3-param"
    mkdir -p "$full_param_output_dir"

    # 执行全参数训练
    cd "$BASE_DIR" || exit 1
    log_info "运行命令: python3 $SRC_DIR/train.py"
    python3 "$SRC_DIR/train.py"
    check_status "全参数微调训练完成" "全参数微调训练失败"

    # 检查训练结果
    if [ -d "$full_param_output_dir" ]; then
        local checkpoint_count=$(find "$full_param_output_dir" -name "checkpoint-*" -type d | wc -l)
        log_success "全参数训练完成，生成 $checkpoint_count 个检查点"
    fi
}

# 模型评估函数
evaluate_models() {
    log_info "开始模型评估..."

    # 检查评估脚本是否存在
    if [ ! -f "$SRC_DIR/model_compare.py" ] || [ ! -f "$SRC_DIR/compare_param.py" ]; then
        log_error "模型评估脚本不存在"
        exit 1
    fi

    # 检查必要的模型文件
    local lora_model_path="$OUTPUT_DIR/Qwen3-0.6B/$LORA_CHECKPOINT"
    local full_param_model_path="$OUTPUT_DIR/Qwen3-param/$FULL_PARAM_CHECKPOINT"

    if [ ! -d "$lora_model_path" ]; then
        log_warning "LoRA模型检查点不存在: $lora_model_path，跳过LoRA评估"
    else
        log_info "评估LoRA微调模型..."
        cd "$BASE_DIR" || exit 1
        python3 "$SRC_DIR/model_compare.py"
        check_status "LoRA模型评估完成" "LoRA模型评估失败"
    fi

    if [ ! -d "$full_param_model_path" ]; then
        log_warning "全参数模型检查点不存在: $full_param_model_path，跳过多参数评估"
    else
        log_info "评估全参数微调模型..."
        cd "$BASE_DIR" || exit 1
        python3 "$SRC_DIR/compare_param.py"
        check_status "全参数模型评估完成" "全参数模型评估失败"
    fi

    # 检查评估结果文件
    if [ -f "$BASE_DIR/model_comparison_results.json" ]; then
        log_success "LoRA评估结果已保存: model_comparison_results.json"
    fi

    if [ -f "$BASE_DIR/full_param_model_comparison_results.json" ]; then
        log_success "全参数评估结果已保存: full_param_model_comparison_results.json"
    fi

    # 将结果文件移动到results目录
    mv -f "$BASE_DIR/model_comparison_results.json" "$RESULTS_DIR/" 2>/dev/null
    mv -f "$BASE_DIR/full_param_model_comparison_results.json" "$RESULTS_DIR/" 2>/dev/null
}

# 推理对比函数
run_inference() {
    log_info "开始推理对比..."

    # 检查推理脚本是否存在
    if [ ! -f "$SRC_DIR/infer.py" ]; then
        log_error "推理脚本不存在: $SRC_DIR/infer.py"
        exit 1
    fi

    # 执行推理对比
    cd "$BASE_DIR" || exit 1
    python3 "$SRC_DIR/infer.py"
    check_status "推理对比完成" "推理对比失败"

    log_success "三个模型的回答对比已完成，请查看上方输出"
}

# 主函数
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    Qwen3医疗大模型微调实验脚本${NC}"
    echo -e "${GREEN}========================================${NC}"

    # 参数解析
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        --all)
            log_info "执行完整实验流程"
            check_environment
            prepare_data
            train_lora
            train_full_param
            evaluate_models
            run_inference
            ;;
        --data-only)
            log_info "仅执行数据准备"
            check_environment
            prepare_data
            ;;
        --train-only)
            log_info "仅执行模型训练"
            check_environment
            train_lora
            train_full_param
            ;;
        --train-lora)
            log_info "仅执行LoRA微调训练"
            check_environment
            train_lora
            ;;
        --train-full)
            log_info "仅执行全参数微调训练"
            check_environment
            train_full_param
            ;;
        --eval-only)
            log_info "仅执行模型评估"
            check_environment
            evaluate_models
            ;;
        --infer-only)
            log_info "仅执行推理对比"
            check_environment
            run_inference
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}    实验流程执行完成！${NC}"
    echo -e "${GREEN}========================================${NC}"

    # 显示结果文件位置
    if [ -d "$RESULTS_DIR" ]; then
        echo -e "${BLUE}结果文件位于:${NC} $RESULTS_DIR/"
        ls -la "$RESULTS_DIR/" 2>/dev/null || echo "结果目录为空"
    fi

    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "${BLUE}模型文件位于:${NC} $OUTPUT_DIR/"
        find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | head -5
    fi
}

# 执行主函数
main "$@"