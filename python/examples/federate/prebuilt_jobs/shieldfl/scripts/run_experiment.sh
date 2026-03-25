#!/usr/bin/env bash
# 使用方法：
#   bash scripts/run_experiment.sh \
#     --model ResNet18 --dataset cifar10 \
#     --attack byzantine --defense none --aggregator fedavg \
#     --pmr 0.2 --alpha 0.5 --seed 0 \
#     --rounds 3 --clients 5 --epochs 1 --batch_size 32

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ----------- 参数默认值 -----------
MODEL="SimpleCNN"
DATASET="cifar10"
ATTACK="none"
DEFENSE="none"
AGGREGATOR="fedavg"
PMR="0.0"
ALPHA="0.5"
SEED="0"
ROUNDS="3"
CLIENTS="3"
EPOCHS="1"
BATCH="32"
MAX_SAMPLES="300"
VAL_PER_CLASS="50"
TEST_SUBSET="500"
GPU="false"
RUNTIME_MODE="cpu-deterministic"
GPU_MAPPING_KEY="mapping_default"
CPU_TRANSFER="true"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)      MODEL="$2";       shift 2 ;;
    --dataset)    DATASET="$2";     shift 2 ;;
    --attack)     ATTACK="$2";      shift 2 ;;
    --defense)    DEFENSE="$2";     shift 2 ;;
    --aggregator) AGGREGATOR="$2";  shift 2 ;;
    --pmr)        PMR="$2";         shift 2 ;;
    --alpha)      ALPHA="$2";       shift 2 ;;
    --seed)       SEED="$2";        shift 2 ;;
    --rounds)     ROUNDS="$2";      shift 2 ;;
    --clients)    CLIENTS="$2";     shift 2 ;;
    --epochs)     EPOCHS="$2";      shift 2 ;;
    --batch_size) BATCH="$2";       shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --gpu)          GPU="true";            shift 1 ;;
    --runtime)      RUNTIME_MODE="$2";     shift 2 ;;
    --gpu_mapping)  GPU_MAPPING_KEY="$2";  shift 2 ;;
    --cpu_transfer) CPU_TRANSFER="$2";     shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# GPU 训练时 MPI 通信仍走 CPU tensor
if [[ "$GPU" == "true" ]]; then
  CPU_TRANSFER="true"
fi

# ----------- 计算攻击参数 -----------
ENABLE_ATTACK="false"
ATTACK_TYPE="none"
BYZANTINE_NUM=0
EVAL_ASR="false"

if [[ "$ATTACK" != "none" ]]; then
  ENABLE_ATTACK="true"
  ATTACK_TYPE="$ATTACK"
  BYZANTINE_NUM=$(python3 -c "import math; print(max(1, math.ceil($CLIENTS * $PMR)))")
  if [[ "$ATTACK" == "model_replacement" ]]; then
    EVAL_ASR="true"
  fi
fi

ENABLE_DEFENSE="false"
DEFENSE_TYPE="none"
if [[ "$DEFENSE" != "none" ]]; then
  ENABLE_DEFENSE="true"
  DEFENSE_TYPE="$DEFENSE"
fi

# ----------- 生成临时配置 -----------
CONFIG_FILE="/tmp/shieldfl_exp_${MODEL}_${DATASET}_${ATTACK}_${DEFENSE}_a${ALPHA}_s${SEED}.yaml"
WORKER_NUM=$CLIENTS

cat > "$CONFIG_FILE" <<EOF
common_args:
  training_type: "cross_silo"
  random_seed: ${SEED}

data_args:
  dataset: "${DATASET}"
  data_cache_dir: ./data
  partition_method: "hetero"
  partition_alpha: ${ALPHA}
  val_per_class: ${VAL_PER_CLASS}
  trust_per_class: ${VAL_PER_CLASS}
  max_samples_per_client: ${MAX_SAMPLES}
  test_subset_size: ${TEST_SUBSET}
  num_workers: 0

model_args:
  model: "${MODEL}"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: ${CLIENTS}
  client_num_per_round: ${CLIENTS}
  comm_round: ${ROUNDS}
  epochs: ${EPOCHS}
  batch_size: ${BATCH}
  client_optimizer: sgd
  learning_rate: 0.01
  weight_decay: 0.0
  momentum: 0.9
  server_momentum: 0.9
  server_lr: 0.3
  pop_size: 15
  generations: 10
  lambda_reg: 0.01
  cpu_transfer: ${CPU_TRANSFER}
  enable_attack: ${ENABLE_ATTACK}
  attack_type: "${ATTACK_TYPE}"
  byzantine_client_num: ${BYZANTINE_NUM}
  attack_mode: "flip"
  enable_defense: ${ENABLE_DEFENSE}
  defense_type: "${DEFENSE_TYPE}"
  eval_asr: ${EVAL_ASR}
  target_label: 0
  trigger_size: 3
  trigger_value: 1.0

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: ${WORKER_NUM}
  using_gpu: ${GPU}
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: ${GPU_MAPPING_KEY}

comm_args:
  backend: "MPI"
  is_mobile: 0

tracking_args:
  log_file_dir: ./log
  enable_wandb: false
  using_mlops: false

shieldfl_args:
  runtime_mode: "${RUNTIME_MODE}"
  enforce_determinism: true
  sort_client_updates: true
  aggregator_type: "${AGGREGATOR}"
  metrics_output_dir: "./results"
EOF

echo "=== ShieldFL Experiment ==="
echo "  model=${MODEL} dataset=${DATASET} attack=${ATTACK} defense=${DEFENSE}"
echo "  aggregator=${AGGREGATOR} pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
echo "  rounds=${ROUNDS} clients=${CLIENTS} epochs=${EPOCHS}"
echo "  gpu=${GPU} runtime=${RUNTIME_MODE} gpu_mapping=${GPU_MAPPING_KEY}"
echo "  config=${CONFIG_FILE}"

cd "$SCRIPT_DIR"
TOTAL_PROC=$((WORKER_NUM + 1))
mpirun -np $TOTAL_PROC python main_fedml_shieldfl.py --cf "$CONFIG_FILE"
