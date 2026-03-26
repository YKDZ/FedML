#!/usr/bin/env bash
# ================================================================
# M2 攻击实验 — GPU 优先子集
# PMR=0.2, alpha∈{0.1,0.5}, seeds={0,1,2}
# CIFAR10(50轮) + MNIST(50轮)
# 共 3 × 1 × 2 × 3 × 2 = 36 次实验
# ================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_ID="${GPU_ID:-1}" # 默认 GPU 1

ATTACKS="byzantine label_flipping model_replacement"
PMRS="0.2"
ALPHAS="0.1 0.5"
SEEDS="0 1 2"
CLIENTS=10
ROUNDS_CIFAR=50
ROUNDS_MNIST=50

TOTAL=$(echo "$ATTACKS" | wc -w)
TOTAL=$((TOTAL * 1 * 2 * 3 * 2))
COUNT=0

echo "=== M2 Attacks GPU (Priority Subset) ==="
echo "GPU_ID=${GPU_ID}  Total experiments: ${TOTAL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

echo "--- ResNet18 + CIFAR10 (${ROUNDS_CIFAR} rounds) ---"
for ATTACK in $ATTACKS; do
	for PMR in $PMRS; do
		for ALPHA in $ALPHAS; do
			for SEED in $SEEDS; do
				COUNT=$((COUNT + 1))
				echo ""
				echo "[$COUNT/$TOTAL] [M2-CIFAR10] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
				bash "$DIR/run_experiment.sh" \
					--model ResNet18 --dataset cifar10 \
					--attack "${ATTACK}" --defense none --aggregator fedavg \
					--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
					--rounds "${ROUNDS_CIFAR}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
					--max_samples 0 --test_subset 0 \
					--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
					--gpu_mapping mapping_single_gpu
			done
		done
	done
done

echo "--- LeNet5 + MNIST (${ROUNDS_MNIST} rounds) ---"
for ATTACK in $ATTACKS; do
	for PMR in $PMRS; do
		for ALPHA in $ALPHAS; do
			for SEED in $SEEDS; do
				COUNT=$((COUNT + 1))
				echo ""
				echo "[$COUNT/$TOTAL] [M2-MNIST] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
				bash "$DIR/run_experiment.sh" \
					--model LeNet5 --dataset mnist \
					--attack "${ATTACK}" --defense none --aggregator fedavg \
					--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
					--rounds "${ROUNDS_MNIST}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
					--max_samples 0 --test_subset 0 \
					--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
					--gpu_mapping mapping_single_gpu
			done
		done
	done
done

echo ""
echo "=== M2 Priority Subset DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
