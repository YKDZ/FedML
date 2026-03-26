#!/usr/bin/env bash
# ================================================================
# M3 防御实验 — GPU 优先子集
# 内置防御: krum,coordinate_wise_trimmed_mean  × byzantine × PMR=0.2
#           × alpha∈{0.1,0.5} × seeds={0,1,2}
# VeriFL:   byzantine × PMR=0.2 × alpha∈{0.1,0.5} × seeds={0,1,2}
# CIFAR10(50轮) + MNIST(50轮)
# 共 (2+1) × 1 × 1 × 2 × 3 × 2 = 36 次实验
# ================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_ID="${GPU_ID:-3}" # 默认 GPU 3

DEFENSES="krum coordinate_wise_trimmed_mean"
ATTACKS="byzantine"
PMRS="0.2"
ALPHAS="0.1 0.5"
SEEDS="0 1 2"
CLIENTS=10
ROUNDS_CIFAR=50
ROUNDS_MNIST=50

# 计算总数: (2 defense + 1 verifl) × 1 atk × 1 pmr × 2 alpha × 3 seeds × 2 tasks
TOTAL=$(((2 + 1) * 1 * 2 * 3 * 2))
COUNT=0

echo "=== M3 Defense GPU (Priority Subset) ==="
echo "GPU_ID=${GPU_ID}  Total experiments: ${TOTAL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

echo "--- Part A: FedML 内置防御 + FedAvg ---"
for DEFENSE in $DEFENSES; do
	for ATTACK in $ATTACKS; do
		for PMR in $PMRS; do
			for ALPHA in $ALPHAS; do
				for SEED in $SEEDS; do
					COUNT=$((COUNT + 1))
					echo ""
					echo "[$COUNT/$TOTAL] [M3-DEF-CIFAR10] defense=${DEFENSE} attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
					bash "$DIR/run_experiment.sh" \
						--model ResNet18 --dataset cifar10 \
						--attack "${ATTACK}" --defense "${DEFENSE}" --aggregator fedavg \
						--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
						--rounds "${ROUNDS_CIFAR}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
						--max_samples 0 --test_subset 0 \
						--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
						--gpu_mapping mapping_single_gpu
				done
			done
		done
	done
done

for DEFENSE in $DEFENSES; do
	for ATTACK in $ATTACKS; do
		for PMR in $PMRS; do
			for ALPHA in $ALPHAS; do
				for SEED in $SEEDS; do
					COUNT=$((COUNT + 1))
					echo ""
					echo "[$COUNT/$TOTAL] [M3-DEF-MNIST] defense=${DEFENSE} attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
					bash "$DIR/run_experiment.sh" \
						--model LeNet5 --dataset mnist \
						--attack "${ATTACK}" --defense "${DEFENSE}" --aggregator fedavg \
						--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
						--rounds "${ROUNDS_MNIST}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
						--max_samples 0 --test_subset 0 \
						--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
						--gpu_mapping mapping_single_gpu
				done
			done
		done
	done
done

echo "--- Part B: VeriFL 聚合器 (defense=none, aggregator=verifl) ---"
for ATTACK in $ATTACKS; do
	for PMR in $PMRS; do
		for ALPHA in $ALPHAS; do
			for SEED in $SEEDS; do
				COUNT=$((COUNT + 1))
				echo ""
				echo "[$COUNT/$TOTAL] [M3-VeriFL-CIFAR10] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
				bash "$DIR/run_experiment.sh" \
					--model ResNet18 --dataset cifar10 \
					--attack "${ATTACK}" --defense none --aggregator verifl \
					--pmr "${PMR}" --alpha "${ALPHA}" --seed "${SEED}" \
					--rounds "${ROUNDS_CIFAR}" --clients "${CLIENTS}" --epochs 1 --batch_size 64 \
					--max_samples 0 --test_subset 0 \
					--gpu --gpu_id "${GPU_ID}" --runtime single-gpu-deterministic \
					--gpu_mapping mapping_single_gpu
			done
		done
	done
done

for ATTACK in $ATTACKS; do
	for PMR in $PMRS; do
		for ALPHA in $ALPHAS; do
			for SEED in $SEEDS; do
				COUNT=$((COUNT + 1))
				echo ""
				echo "[$COUNT/$TOTAL] [M3-VeriFL-MNIST] attack=${ATTACK} pmr=${PMR} alpha=${ALPHA} seed=${SEED}  $(date '+%H:%M:%S')"
				bash "$DIR/run_experiment.sh" \
					--model LeNet5 --dataset mnist \
					--attack "${ATTACK}" --defense none --aggregator verifl \
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
echo "=== M3 Priority Subset DONE at $(date '+%Y-%m-%d %H:%M:%S') ==="
