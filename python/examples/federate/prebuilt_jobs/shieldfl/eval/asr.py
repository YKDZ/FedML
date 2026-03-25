"""
后门攻击成功率 (ASR) 评估。

触发器注入规则：
- 在图片右下角 trigger_size × trigger_size 区域填充 trigger_value
- 仅对 label != target_label 的样本注入触发器
- ASR = 被模型预测为 target_label 的比例

使用场景：
- model_replacement_backdoor_attack 实验的 ASR 评估
"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_asr(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_label: int = 0,
    trigger_size: int = 3,
    trigger_value: float = 1.0,
) -> Dict[str, float]:
    """计算后门攻击成功率 (ASR)。

    Args:
        model: 待评估的模型。
        test_loader: 测试数据加载器（干净数据）。
        device: 运行设备。
        target_label: 触发器目标标签（攻击者希望模型预测的类别）。
        trigger_size: 右下角触发器边长（像素）。
        trigger_value: 触发器填充值（对归一化后的数据而言通常用 1.0 或更大）。

    Returns:
        dict: {"asr": float, "asr_total": int, "asr_success": int}
    """
    model.eval()
    total = 0
    success = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.clone()
            labels = labels.to(device)
            # 仅处理标签不等于目标标签的样本
            non_target_mask = (labels != target_label)
            if non_target_mask.sum() == 0:
                continue
            # 在右下角注入触发器
            images[:, :, -trigger_size:, -trigger_size:] = trigger_value
            images = images.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, dim=1)
            # 仅统计原本不是目标类、触发后被预测为目标类的样本
            success += ((predicted == target_label) & non_target_mask).sum().item()
            total += non_target_mask.sum().item()

    asr = success / max(1, total)
    logging.info("ASR evaluation | total=%d | success=%d | asr=%.4f", total, success, asr)
    return {"asr": asr, "asr_total": total, "asr_success": success}
