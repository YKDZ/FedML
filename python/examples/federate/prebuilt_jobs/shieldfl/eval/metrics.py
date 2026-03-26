"""
结构化指标采集器。
每轮 append 一行 JSON 到指标文件，字段包括：
{
    "round": int,
    "aggregator": str,
    "attack_type": str,
    "defense_type": str,
    "test_accuracy": float,
    "test_loss": float,
    "test_total": int,
    "asr": float | null,
    "agg_time": float | null,
    "timestamp": str
}
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional


class MetricsCollector:
    def __init__(self, output_dir: str, args):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        aggregator_type = str(getattr(args, "aggregator_type", "verifl"))
        attack_type = str(getattr(args, "attack_type", "none"))
        defense_type = str(getattr(args, "defense_type", "none"))
        model_name = str(getattr(args, "model", "unknown"))
        dataset = str(getattr(args, "dataset", "unknown"))
        seed = str(getattr(args, "random_seed", 0))
        alpha = str(getattr(args, "partition_alpha", "unknown"))

        fname = f"metrics_{model_name}_{dataset}_{aggregator_type}_atk{attack_type}_def{defense_type}_a{alpha}_seed{seed}.jsonl"
        self.metrics_file = os.path.join(output_dir, fname)
        import torch
        self._meta = {
            "aggregator": aggregator_type,
            "attack_type": attack_type,
            "defense_type": defense_type,
            "model": model_name,
            "dataset": dataset,
            "alpha": alpha,
            "runtime_mode": str(getattr(args, "runtime_mode", "unknown")),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        }
        logging.info("MetricsCollector initialized | output=%s", self.metrics_file)

    def log_round(
        self,
        round_idx: int,
        test_accuracy: float,
        test_loss: float,
        test_total: int,
        asr: Optional[float] = None,
        agg_time: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        record = {
            "round": round_idx,
            "test_accuracy": round(float(test_accuracy), 6),
            "test_loss": round(float(test_loss), 6),
            "test_total": int(test_total),
            "asr": round(float(asr), 6) if asr is not None else None,
            "agg_time": round(float(agg_time), 4) if agg_time is not None else None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        record.update(self._meta)
        if extra:
            record.update(extra)
        try:
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logging.warning("MetricsCollector failed to write: %s", exc)
