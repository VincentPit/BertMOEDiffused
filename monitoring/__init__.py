"""
MLflow model monitoring utilities for BertMoEDiffusion.

Provides:
  - Data drift detection (input token distribution shifts)
  - Prediction quality monitoring (output entropy, perplexity tracking)
  - Performance metrics logging (latency, throughput)
  - Model validation against baseline thresholds

Usage:
    from monitoring.mlflow_monitor import ModelMonitor

    monitor = ModelMonitor(tracking_uri="http://localhost:5000")
    monitor.log_prediction_metrics(prompts, outputs, latency_ms)
    monitor.check_data_drift(reference_tokens, new_tokens)
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Dict, List, Optional

import mlflow
import numpy as np

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Production monitoring for BertMoEDiffusion via MLflow."""

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "BertMoEDiffusion-monitoring",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._request_count = 0
        self._total_latency = 0.0

    def start_monitoring_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a monitoring run for logging metrics."""
        self._run = mlflow.start_run(run_name=run_name or "monitoring")
        return self._run

    def end_monitoring_run(self) -> None:
        mlflow.end_run()

    # ── Prediction quality ─────────────────────────────────────────────────────

    def log_prediction_metrics(
        self,
        prompts: List[str],
        outputs: List[str],
        latency_ms: float,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Log prediction-level quality and performance metrics.

        Returns dict of computed metrics.
        """
        batch_size = len(prompts)
        self._request_count += batch_size
        self._total_latency += latency_ms

        # Output length stats
        output_lengths = [len(o.split()) for o in outputs]
        avg_output_len = np.mean(output_lengths)
        std_output_len = np.std(output_lengths)

        # Input length stats
        input_lengths = [len(p.split()) for p in prompts]
        avg_input_len = np.mean(input_lengths)

        # Diversity: unique token ratio in outputs
        all_tokens = " ".join(outputs).split()
        unique_ratio = len(set(all_tokens)) / max(len(all_tokens), 1)

        # Repetition rate: fraction of consecutive repeated tokens
        repeated = sum(
            1 for i in range(1, len(all_tokens)) if all_tokens[i] == all_tokens[i - 1]
        )
        repetition_rate = repeated / max(len(all_tokens) - 1, 1)

        metrics = {
            "monitor/batch_size": batch_size,
            "monitor/latency_ms": latency_ms,
            "monitor/latency_per_sample_ms": latency_ms / batch_size,
            "monitor/throughput_samples_per_sec": (batch_size / latency_ms) * 1000,
            "monitor/avg_output_length": avg_output_len,
            "monitor/std_output_length": std_output_len,
            "monitor/avg_input_length": avg_input_len,
            "monitor/unique_token_ratio": unique_ratio,
            "monitor/repetition_rate": repetition_rate,
            "monitor/total_requests": self._request_count,
        }

        mlflow.log_metrics(metrics, step=step)
        return metrics

    # ── Data drift detection ───────────────────────────────────────────────────

    def check_data_drift(
        self,
        reference_tokens: List[str],
        new_tokens: List[str],
        threshold: float = 0.1,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Detect distributional drift between reference and new token distributions.

        Uses Jensen-Shannon divergence. Logs drift metrics and returns them.
        """
        ref_dist = self._token_distribution(reference_tokens)
        new_dist = self._token_distribution(new_tokens)

        # Compute JS divergence
        all_tokens = set(ref_dist.keys()) | set(new_dist.keys())
        p = np.array([ref_dist.get(t, 1e-10) for t in all_tokens])
        q = np.array([new_dist.get(t, 1e-10) for t in all_tokens])

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)
        js_div = float(
            0.5 * np.sum(p * np.log(p / m + 1e-10))
            + 0.5 * np.sum(q * np.log(q / m + 1e-10))
        )

        drift_detected = js_div > threshold

        metrics = {
            "drift/js_divergence": js_div,
            "drift/threshold": threshold,
            "drift/detected": float(drift_detected),
            "drift/ref_vocab_size": len(ref_dist),
            "drift/new_vocab_size": len(new_dist),
        }

        mlflow.log_metrics(metrics, step=step)

        if drift_detected:
            logger.warning(
                f"Data drift detected! JS divergence={js_div:.4f} > threshold={threshold}"
            )
            mlflow.set_tag("drift_alert", "true")

        return metrics

    # ── Model validation ───────────────────────────────────────────────────────

    def validate_model(
        self,
        model_name: str,
        version: int,
        validation_metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Validate a model version against quality thresholds.

        Default thresholds check BPD and latency. Returns True if model passes.
        """
        if thresholds is None:
            thresholds = {
                "max_bpd": 6.0,
                "max_latency_ms": 5000.0,
                "min_unique_ratio": 0.3,
            }

        passed = True
        results = {}

        bpd = validation_metrics.get("bpd")
        if bpd is not None and bpd > thresholds.get("max_bpd", float("inf")):
            logger.warning(f"Validation FAILED: BPD {bpd:.4f} > {thresholds['max_bpd']}")
            passed = False
            results["validation/bpd_pass"] = 0.0
        elif bpd is not None:
            results["validation/bpd_pass"] = 1.0

        latency = validation_metrics.get("latency_ms")
        if latency is not None and latency > thresholds.get("max_latency_ms", float("inf")):
            logger.warning(f"Validation FAILED: latency {latency:.1f}ms > {thresholds['max_latency_ms']}ms")
            passed = False
            results["validation/latency_pass"] = 0.0
        elif latency is not None:
            results["validation/latency_pass"] = 1.0

        unique_ratio = validation_metrics.get("unique_ratio")
        if unique_ratio is not None and unique_ratio < thresholds.get("min_unique_ratio", 0):
            logger.warning(f"Validation FAILED: unique ratio {unique_ratio:.4f} < {thresholds['min_unique_ratio']}")
            passed = False
            results["validation/unique_ratio_pass"] = 0.0
        elif unique_ratio is not None:
            results["validation/unique_ratio_pass"] = 1.0

        results["validation/overall_pass"] = float(passed)
        mlflow.log_metrics(results)
        mlflow.set_tag("validation_passed", str(passed))

        if passed:
            logger.info(f"Model {model_name} v{version} passed validation.")
        else:
            logger.warning(f"Model {model_name} v{version} FAILED validation.")

        return passed

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _token_distribution(tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()}
