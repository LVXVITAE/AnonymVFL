#!/usr/bin/env python3
"""Regenerate all performance plots from CSV data."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import (
    plot_time_vs_samples,
    plot_batch_size_impact,
    plot_inference_latency,
    plot_wan_network_impact,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "test_results", "performance")


def _read(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  SKIP {name}: file not found")
        return None
    df = pd.read_csv(path)
    print(f"  {name}: {len(df)} rows")
    return df.to_dict("records")


def main():
    print("=== Regenerating all performance plots ===\n")

    # 1. psi_scalability
    print("[1/11] PSI scalability")
    records = _read("psi_scalability.csv")
    if records:
        plot_time_vs_samples(records, "PSI 对齐时间 vs 样本数量",
                              os.path.join(RESULTS_DIR, "psi_scalability.png"),
                              y_key="对齐时间(s)")

    # 2. sslr_scalability
    print("[2/11] SSLR scalability")
    records = _read("sslr_scalability.csv")
    if records:
        plot_time_vs_samples(records, "SSLR 训练时间 vs 样本数量",
                              os.path.join(RESULTS_DIR, "sslr_scalability.png"))

    # 3. xgboost_scalability
    print("[3/11] SSXGBoost scalability")
    records = _read("xgboost_scalability.csv")
    if records:
        plot_time_vs_samples(records, "SSXGBoost 训练时间 vs 样本数量",
                              os.path.join(RESULTS_DIR, "xgboost_scalability.png"))

    # 4. batch_size_impact
    print("[4/11] Batch size impact")
    records = _read("batch_size_impact.csv")
    if records:
        plot_batch_size_impact(records, os.path.join(RESULTS_DIR, "batch_size_impact.png"))

    # 5. lr_inference_latency
    print("[5/11] LR inference latency")
    records = _read("lr_inference_latency.csv")
    if records:
        plot_inference_latency(records, os.path.join(RESULTS_DIR, "lr_inference_latency.png"))

    # 6. xgboost_inference_latency
    print("[6/11] XGBoost inference latency")
    records = _read("xgboost_inference_latency.csv")
    if records:
        plot_inference_latency(records, os.path.join(RESULTS_DIR, "xgboost_inference_latency.png"))

    # 7. quantile_impact
    print("[7/11] Quantile impact")
    records = _read("quantile_impact.csv")
    if records:
        plot_time_vs_samples(records, "分位点数量 k 对 SSXGBoost 训练时间的影响",
                              os.path.join(RESULTS_DIR, "quantile_impact.png"),
                              x_key="分位点数量k", y_key="总训练时间(s)")

    # 8. psi_stress
    print("[8/11] PSI stress")
    records = _read("psi_stress.csv")
    if records:
        plot_time_vs_samples(records, "PSI 压力测试: 对齐时间 vs 样本数量",
                              os.path.join(RESULTS_DIR, "psi_stress.png"),
                              x_key="样本数量", y_key="对齐时间(s)")

    # 9. sslr_stress
    print("[9/11] SSLR stress")
    records = _read("sslr_stress.csv")
    if records:
        plot_time_vs_samples(records, "SSLR 压力测试: 训练时间 vs 样本数量",
                              os.path.join(RESULTS_DIR, "sslr_stress.png"),
                              x_key="样本数量", y_key="总时间(s)")

    # 10. xgboost_stress
    print("[10/11] SSXGBoost stress")
    records = _read("xgboost_stress.csv")
    if records:
        plot_time_vs_samples(records, "SSXGBoost 压力测试: 训练时间 vs 样本数量",
                              os.path.join(RESULTS_DIR, "xgboost_stress.png"),
                              x_key="样本数量", y_key="总时间(s)")

    # 11. WAN network impact
    print("[11/11] WAN network impact plots")
    for model, csv_name, y_key, title in [
        ("psi", "psi_network_impact_wan.csv", "对齐时间(s)", "WAN 条件对 PSI 对齐时间的影响"),
        ("sslr", "sslr_network_impact_wan.csv", "总时间(s)", "WAN 条件对 SSLR 训练时间的影响"),
        ("xgboost", "xgboost_network_impact_wan.csv", "总时间(s)", "WAN 条件对 SSXGBoost 训练时间的影响"),
    ]:
        records = _read(csv_name)
        if records:
            png_name = f"{model}_network_impact_wan.png"
            plot_wan_network_impact(records, title,
                                     os.path.join(RESULTS_DIR, png_name), y_key)

    print("\n=== All plots regenerated ===")


if __name__ == "__main__":
    main()