#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FILE: ablation_runner.py
ROLE: Automated Scientific Experiment & Ablation Suite:
      - Table 1: Main Benchmark (ZS-SBIR, GZS-SBIR, FG-ZS-SBIR on Split 2)
      - Table 2: Loss Component Ablation (L_triplet, +L_NT-Xent, +L_class, +L_consist)
      - Table 3: Sketch CV Pipeline Ablation (Raw Resize vs. Tight-Crop + 1:1 Pad + Morph)
      - Table 4: LoRA Rank Sensitivity (r in {4, 8, 16}) & LayerNorm Unfreezing
      Exports clean Markdown & CSV tables directly ready for the final report.
PROJECT: Zero-Shot & Generalized Sketch-Based Image Retrieval (CS331)
================================================================================
"""

import os
import csv
import argparse
from typing import List, Dict, Any, Tuple

from config import SBIRConfig


ABLATION_EXPERIMENTS: List[Dict[str, Any]] = [
    # --- Table 2: Loss Component Ablation (matches Table 5 in SeCo-SBIR MM'26) ---
    {
        "exp_id": "T2_1_Triplet_Only",
        "table": "Table2_Loss_Ablation",
        "description": "L_triplet only (Single-objective baseline)",
        "overrides": {"lambda_cls": 0.0, "lambda_consist": 0.0, "lora_rank": 8},
        "use_ntxent": False,
    },
    {
        "exp_id": "T2_2_Triplet_NTXent",
        "table": "Table2_Loss_Ablation",
        "description": "L_triplet + L_NT-Xent (Contrastive Dual-Loss)",
        "overrides": {"lambda_cls": 0.0, "lambda_consist": 0.0, "lora_rank": 8},
        "use_ntxent": True,
    },
    {
        "exp_id": "T2_3_Add_TextClass",
        "table": "Table2_Loss_Ablation",
        "description": "L_triplet + L_NT-Xent + L_class (Text Semantic Anchor)",
        "overrides": {"lambda_cls": 1.0, "lambda_consist": 0.0, "lora_rank": 8},
        "use_ntxent": True,
    },
    {
        "exp_id": "T2_4_Full_SeCo_LoRA",
        "table": "Table2_Loss_Ablation",
        "description": "L_triplet + L_NT-Xent + L_class + L_consist (Full Proposed)",
        "overrides": {"lambda_cls": 1.0, "lambda_consist": 1.0, "lora_rank": 8},
        "use_ntxent": True,
    },
    # --- Table 3: CV Preprocessing & Perturbation Ablation ---
    {
        "exp_id": "T3_1_No_Morph_No_Erase",
        "table": "Table3_CV_Pipeline",
        "description": "Tight-Crop + 1:1 Pad only (w/o Morphology & Random Erase)",
        "overrides": {"enable_morph": False, "enable_random_erase": False},
        "use_ntxent": True,
    },
    {
        "exp_id": "T3_2_With_Morph_Only",
        "table": "Table3_CV_Pipeline",
        "description": "Tight-Crop + 1:1 Pad + Morphological Dilation/Erosion",
        "overrides": {"enable_morph": True, "enable_random_erase": False},
        "use_ntxent": True,
    },
    # --- Table 4: LoRA Rank & LayerNorm Architecture Ablation ---
    {
        "exp_id": "T4_1_LoRA_Rank4",
        "table": "Table4_Architecture",
        "description": "LoRA Rank r=4 + Unfrozen LayerNorm",
        "overrides": {"lora_rank": 4, "lora_alpha": 8.0, "unfreeze_sketch_ln": True},
        "use_ntxent": True,
    },
    {
        "exp_id": "T4_2_LoRA_Rank16",
        "table": "Table4_Architecture",
        "description": "LoRA Rank r=16 + Unfrozen LayerNorm",
        "overrides": {"lora_rank": 16, "lora_alpha": 32.0, "unfreeze_sketch_ln": True},
        "use_ntxent": True,
    },
    {
        "exp_id": "T4_3_Frozen_LayerNorm",
        "table": "Table4_Architecture",
        "description": "LoRA Rank r=8 with Frozen LayerNorm (w/o LN tuning)",
        "overrides": {"lora_rank": 8, "lora_alpha": 16.0, "unfreeze_sketch_ln": False},
        "use_ntxent": True,
    },
]


def export_ablation_tables(results_list: List[Dict[str, Any]], out_dir: str) -> Tuple[str, str]:
    """Exports structured CSV and Markdown report tables from experiment records."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ablation_results_summary.csv")
    md_path = os.path.join(out_dir, "ablation_results_summary.md")

    fieldnames = [
        "exp_id", "table", "description",
        "ZS_mAP@200", "ZS_P@200", "ZS_mAP@all", "ZS_P@100",
        "GZS_mAP@200", "GZS_P@200", "FG_Acc@1", "FG_Acc@5", "OOD_AUROC"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_list:
            writer.writerow({k: row.get(k, "N/A") for k in fieldnames})

    md_lines = [
        "# Kết quả Thực nghiệm & Ablation Study (Sketchy-Extended Split 2: 104 Seen / 21 Unseen)\n",
        "| Mã TN | Nhóm Bảng | Cấu hình Thử nghiệm | ZS mAP@200 | ZS P@200 | GZS mAP@200 | FG Acc@1 | OOD AUROC |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results_list:
        md_lines.append(
            f"| `{r.get('exp_id')}` | {r.get('table')} | {r.get('description')} | "
            f"{r.get('ZS_mAP@200')} | {r.get('ZS_P@200')} | {r.get('GZS_mAP@200')} | "
            f"{r.get('FG_Acc@1')} | {r.get('OOD_AUROC')} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return csv_path, md_path


def run_ablation_self_test() -> None:
    """Verifies experiment definitions and Markdown/CSV table export."""
    assert len(ABLATION_EXPERIMENTS) == 9
    dummy_rows = []
    for exp in ABLATION_EXPERIMENTS:
        cfg = SBIRConfig(**exp["overrides"])
        assert cfg.image_size == 224
        dummy_rows.append({
            "exp_id": exp["exp_id"],
            "table": exp["table"],
            "description": exp["description"],
            "ZS_mAP@200": "Pending GPU Run",
            "ZS_P@200": "Pending GPU Run",
            "ZS_mAP@all": "Pending GPU Run",
            "ZS_P@100": "Pending GPU Run",
            "GZS_mAP@200": "Pending GPU Run",
            "GZS_P@200": "Pending GPU Run",
            "FG_Acc@1": "Pending GPU Run",
            "FG_Acc@5": "Pending GPU Run",
            "OOD_AUROC": "Pending GPU Run",
        })
    csv_p, md_p = export_ablation_tables(dummy_rows, "/tmp/sbir_ablation_test")
    assert os.path.exists(csv_p) and os.path.exists(md_p)
    print(f"[PASS] ablation_runner.py: Verified 9 ablation configs & table exporter ({md_p}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS331 Automated Ablation Study Suite")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    run_ablation_self_test()