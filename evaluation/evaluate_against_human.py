"""
Сверка предсказаний пайплайна с эталонной разметкой людей.

Эталон — XLSX-файлы вида «Отчет для ИИ DD.MM.YYYY.xlsx», где:
- колонка `Профиль` — категория, проставленная человеком;
- непустая колонка `Действие` — означает «тендер взяли в работу» (positive).

Предсказание — JSONL-файл из пайплайна
(`storage/tenders/jsonl/all_tenders_data_*.jsonl`), где для каждого тендера
есть как минимум поля `number`, `predicted_category`, `is_interesting`,
`interest_score`. Если есть deep_rag — учитывается также `deep_rag_decision`.

Скрипт сопоставляет записи по `Номер тендера` ↔ `number` и считает
confusion-матрицу + по-категорийную точность.

Запуск:
    python -m evaluation.evaluate_against_human \
        --gt "Отчет для ИИ 21.04.2026.xlsx" \
        --pred storage/tenders/jsonl/all_tenders_data_20260422_010101.jsonl

Можно передать несколько эталонов:
    --gt "Отчет для ИИ 21.04.2026.xlsx" "Отчет для ИИ 22.04.2026.xlsx"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd


GT_NUMBER_COLS = ("Номер тендера", "Номер", "Номер закупки")
GT_CATEGORY_COL = "Профиль"
GT_ACTION_COL = "Действие"
GT_TITLE_COLS = ("Предмет тендера", "Наименование закупки")


def _normalize_number(s) -> str:
    if s is None:
        return ""
    return str(s).strip().lstrip("№").strip()


def load_ground_truth(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_excel(p)
        # Найти колонку номера
        num_col = next((c for c in GT_NUMBER_COLS if c in df.columns), None)
        if num_col is None:
            raise ValueError(f"В {p} нет колонки с номером тендера: {list(df.columns)}")
        title_col = next((c for c in GT_TITLE_COLS if c in df.columns), None)
        out = pd.DataFrame({
            "_src": os.path.basename(p),
            "number": df[num_col].apply(_normalize_number),
            "gt_category": df.get(GT_CATEGORY_COL, "").fillna("").astype(str).str.strip(),
            "gt_action": df.get(GT_ACTION_COL, "").fillna("").astype(str).str.strip(),
            "gt_title": (df[title_col].fillna("").astype(str) if title_col else ""),
        })
        out["gt_interesting"] = out["gt_action"] != ""
        frames.append(out)
    gt = pd.concat(frames, ignore_index=True)
    gt = gt[gt["number"] != ""].reset_index(drop=True)
    return gt


def load_predictions(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({
                "number": _normalize_number(rec.get("number")),
                "pred_category": rec.get("predicted_category", ""),
                "is_interesting": bool(rec.get("is_interesting", False)),
                "interest_score": float(rec.get("interest_score", 0.0) or 0.0),
                "rag_signal": float(rec.get("rag_interest_signal", 0.0) or 0.0),
                "deep_decision": rec.get("deep_rag_decision", ""),
                "deep_score": float(rec.get("deep_rag_score", 0.0) or 0.0),
                "applied_rule": (rec.get("deep_rag_reasoning", "") or "")[:200],
                "title": rec.get("title", ""),
            })
    return pd.DataFrame(rows)


def confusion(gt: pd.Series, pred: pd.Series) -> Tuple[int, int, int, int]:
    tp = int(((gt) & (pred)).sum())
    fp = int(((~gt) & (pred)).sum())
    fn = int(((gt) & (~pred)).sum())
    tn = int(((~gt) & (~pred)).sum())
    return tp, fp, fn, tn


def fmt_metrics(tp: int, fp: int, fn: int, tn: int) -> str:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return (
        f"  TP={tp:>4} FP={fp:>4} FN={fn:>4} TN={tn:>4} | "
        f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} acc={accuracy:.3f}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", nargs="+", required=True, help="Один или несколько XLSX эталонов")
    p.add_argument("--pred", required=True, help="JSONL с предсказаниями пайплайна")
    p.add_argument("--use-deep", action="store_true",
                   help="Использовать deep_rag_decision == 'Подходит' как pred вместо is_interesting")
    p.add_argument("--show-errors", type=int, default=20,
                   help="Сколько ошибок (FN/FP) распечатать (по 20 на каждый тип). 0 — не печатать.")
    args = p.parse_args()

    gt_df = load_ground_truth(args.gt)
    pred_df = load_predictions(args.pred)

    merged = gt_df.merge(pred_df, on="number", how="left", suffixes=("", "_pred"))
    matched = merged.dropna(subset=["pred_category"]).reset_index(drop=True)
    unmatched = merged[merged["pred_category"].isna()].reset_index(drop=True)

    print(f"Эталон: {len(gt_df)} записей, предсказаний: {len(pred_df)}")
    print(f"Совпало по number: {len(matched)} | без предсказания: {len(unmatched)}")
    print()

    if matched.empty:
        print("⚠  Нет совпадений по номеру тендера — проверьте, что эталон и пайплайн пересекаются по периоду.")
        return 1

    if args.use_deep:
        pred_pos = matched["deep_decision"] == "Подходит"
        pred_label = "deep_rag_decision == 'Подходит'"
    else:
        pred_pos = matched["is_interesting"].astype(bool)
        pred_label = "is_interesting"

    gt_pos = matched["gt_interesting"].astype(bool)

    # Общая confusion
    tp, fp, fn, tn = confusion(gt_pos, pred_pos)
    print(f"=== Общая метрика ({pred_label}) ===")
    print(fmt_metrics(tp, fp, fn, tn))
    print()

    # Confusion по категории эталона
    print("=== По эталонной категории (Профиль) ===")
    for cat, sub in matched.groupby(matched["gt_category"].fillna("")):
        if cat == "":
            continue
        sub_gt = sub["gt_interesting"].astype(bool)
        sub_pred = (sub["deep_decision"] == "Подходит") if args.use_deep else sub["is_interesting"].astype(bool)
        ttp, tfp, tfn, ttn = confusion(sub_gt, sub_pred)
        total_gt_pos = int(sub_gt.sum())
        print(f"[{cat}] (всего={len(sub)}, эталон+={total_gt_pos})")
        print(fmt_metrics(ttp, tfp, tfn, ttn))
    print()

    # Cross-tab по категории
    print("=== Cross-tab predicted_category × gt_category ===")
    ct = pd.crosstab(matched["pred_category"].fillna("(none)"),
                     matched["gt_category"].fillna("(none)"))
    print(ct.to_string())
    print()

    # Распределение interest_score / deep_score у positive vs negative
    if "interest_score" in matched.columns:
        pos = matched[gt_pos]["interest_score"]
        neg = matched[~gt_pos]["interest_score"]
        print("=== interest_score: эталон+ vs эталон- ===")
        print(f"  positive: n={len(pos):>4} median={pos.median():.3f} mean={pos.mean():.3f} min={pos.min():.3f} max={pos.max():.3f}")
        print(f"  negative: n={len(neg):>4} median={neg.median():.3f} mean={neg.mean():.3f} min={neg.min():.3f} max={neg.max():.3f}")
        print()

    # Печатаем ошибки
    if args.show_errors > 0:
        fn_rows = matched[gt_pos & ~pred_pos].sort_values("interest_score", ascending=False)
        fp_rows = matched[~gt_pos & pred_pos].sort_values("interest_score", ascending=False)

        print(f"=== FN (упустили — пропустили интересные) — топ {min(args.show_errors, len(fn_rows))} ===")
        for _, r in fn_rows.head(args.show_errors).iterrows():
            print(f"  [{r['gt_category']:>30}|{r.get('pred_category',''):>20}] "
                  f"score={r.get('interest_score', 0):.2f} action={r['gt_action']!r} "
                  f"| {str(r.get('gt_title') or r.get('title',''))[:120]}")
        print()

        print(f"=== FP (взяли лишнее — попало в is_interesting, но человек не брал) — топ {min(args.show_errors, len(fp_rows))} ===")
        for _, r in fp_rows.head(args.show_errors).iterrows():
            print(f"  [{r['gt_category']:>30}|{r.get('pred_category',''):>20}] "
                  f"score={r.get('interest_score', 0):.2f} "
                  f"| {str(r.get('gt_title') or r.get('title',''))[:120]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
