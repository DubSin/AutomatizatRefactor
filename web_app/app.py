"""
Flask backend для React SPA по тендерам.
Отдаёт статический фронт + JSON API из storage/tenders/jsonl.
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, abort, send_from_directory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
JSONL_DIR = Path(os.environ.get("TENDERS_JSONL_DIR") or (PROJECT_ROOT / "storage" / "tenders" / "jsonl"))

FILENAME_RE = re.compile(r"all_tenders_data_(\d{8})_(\d{6})\.jsonl$")

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)


def _format_label(stem_date: str, stem_time: str) -> str:
    try:
        dt = datetime.strptime(stem_date + stem_time, "%Y%m%d%H%M%S")
        return f"Отчет от {dt.strftime('%d.%m.%Y %H:%M')}"
    except ValueError:
        return f"Отчет от {stem_date}"


def _list_reports() -> list[dict[str, str]]:
    if not JSONL_DIR.is_dir():
        logger.warning("Папка с jsonl не найдена: %s", JSONL_DIR)
        return []
    reports = []
    for path in JSONL_DIR.iterdir():
        if not path.is_file() or path.suffix != ".jsonl":
            continue
        m = FILENAME_RE.search(path.name)
        if not m:
            continue
        d, t = m.group(1), m.group(2)
        reports.append({
            "id": path.stem,
            "filename": path.name,
            "label": _format_label(d, t),
            "_sk": d + t,
        })
    reports.sort(key=lambda r: r["_sk"], reverse=True)
    for r in reports:
        r.pop("_sk", None)
    return reports


def _load_jsonl(filename: str) -> list[dict[str, Any]]:
    if not FILENAME_RE.search(filename):
        abort(400, description="Некорректное имя файла")
    path = JSONL_DIR / filename
    if not path.is_file():
        abort(404, description="Файл отчёта не найден")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    records.append(data)
            except json.JSONDecodeError as e:
                logger.warning("Ошибка JSON в %s:%d — %s", filename, line_num, e)
    return records


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/reports")
def api_reports():
    return jsonify(_list_reports())


@app.route("/api/reports/<report_id>")
def api_report_data(report_id: str):
    filename = f"{report_id}.jsonl"
    records = _load_jsonl(filename)

    categories = sorted({
        (str(r.get("predicted_category") or "").strip() or "Не классифицировано")
        for r in records
    })

    return jsonify({
        "id": report_id,
        "count": len(records),
        "categories": categories,
        "records": records,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
