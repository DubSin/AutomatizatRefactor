const { useState, useEffect, useMemo, useRef, Fragment } = React;

const FIELD_KEYS = [
    "number", "title", "status", "start_price", "end_date",
    "delivery_place", "customer", "publish_date", "positions",
    "predicted_category", "interest_score", "interest_reasoning",
    "deep_rag_decision", "deep_rag_reasoning",
];

const HEADER_NAMES = {
    number: "Номер тендера",
    title: "Название",
    status: "Статус",
    start_price: "Начальная цена",
    end_date: "Окончание",
    delivery_place: "Место поставки",
    customer: "Заказчик",
    publish_date: "Дата публикации",
    positions: "Позиции",
    predicted_category: "Категория (ИИ)",
    interest_score: "Оценка интереса",
    interest_reasoning: "Пояснение интереса",
    deep_rag_decision: "Deep RAG: Решение",
    deep_rag_reasoning: "Deep RAG: Пояснение",
};

const MAIN_FIELDS = new Set([
    "number", "title", "status", "start_price", "end_date",
    "delivery_place", "customer", "publish_date", "positions",
    "predicted_category", "deep_rag_decision",
]);

const MAIN_FILTER_OPTIONS = [
    ["all", "Все"],
    ["interesting", "Интересные"],
    ["not-interesting", "Не интересные"],
    ["deeprag", "Подходящие (Глубокий анализ)"],
];

function parseRussianDate(s) {
    if (!s) return new Date(0);
    const [datePart, timePart = "00:00:00"] = String(s).split(" ");
    const [d, m, y] = datePart.split(".");
    if (!d || !m || !y) return new Date(0);
    const year = parseInt(y, 10);
    const fullYear = year < 100 ? 2000 + year : year;
    const [hh = 0, mm = 0, ss = 0] = timePart.split(":").map(x => parseInt(x, 10) || 0);
    return new Date(fullYear, parseInt(m, 10) - 1, parseInt(d, 10), hh, mm, ss);
}

function formatPositions(positions) {
    if (!Array.isArray(positions)) return "";
    return positions.map(p => {
        const name = (p && p.name) || "";
        const price = (p && p.price) || "";
        const qty = (p && p.quantity) || "";
        const unit = (p && p.unit) || "";
        const qs = unit ? `${qty} ${unit}` : qty;
        return price ? `${name} - ${price} - ${qs}` : `${name} - ${qs}`;
    }).join("\n");
}

function isInteresting(rec) {
    return rec.is_interesting === true || rec.is_interesting === "true" || rec.is_interesting === "Да";
}

const XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

// Кроссбраузерное скачивание xlsx: пробуем встроенный XLSX.writeFile,
// потом msSaveBlob (старый Edge/IE), потом ручной Blob + <a download>.
function saveExcelFile(wb, filename) {
    const XLSX = window.XLSX;
    if (!XLSX) throw new Error("XLSX library is not loaded");

    if (typeof XLSX.writeFile === "function") {
        try {
            XLSX.writeFile(wb, filename, { bookType: "xlsx", cellStyles: true });
            return;
        } catch (e) {
            console.warn("XLSX.writeFile failed, falling back to manual download:", e);
        }
    }

    const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array", cellStyles: true });
    const blob = new Blob([wbout], { type: XLSX_MIME });

    if (typeof navigator !== "undefined" && navigator.msSaveBlob) {
        navigator.msSaveBlob(blob, filename);
        return;
    }

    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filename;
    a.rel = "noopener";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(blobUrl);
    }, 1000);
}

function HighlightedText({ text, query }) {
    const s = String(text == null ? "" : text);
    if (!query) return <Fragment>{s}</Fragment>;
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(${escaped})`, "gi");
    const parts = s.split(re);
    return (
        <Fragment>
            {parts.map((p, i) =>
                i % 2 === 1
                    ? <mark key={i} className="search-hl">{p}</mark>
                    : <Fragment key={i}>{p}</Fragment>
            )}
        </Fragment>
    );
}

function getCellClass(key) {
    switch (key) {
        case "number": return "col-number";
        case "title": return "col-title";
        case "customer": return "col-customer";
        case "delivery_place": return "col-place";
        case "positions": return "col-positions";
        case "interest_reasoning":
        case "deep_rag_reasoning": return "col-reasoning";
        default: return "";
    }
}

function renderCellContent(rec, key, search) {
    const value = rec[key];

    if (key === "number") {
        const num = String(value == null ? "" : value);
        const url = rec.url;
        const text = <HighlightedText text={num} query={search.number} />;
        return url
            ? <a href={url} target="_blank" rel="noopener noreferrer" title={num}>{text}</a>
            : text;
    }
    if (key === "title") {
        const txt = String(value == null ? "" : value);
        const trimmed = txt.length > 500 ? txt.slice(0, 500) + "…" : txt;
        return <HighlightedText text={trimmed} query={search.title} />;
    }
    if (key === "customer") {
        const txt = String(value == null ? "" : value);
        const trimmed = txt.length > 200 ? txt.slice(0, 200) + "…" : txt;
        return <HighlightedText text={trimmed} query={search.customer} />;
    }
    if (key === "delivery_place") {
        const txt = String(value == null ? "" : value);
        return txt.length > 200 ? txt.slice(0, 200) + "…" : txt;
    }
    if (key === "positions") {
        const formatted = formatPositions(value);
        const trimmed = formatted.length > 500 ? formatted.slice(0, 500) + "…" : formatted;
        return <span style={{ whiteSpace: "pre-wrap" }}>{trimmed}</span>;
    }
    if (key === "start_price") {
        return <strong>{String(value == null ? "" : value)}</strong>;
    }
    if (key === "status") {
        const s = String(value == null ? "" : value);
        const lower = s.toLowerCase();
        let cls = "";
        if (lower.includes("прием заявок") || lower.includes("приём заявок")) cls = "status-active";
        else if (lower.includes("завершен") || lower.includes("завершён")) cls = "status-completed";
        else if (lower.includes("отменен") || lower.includes("отменён")) cls = "status-cancelled";
        return <span className={cls}>{s}</span>;
    }
    if (key === "end_date" || key === "publish_date") {
        return <span className="date">{String(value == null ? "" : value)}</span>;
    }
    if (key === "predicted_category" || key === "deep_rag_decision") {
        return <strong>{String(value == null ? "" : value)}</strong>;
    }
    if (key === "interest_reasoning" || key === "deep_rag_reasoning") {
        const txt = String(value == null ? "" : value);
        return txt || "—";
    }
    if (typeof value === "object" && value !== null) {
        const json = JSON.stringify(value, null, 2);
        return json.length > 200 ? json.slice(0, 200) + "…" : json;
    }
    const txt = String(value == null ? "" : value);
    return txt.length > 200 ? txt.slice(0, 200) + "…" : txt;
}

function ReportDropdown({ reports, activeId, onSelect }) {
    const [open, setOpen] = useState(false);
    const dropdownRef = useRef(null);

    useEffect(() => {
        function handleClickOutside(event) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setOpen(false);
            }
        }
        if (open) {
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [open]);

    if (!reports.length) {
        return <div className="no-reports">Нет отчётов в storage/tenders/jsonl</div>;
    }

    const activeReport = reports.find(r => r.id === activeId);
    const label = activeReport ? activeReport.label : "Выберите отчёт";

    return (
        <div className="dropdown-container" ref={dropdownRef}>
            <button
                className="dropdown-btn"
                onClick={() => setOpen(!open)}
            >
                {label} <span className="caret">▼</span>
            </button>
            {open && (
                <ul className="dropdown-menu">
                    {reports.map(r => (
                        <li
                            key={r.id}
                            className={`dropdown-item${r.id === activeId ? " active" : ""}`}
                            onClick={() => { onSelect(r.id); setOpen(false); }}
                        >
                            {r.label}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

function FiltersBar(props) {
    const {
        mainFilter, setMainFilter,
        category, setCategory, categories,
        columnsMode, setColumnsMode,
        onReset,
        search, setSearch,
        totalCount, visibleCount,
        onExport,
    } = props;

    const labels = {
        all: null,
        interesting: "Интересные",
        "not-interesting": "Не интересные",
        deeprag: "Подходящие (Глубокий анализ)",
    };
    const active = [];
    if (labels[mainFilter]) active.push(labels[mainFilter]);
    if (category !== "all") active.push(`Категория: ${category}`);
    if (search.number) active.push(`Номер: «${search.number}»`);
    if (search.title) active.push(`Название: «${search.title}»`);
    if (search.customer) active.push(`Заказчик: «${search.customer}»`);

    return (
        <section className="filters">
            <div className="filter-row">
                <div className="filter-group">
                    <span className="filter-label">Фильтр:</span>
                    {MAIN_FILTER_OPTIONS.map(([v, l]) => (
                        <button
                            key={v}
                            className={`filter-btn${mainFilter === v ? " active" : ""}`}
                            onClick={() => setMainFilter(v)}
                        >{l}</button>
                    ))}
                </div>
                <div className="filter-group">
                    <span className="filter-label">Категория:</span>
                    <select value={category} onChange={e => setCategory(e.target.value)}>
                        <option value="all">Все категории</option>
                        {categories.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                </div>
                <div className="filter-group">
                    <span className="filter-label">Колонки:</span>
                    <button
                        className={`filter-btn${columnsMode === "main" ? " active" : ""}`}
                        onClick={() => setColumnsMode("main")}
                    >Основные поля</button>
                    <button
                        className={`filter-btn${columnsMode === "all" ? " active" : ""}`}
                        onClick={() => setColumnsMode("all")}
                    >Все поля</button>
                </div>
                <button className="filter-btn" onClick={onReset}>Сбросить фильтры</button>
                <button className="filter-btn" onClick={onExport}>📊 Скачать Excel</button>
                <div className="filter-group counter">
                    <span className="filter-label">Показано:</span>
                    <span>{visibleCount}</span> из <span>{totalCount}</span>
                </div>
            </div>

            <div className="filter-row search-row">
                <span className="filter-label">🔎 Поиск:</span>
                <input
                    type="text"
                    placeholder="Номер тендера"
                    value={search.number}
                    onChange={e => setSearch({ ...search, number: e.target.value })}
                />
                <input
                    type="text"
                    placeholder="Ключевые слова в названии"
                    value={search.title}
                    onChange={e => setSearch({ ...search, title: e.target.value })}
                />
                <input
                    type="text"
                    placeholder="Заказчик"
                    value={search.customer}
                    onChange={e => setSearch({ ...search, customer: e.target.value })}
                />
                <button
                    className="filter-btn"
                    onClick={() => setSearch({ number: "", title: "", customer: "" })}
                >Очистить поиск</button>
            </div>

            <div className="filters-info">
                {active.length === 0
                    ? "✅ Фильтры не применены. Показаны все записи."
                    : `🔍 Активные фильтры: ${active.join(" | ")}`}
            </div>
        </section>
    );
}

function TenderTable({ records, columnsMode, sortColumn, sortDirection, onSort, search }) {
    const sortArrow = (idx) =>
        sortColumn === idx ? (sortDirection > 0 ? " ↑" : " ↓") : " ↕";

    return (
        <table>
            <thead>
                <tr>
                    <th style={{ textAlign: "center" }} onClick={() => onSort(0)}>
                        №<span className="sort-arrow">{sortArrow(0)}</span>
                    </th>
                    {FIELD_KEYS.map((key, i) => {
                        const colIdx = i + 1;
                        if (columnsMode === "main" && !MAIN_FIELDS.has(key)) return null;
                        return (
                            <th key={key} title={key} onClick={() => onSort(colIdx)}>
                                {HEADER_NAMES[key] || key}
                                <span className="sort-arrow">{sortArrow(colIdx)}</span>
                            </th>
                        );
                    })}
                </tr>
            </thead>
            <tbody>
                {records.map((rec, idx) => {
                    const interest = isInteresting(rec);
                    const decisionLower = String(rec.deep_rag_decision || "").toLowerCase();
                    let rowClass = "";
                    if (interest) rowClass = "row-interesting";
                    else if (decisionLower.includes("не подходит")) rowClass = "row-not-suitable";

                    return (
                        <tr key={`${rec.number || idx}-${idx}`} className={rowClass}>
                            <td style={{ textAlign: "center", fontWeight: "bold" }}>{idx + 1}</td>
                            {FIELD_KEYS.map(key => {
                                if (columnsMode === "main" && !MAIN_FIELDS.has(key)) return null;
                                return (
                                    <td key={key} className={getCellClass(key)}>
                                        {renderCellContent(rec, key, search)}
                                    </td>
                                );
                            })}
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

function App() {
    const [reports, setReports] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [data, setData] = useState({ records: [], categories: [] });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const [mainFilter, setMainFilter] = useState("all");
    const [category, setCategory] = useState("all");
    const [columnsMode, setColumnsMode] = useState("main");
    const [search, setSearch] = useState({ number: "", title: "", customer: "" });
    const [sortColumn, setSortColumn] = useState(-1);
    const [sortDirection, setSortDirection] = useState(1);

    useEffect(() => {
        fetch("/api/reports")
            .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
            .then(rs => {
                setReports(rs);
                if (rs.length > 0) setActiveId(rs[0].id);
            })
            .catch(e => setError(String(e)));
    }, []);

    useEffect(() => {
        if (!activeId) return;
        let cancelled = false;
        setLoading(true);
        setError(null);
        fetch(`/api/reports/${encodeURIComponent(activeId)}`)
            .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
            .then(d => {
                if (cancelled) return;
                setData(d);
                setSortColumn(-1);
                setSortDirection(1);
                setCategory(prev => (d.categories || []).includes(prev) ? prev : "all");
            })
            .catch(e => { if (!cancelled) setError(String(e)); })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, [activeId]);

    const filteredRecords = useMemo(() => {
        const q = {
            number: search.number.trim().toLowerCase(),
            title: search.title.trim().toLowerCase(),
            customer: search.customer.trim().toLowerCase(),
        };
        return data.records.filter(rec => {
            const interest = isInteresting(rec);
            if (mainFilter === "interesting" && !interest) return false;
            if (mainFilter === "not-interesting" && interest) return false;
            if (mainFilter === "deeprag") {
                const dec = String(rec.deep_rag_decision || "").toLowerCase();
                const ok = (dec.includes("подходит") && !dec.includes("не подходит")) || dec.includes("на грани");
                if (!ok) return false;
            }
            const cat = String(rec.predicted_category || "").trim() || "Не классифицировано";
            if (category !== "all" && cat !== category) return false;

            if (q.number && !String(rec.number || "").toLowerCase().includes(q.number)) return false;
            if (q.title && !String(rec.title || "").toLowerCase().includes(q.title)) return false;
            if (q.customer && !String(rec.customer || "").toLowerCase().includes(q.customer)) return false;

            return true;
        });
    }, [data.records, mainFilter, category, search]);

    const sortedRecords = useMemo(() => {
        if (sortColumn < 0) return filteredRecords;
        const sortKey = sortColumn === 0 ? null : FIELD_KEYS[sortColumn - 1];
        const dir = sortDirection;
        const arr = filteredRecords.slice();
        arr.sort((a, b) => {
            if (sortKey === null) return 0;
            let av = a[sortKey], bv = b[sortKey];
            if (sortKey === "start_price") {
                const na = parseFloat(String(av || "").replace(/[^\d.,-]/g, "").replace(",", "."));
                const nb = parseFloat(String(bv || "").replace(/[^\d.,-]/g, "").replace(",", "."));
                av = isNaN(na) ? -Infinity : na;
                bv = isNaN(nb) ? -Infinity : nb;
            } else if (sortKey === "end_date" || sortKey === "publish_date") {
                av = parseRussianDate(av);
                bv = parseRussianDate(bv);
            } else if (sortKey === "interest_score") {
                av = Number(av) || 0;
                bv = Number(bv) || 0;
            } else {
                av = String(av == null ? "" : av).toLowerCase();
                bv = String(bv == null ? "" : bv).toLowerCase();
            }
            if (av < bv) return -1 * dir;
            if (av > bv) return 1 * dir;
            return 0;
        });
        return arr;
    }, [filteredRecords, sortColumn, sortDirection]);

    const onSort = (col) => {
        if (sortColumn === col) setSortDirection(d => -d);
        else { setSortColumn(col); setSortDirection(1); }
    };

    const onResetFilters = () => {
        setMainFilter("all");
        setCategory("all");
    };

    // --- Экспорт в Excel -------------------------------------------------
    const handleExportExcel = () => {
        try {
            const XLSX = window.XLSX;
            if (!XLSX) {
                alert("Библиотека XLSX не загружена. Проверьте подключение к интернету.");
                return;
            }

            const exportFieldKeys = FIELD_KEYS.filter(key =>
                columnsMode === "main" ? MAIN_FIELDS.has(key) : true
            );

            const headerRow = ["№", ...exportFieldKeys.map(key => HEADER_NAMES[key] || key)];
            const MAX_CELL_LEN = 32767;
            const hyperlinkCells = [];

            const dataRows = sortedRecords.map((rec, idx) => {
                const row = [idx + 1];
                exportFieldKeys.forEach((fieldKey, fIdx) => {
                    let val = rec[fieldKey];
                    let cellValue;

                    if (fieldKey === "number") {
                        cellValue = String(val == null ? "" : val);
                        const url = rec.url || "";
                        if (url) {
                            // +1 для № в начале; +1 для пропуска заголовка (он добавится в общий sheetData)
                            hyperlinkCells.push({
                                colIdx: fIdx + 1,
                                rowIdx: idx + 1,
                                url,
                            });
                        }
                    } else if (fieldKey === "positions" && Array.isArray(val)) {
                        cellValue = formatPositions(val);
                    } else if (val === null || val === undefined) {
                        cellValue = "";
                    } else if (typeof val === "object") {
                        cellValue = JSON.stringify(val, null, 2);
                    } else {
                        cellValue = String(val);
                    }

                    if (typeof cellValue === "string" && cellValue.length > MAX_CELL_LEN) {
                        cellValue = cellValue.slice(0, MAX_CELL_LEN);
                    }
                    row.push(cellValue);
                });
                return row;
            });

            const sheetData = [headerRow, ...dataRows];
            const ws = XLSX.utils.aoa_to_sheet(sheetData);

            // Гиперссылки + синий цвет + подчёркивание для номеров тендеров
            hyperlinkCells.forEach(({ rowIdx, colIdx, url }) => {
                const cellRef = XLSX.utils.encode_cell({ r: rowIdx, c: colIdx });
                if (ws[cellRef]) {
                    ws[cellRef].l = { Target: url };
                    ws[cellRef].s = {
                        font: { color: { rgb: "0000FF" }, underline: true },
                    };
                }
            });

            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, "Тендеры");

            const filename = `tenders_filtered_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.xlsx`;
            saveExcelFile(wb, filename);
        } catch (e) {
            alert("Ошибка при создании Excel-файла: " + e.message);
            console.error("handleExportExcel error:", e);
        }
    };

    return (
        <div className="app">
            <header><h1>📊 Сводная информация по тендерам</h1></header>

            <ReportDropdown reports={reports} activeId={activeId} onSelect={setActiveId} />

            <FiltersBar
                mainFilter={mainFilter}
                setMainFilter={setMainFilter}
                category={category}
                setCategory={setCategory}
                categories={data.categories || []}
                columnsMode={columnsMode}
                setColumnsMode={setColumnsMode}
                onReset={onResetFilters}
                search={search}
                setSearch={setSearch}
                totalCount={data.records.length}
                visibleCount={sortedRecords.length}
                onExport={handleExportExcel}
            />

            <main className="table-container">
                {error && <div className="loading-msg">Ошибка: {error}</div>}
                {loading && !error && <div className="loading-msg">Загрузка отчёта…</div>}
                {!loading && !error && reports.length === 0 && (
                    <div className="loading-msg">Нет отчётов в storage/tenders/jsonl</div>
                )}
                {!loading && !error && reports.length > 0 && data.records.length === 0 && activeId && (
                    <div className="loading-msg">В отчёте нет записей</div>
                )}
                {!loading && !error && data.records.length > 0 && (
                    <TenderTable
                        records={sortedRecords}
                        columnsMode={columnsMode}
                        sortColumn={sortColumn}
                        sortDirection={sortDirection}
                        onSort={onSort}
                        search={search}
                    />
                )}
            </main>

            <footer className="footer">Tender Web · {reports.length} отчёт(ов)</footer>
        </div>
    );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);