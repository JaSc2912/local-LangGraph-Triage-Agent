from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st

from insurance_triage.config import Settings, resolve_profile
from insurance_triage.data import discover_dataset, load_tickets, write_json, write_results
from insurance_triage.runner import BatchRunner
from insurance_triage.schemas import TriageResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def results_frame(results: list[TriageResult]) -> pd.DataFrame:
    return pd.DataFrame(result.to_csv_row() for result in results)


def comparison_frame(comparison: dict[str, object]) -> pd.DataFrame:
    profiles = comparison.get("profiles", {})
    rows: list[dict[str, object]] = []
    if not isinstance(profiles, dict):
        return pd.DataFrame()
    for name, raw_summary in profiles.items():
        if not isinstance(raw_summary, dict):
            continue
        latency = raw_summary.get("latency_ms", {})
        latency = latency if isinstance(latency, dict) else {}
        ticket_count = int(raw_summary.get("ticket_count", 0) or 0)
        rows.append(
            {
                "profile": name,
                "model": raw_summary.get("model", ""),
                "tickets": ticket_count,
                "errors": int(raw_summary.get("error_count", 0) or 0),
                "review_rate": (
                    float(raw_summary.get("reviewed_count", 0) or 0) / ticket_count
                    if ticket_count
                    else 0.0
                ),
                "missing_info_rate": (
                    float(raw_summary.get("needs_more_information_count", 0) or 0) / ticket_count
                    if ticket_count
                    else 0.0
                ),
                "mean_latency_ms": int(latency.get("mean", 0) or 0),
                "p95_latency_ms": int(latency.get("p95", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def _dashboard_settings(*, language: str, limit: int, seed: int) -> Settings:
    base = Settings()
    return replace(
        base,
        input_dir=PROJECT_ROOT / "data" / "raw",
        output_dir=PROJECT_ROOT / "outputs",
        comparison_output=PROJECT_ROOT / "outputs" / "profile_comparison.json",
        language=language,
        limit=limit,
        seed=seed,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", base.ollama_base_url),
    )


def _ollama_status(base_url: str) -> tuple[bool, str, list[str]]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/version", timeout=1.5) as response:
            version_data = json.load(response)
        with urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=1.5) as response:
            process_data = json.load(response)
    except (OSError, URLError, TimeoutError, ValueError):
        return False, "nicht erreichbar", []

    models = [
        str(model.get("name", ""))
        for model in process_data.get("models", [])
        if isinstance(model, dict) and model.get("name")
    ]
    return True, str(version_data.get("version", "unbekannt")), models


def _render_metric_row(summary: dict[str, object]) -> None:
    latency = summary.get("latency_ms", {})
    latency = latency if isinstance(latency, dict) else {}
    columns = st.columns(5)
    columns[0].metric("Tickets", int(summary.get("ticket_count", 0) or 0))
    columns[1].metric("Erfolgreich", int(summary.get("successful_count", 0) or 0))
    columns[2].metric("Fehler", int(summary.get("error_count", 0) or 0))
    columns[3].metric("LLM-Reviews", int(summary.get("reviewed_count", 0) or 0))
    columns[4].metric("Ø Latenz", f"{int(latency.get('mean', 0) or 0):,} ms")


def _render_result_charts(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Noch keine Ergebnisse vorhanden.")
        return

    left, middle, right = st.columns(3)
    chart_specs = (
        (left, "Topics", "topic"),
        (middle, "Dringlichkeit", "urgency"),
        (right, "Nächste Aktion", "next_action"),
    )
    for container, title, column in chart_specs:
        with container:
            st.markdown(f"#### {title}")
            counts = frame[column].replace("", pd.NA).dropna().value_counts()
            st.bar_chart(counts, horizontal=True, height=260)


def _render_results(frame: pd.DataFrame, summary: dict[str, object]) -> None:
    _render_metric_row(summary)
    _render_result_charts(frame)
    st.markdown("#### Ticket-Ergebnisse")
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_order=(
            "ticket_id",
            "topic",
            "topic_confidence",
            "urgency",
            "urgency_confidence",
            "next_action",
            "needs_more_information",
            "reviewed_by_llm",
            "latency_ms",
            "text_snippet",
        ),
        column_config={
            "topic_confidence": st.column_config.ProgressColumn(
                "Topic confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "urgency_confidence": st.column_config.ProgressColumn(
                "Urgency confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "latency_ms": st.column_config.NumberColumn("Latency", format="%d ms"),
        },
    )


def _execute_live_run(
    *,
    profile_name: str,
    input_value: str,
    language: str,
    limit: int,
    seed: int,
) -> None:
    settings = _dashboard_settings(language=language, limit=limit, seed=seed)
    profile = resolve_profile(profile_name)
    input_path = Path(input_value).expanduser() if input_value.strip() else None

    progress = st.progress(0.0, text="Lauf wird vorbereitet …")
    recent_results = st.empty()
    status = st.status("Dataset und Modell werden geprüft …", expanded=True)
    try:
        dataset_path = discover_dataset(input_path, settings.input_dir)
        tickets = load_tickets(
            dataset_path,
            language=settings.language,
            limit=settings.limit,
            seed=settings.seed,
        )
        runner = BatchRunner(profile, settings)
        runner.preflight()
        status.write(f"{len(tickets)} Tickets aus `{dataset_path.name}` geladen.")
        status.write(f"Modell `{profile.model}` ist verfügbar.")

        completed: list[TriageResult] = []

        def update_progress(index: int, total: int, result: TriageResult) -> None:
            completed.append(result)
            progress.progress(index / total, text=f"Ticket {index} von {total}")
            recent_results.dataframe(
                results_frame(completed[-8:])[
                    [
                        "ticket_id",
                        "topic",
                        "urgency",
                        "next_action",
                        "processing_status",
                        "latency_ms",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        results = runner.run_tickets(tickets, progress_callback=update_progress)
        summary = {
            "profile": profile.name,
            "model": profile.model,
            "dataset": dataset_path.name,
            "language": settings.language,
            "limit": settings.limit,
            "seed": settings.seed,
            **runner.summarize(results),
        }
        output_path = settings.output_dir / f"dashboard_results_{profile.name}.csv"
        summary_path = settings.output_dir / f"dashboard_results_{profile.name}_summary.json"
        write_results(results, output_path)
        write_json(summary, summary_path)

        st.session_state["dashboard_results"] = results_frame(results)
        st.session_state["dashboard_summary"] = summary
        st.session_state["dashboard_output"] = str(output_path)
        status.update(
            label=f"Lauf abgeschlossen: {summary['successful_count']} erfolgreich",
            state="complete",
            expanded=False,
        )
        progress.progress(1.0, text="Fertig")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        status.update(label="Lauf konnte nicht gestartet werden", state="error")
        progress.empty()
        st.error(str(exc))


def _render_sample_output() -> None:
    result_path = PROJECT_ROOT / "outputs" / "triage_results_quality.csv"
    summary_path = PROJECT_ROOT / "outputs" / "triage_results_quality_summary.json"
    if not result_path.exists() or not summary_path.exists():
        st.info("Der publizierte Quality-Sample-Output ist noch nicht vorhanden.")
        return
    frame = pd.read_csv(result_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _render_results(frame, summary)


def _render_comparison() -> None:
    path = PROJECT_ROOT / "outputs" / "profile_comparison.json"
    if not path.exists():
        st.info("Noch kein Profilvergleich vorhanden. Starte zuerst `uv run triage-compare`.")
        return

    comparison = json.loads(path.read_text(encoding="utf-8"))
    agreement = comparison.get("agreement", {})
    agreement = agreement if isinstance(agreement, dict) else {}
    columns = st.columns(4)
    columns[0].metric("Vergleichbare Tickets", int(agreement.get("comparable_tickets", 0) or 0))
    columns[1].metric("Topic agreement", f"{float(agreement.get('topic_agreement', 0)):.0%}")
    columns[2].metric("Urgency agreement", f"{float(agreement.get('urgency_agreement', 0)):.0%}")
    columns[3].metric("Routing agreement", f"{float(agreement.get('action_agreement', 0)):.0%}")

    frame = comparison_frame(comparison)
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "review_rate": st.column_config.NumberColumn("Review rate", format="percent"),
            "missing_info_rate": st.column_config.NumberColumn(
                "Missing-info rate", format="percent"
            ),
            "mean_latency_ms": st.column_config.NumberColumn("Mean latency", format="%d ms"),
            "p95_latency_ms": st.column_config.NumberColumn("P95 latency", format="%d ms"),
        },
    )
    if not frame.empty:
        st.markdown("#### Laufzeitvergleich")
        st.bar_chart(
            frame.set_index("profile")[["mean_latency_ms", "p95_latency_ms"]],
            height=320,
        )


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Insurance Triage Control Room",
        page_icon="🛡️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7f6; }
        [data-testid="stSidebar"] { background: #0c2925; }
        [data-testid="stSidebar"] * { color: #eef9f5; }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            color: #ecfff8;
            background: linear-gradient(120deg, #0c2925 0%, #125447 70%, #167c66 100%);
            margin-bottom: 1rem;
            box-shadow: 0 14px 34px rgba(12, 41, 37, 0.16);
        }
        .hero small { color: #90e5c8; font-weight: 700; letter-spacing: .12em; }
        .hero h1 { margin: .35rem 0 .2rem; font-size: 2.25rem; }
        .hero p { margin: 0; color: #d8eee7; max-width: 760px; }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #dce9e5;
            padding: .85rem 1rem;
            border-radius: 14px;
        }
        </style>
        <div class="hero">
          <small>LOCAL · LANGGRAPH · OLLAMA</small>
          <h1>Insurance Triage Control Room</h1>
          <p>Lokale Ticket-Triage beobachten, Modellprofile vergleichen und
          Entscheidungen bis zum deterministischen Routing nachvollziehen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base = Settings()
    base_url = os.getenv("OLLAMA_BASE_URL", base.ollama_base_url)
    online, version, loaded_models = _ollama_status(base_url)
    with st.sidebar:
        st.markdown("## Neuer Lauf")
        if online:
            st.success(f"Ollama {version} erreichbar")
            if loaded_models:
                st.caption("Im Speicher: " + ", ".join(loaded_models))
            else:
                st.caption("Aktuell ist kein Modell geladen.")
        else:
            st.warning(f"Ollama unter {base_url} nicht erreichbar")

        with st.form("triage_run"):
            profile_name = st.selectbox(
                "Modellprofil",
                ("quality", "compact"),
                help="Quality nutzt die RTX-3090-Variante; Compact benötigt weniger VRAM.",
            )
            input_value = st.text_input(
                "Dataset-Pfad (optional)",
                placeholder="Automatische Suche in data/raw/",
            )
            language = st.text_input("Sprache", value="en")
            limit = st.number_input("Tickets", min_value=1, max_value=1000, value=25)
            seed = st.number_input("Seed", min_value=0, value=42)
            submitted = st.form_submit_button(
                "Live-Triage starten",
                type="primary",
                use_container_width=True,
            )

        st.caption(
            "Alle Ticketdaten bleiben lokal. Dashboard-Läufe werden unter "
            "`outputs/dashboard_results_<profile>.csv` gespeichert."
        )

    live_tab, sample_tab, comparison_tab = st.tabs(
        ("Live-Triage", "Quality-Sample", "Profilvergleich")
    )
    with live_tab:
        if submitted:
            _execute_live_run(
                profile_name=profile_name,
                input_value=input_value,
                language=language,
                limit=int(limit),
                seed=int(seed),
            )
        if "dashboard_results" in st.session_state:
            st.markdown("### Letzter Dashboard-Lauf")
            st.caption(f"Gespeichert unter `{st.session_state['dashboard_output']}`")
            _render_results(
                st.session_state["dashboard_results"],
                st.session_state["dashboard_summary"],
            )
        elif not submitted:
            st.info(
                "Wähle links ein Profil und starte die Triage. Für einen schnellen "
                "Funktionstest sind 5–10 Tickets ausreichend."
            )

    with sample_tab:
        st.markdown("### Publizierter Quality-Lauf")
        _render_sample_output()

    with comparison_tab:
        st.markdown("### Quality vs. Compact")
        _render_comparison()


def main() -> None:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print("Usage: triage-dashboard\n\nStart the local Streamlit dashboard.")
        return
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    render_dashboard()
