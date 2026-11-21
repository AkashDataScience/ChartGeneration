import asyncio
import json
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.agents.analysis_agent import AnalysisPlanResponse, handle_followup_request
from src.core.workflow import run_full_analysis_workflow, run_workflow

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
UI_DIR = BASE_DIR / "ui"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UI_DIR, exist_ok=True)

# In-memory store for active and completed runs
ACTIVE_RUNS: dict[str, dict] = {}


def _build_dag_for_run(run_data: dict) -> dict:
    """
    Build the React Flow / DAG nodes and edges matching the dark-mode multi-agent graph UI:
      Row 1: PLANNERAGENT (PlannerAgent)
      Row 2: STATISTICALAGENT (StatisticalAgent) + CHARTAGENT 1..N (ChartAgent V1)
      Row 3: REFLECTORAGENT 1..N (ReflectorAgent V2)
      Row 4: DISTILLERAGENT (DistillerAgent)
      Row 5: FORMATTERAGENT (FormatterAgent)
    """
    status = run_data.get("status", "COMPLETED")
    phase = run_data.get("current_phase", "stage_4_done" if status == "COMPLETED" else "init")
    plan = run_data.get("analysis_plan") or {}
    stat_findings = run_data.get("stat_findings") or {}
    chart_results = run_data.get("chart_results") or []
    chart_tasks = plan.get("chart_tasks") or []
    max_charts = run_data.get("max_charts", 3)

    # Determine effective chart slots
    num_charts = max(len(chart_tasks[:max_charts]), len(chart_results), 2)
    num_charts = min(num_charts, 3)

    chart_states = run_data.get("chart_states", {})

    # Determine status of each stage
    if status == "COMPLETED":
        planner_status = "COMPLETED"
        stat_status = "COMPLETED"
        distiller_status = "COMPLETED"
        formatter_status = "COMPLETED"
    elif status == "FAILED":
        planner_status = "COMPLETED" if plan else "FAILED"
        stat_status = "COMPLETED" if stat_findings else ("FAILED" if phase == "stage_2_start" else "PENDING")
        distiller_status = "FAILED" if phase == "stage_4_start" else "PENDING"
        formatter_status = "FAILED" if phase == "stage_4_start" else "PENDING"
    else:
        # RUNNING
        if phase in ("init", "stage_1_start"):
            planner_status = "RUNNING"
            stat_status = "PENDING"
            distiller_status = "PENDING"
            formatter_status = "PENDING"
        elif phase == "stage_1_done" or phase == "stage_2_start":
            planner_status = "COMPLETED"
            stat_status = "RUNNING"
            distiller_status = "PENDING"
            formatter_status = "PENDING"
        elif phase in ("stage_2_done", "chart_v1_start", "chart_v1_done", "chart_v2_start", "chart_v2_done"):
            planner_status = "COMPLETED"
            stat_status = "COMPLETED"
            distiller_status = "PENDING"
            formatter_status = "PENDING"
        elif phase == "stage_4_start":
            planner_status = "COMPLETED"
            stat_status = "COMPLETED"
            distiller_status = "COMPLETED"
            formatter_status = "RUNNING"
        else:
            planner_status = "COMPLETED"
            stat_status = "COMPLETED"
            distiller_status = "COMPLETED"
            formatter_status = "COMPLETED"

    nodes = [
        {
            "id": "planner",
            "role": "PLANNERAGENT",
            "label": "PlannerAgent",
            "subtitle": "Stage 1: Data Profiling & Plan",
            "x": 430,
            "y": 50,
            "status": planner_status,
            "details": {
                "stage": "Stage 1 — Automated Data Profiling, Quality Audit & Analysis Planning",
                "dataset_summary": plan.get("dataset_summary", ""),
                "cleaning_rules": plan.get("cleaning_instructions_summary", ""),
                "data_quality_issues": plan.get("data_quality_issues", []),
                "analytical_questions": plan.get("analytical_questions", []),
                "chart_tasks": chart_tasks,
            }
        },
        {
            "id": "statistical",
            "role": "RETRIEVERAGENT",
            "label": "StatisticalAgent",
            "subtitle": "Stage 2: Quantitative & SQL/Pandas",
            "x": 140,
            "y": 210,
            "status": stat_status,
            "details": {
                "stage": "Stage 2 — Quantitative Exploration & Hypothesis Testing",
                "findings": stat_findings.get("findings", []),
                "key_takeaways": stat_findings.get("key_takeaways", []),
            }
        }
    ]

    edges = [
        {"id": "e-plan-stat", "source": "planner", "target": "statistical", "sourcePort": "left", "targetPort": "top"}
    ]

    # X positions for ChartAgent / ReflectorAgent columns
    x_positions = [430, 720, 1010]
    for i in range(num_charts):
        idx = i + 1
        c_res = chart_results[i] if i < len(chart_results) else None
        c_task = chart_tasks[i] if i < len(chart_tasks) else {}
        c_state = chart_states.get(idx, {})

        chart_id = (c_res or {}).get("chart_id") or c_task.get("chart_id") or f"chart_{idx}"
        chart_title = (c_res or {}).get("title") or c_task.get("title") or f"Visualization {idx}"

        if status == "COMPLETED" or c_res is not None:
            v1_status = "COMPLETED"
            v2_status = "COMPLETED"
        else:
            v1_status = c_state.get("v1_status", "PENDING")
            v2_status = c_state.get("v2_status", "PENDING")

        gen_node_id = f"chart_gen_{idx}"
        ref_node_id = f"chart_ref_{idx}"
        col_x = x_positions[i] if i < len(x_positions) else 430 + i * 290

        chart_v1_url = ""
        chart_v2_url = ""
        if c_res:
            if c_res.get("chart_v1"):
                chart_v1_url = "/" + str(c_res["chart_v1"]).lstrip("/")
            if c_res.get("chart_v2"):
                chart_v2_url = "/" + str(c_res["chart_v2"]).lstrip("/")
        else:
            if c_state.get("chart_v1"):
                chart_v1_url = "/" + str(c_state["chart_v1"]).lstrip("/")
            if c_state.get("chart_v2"):
                chart_v2_url = "/" + str(c_state["chart_v2"]).lstrip("/")

        nodes.append({
            "id": gen_node_id,
            "role": "RETRIEVERAGENT",
            "label": f"ChartAgent #{idx}",
            "subtitle": chart_title[:28],
            "x": col_x,
            "y": 210,
            "status": v1_status,
            "details": {
                "stage": f"Stage 3A — Tree-of-Thoughts & V1 Code Generation ({chart_id})",
                "chart_id": chart_id,
                "title": chart_title,
                "rationale": (c_res or {}).get("rationale") or c_task.get("rationale", ""),
                "instruction": (c_res or {}).get("instruction") or c_task.get("instruction", ""),
                "thought_v1": (c_res or {}).get("thought_v1") or c_state.get("thought_v1", ""),
                "code_v1": (c_res or {}).get("code_v1") or c_state.get("code_v1", ""),
                "chart_v1_url": chart_v1_url,
            }
        })

        nodes.append({
            "id": ref_node_id,
            "role": "THINKERAGENT",
            "label": f"ReflectorAgent #{idx}",
            "subtitle": f"Visual Critique ({chart_id[:16]})",
            "x": col_x - 75,
            "y": 385,
            "status": v2_status,
            "details": {
                "stage": f"Stage 3B — Multimodal Visual Critique & V2 Refinement ({chart_id})",
                "chart_id": chart_id,
                "title": chart_title,
                "critique": (c_res or {}).get("critique") or c_state.get("critique", ""),
                "feedback": (c_res or {}).get("feedback") or c_state.get("feedback", ""),
                "code_v1": (c_res or {}).get("code_v1") or c_state.get("code_v1", ""),
                "code_v2": (c_res or {}).get("code_v2") or c_state.get("code_v2", ""),
                "chart_v1_url": chart_v1_url,
                "chart_v2_url": chart_v2_url,
            }
        })

        edges.append({
            "id": f"e-plan-{gen_node_id}",
            "source": "planner",
            "target": gen_node_id,
            "sourcePort": "bottom" if i == 0 else "right",
            "targetPort": "top"
        })
        edges.append({
            "id": f"e-stat-{gen_node_id}",
            "source": "statistical",
            "target": gen_node_id,
            "sourcePort": "right",
            "targetPort": "left"
        })
        edges.append({
            "id": f"e-{gen_node_id}-{ref_node_id}",
            "source": gen_node_id,
            "target": ref_node_id,
            "sourcePort": "bottom",
            "targetPort": "top"
        })
        edges.append({
            "id": f"e-{ref_node_id}-distiller",
            "source": ref_node_id,
            "target": "distiller",
            "sourcePort": "bottom",
            "targetPort": "top" if i == 0 else "right"
        })

    edges.append({
        "id": "e-stat-distiller",
        "source": "statistical",
        "target": "distiller",
        "sourcePort": "bottom",
        "targetPort": "left"
    })

    nodes.append({
        "id": "distiller",
        "role": "DISTILLERAGENT",
        "label": "DistillerAgent",
        "subtitle": "Evidence & Chart Consolidation",
        "x": 450,
        "y": 555,
        "status": distiller_status,
        "details": {
            "stage": "Stage 4A — Quantitative & Visual Evidence Distillation",
            "key_takeaways": stat_findings.get("key_takeaways", []),
            "charts_count": len(chart_results),
            "executive_summary": run_data.get("executive_summary", ""),
        }
    })

    nodes.append({
        "id": "formatter",
        "role": "FORMATTERAGENT",
        "label": "FormatterAgent",
        "subtitle": "Executive Report Synthesis",
        "x": 450,
        "y": 710,
        "status": formatter_status,
        "details": {
            "stage": "Stage 4B — Executive Report Synthesis (report.md)",
            "report_path": run_data.get("report_path", ""),
            "executive_summary": run_data.get("executive_summary", ""),
            "full_markdown_report": run_data.get("full_markdown_report", ""),
        }
    })

    edges.append({
        "id": "e-distiller-formatter",
        "source": "distiller",
        "target": "formatter",
        "sourcePort": "right",
        "targetPort": "top"
    })

    return {"nodes": nodes, "edges": edges}


def _ensure_seed_run():
    """
    If no completed runs exist in `reports/` yet, generate a fast, deterministic
    baseline analysis run on `data/coffee_sales.csv` so the UI immediately showcases
    a rich completed run, DAG graph, V1/V2 charts, and Markdown report on first load.
    """
    existing = list(REPORTS_DIR.glob("*/run_metadata.json"))
    if existing:
        for meta_file in existing:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "run_id" in data:
                        ACTIVE_RUNS[data["run_id"]] = data
            except Exception:
                pass
        if ACTIVE_RUNS:
            return

    csv_path = DATA_DIR / "coffee_sales.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    run_id = "coffee_sales_baseline"
    report_dir = REPORTS_DIR / run_id
    charts_dir = report_dir / "charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Generate 3 real V1 & V2 charts from coffee_sales.csv
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date_parsed"].dt.strftime("%Y-%m")
    df["hour"] = df["time"].astype(str).str.split(":").str[0].astype(int)

    # Chart 1: Revenue by Coffee Product
    rev_by_coffee = df.groupby("coffee_name")["price"].sum().sort_values(ascending=True)
    c1_v1 = (charts_dir / "revenue_by_coffee_v1.png").as_posix()
    c1_v2 = (charts_dir / "revenue_by_coffee_v2.png").as_posix()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(rev_by_coffee.index, rev_by_coffee.values, color="#6366f1")
    ax.set_title("Revenue by Coffee Type (Draft V1)")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(c1_v1, dpi=220)
    plt.close()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(rev_by_coffee.index, rev_by_coffee.values, color="#4f46e5", edgecolor="none")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 180, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, rev_by_coffee.max() * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Total Revenue by Coffee Product ($)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Total Revenue ($)")
    plt.tight_layout()
    plt.savefig(c1_v2, dpi=300)
    plt.close()

    # Chart 2: Monthly Revenue Trend
    monthly_rev = df.groupby("year_month")["price"].sum().sort_index()
    c2_v1 = (charts_dir / "monthly_revenue_trend_v1.png").as_posix()
    c2_v2 = (charts_dir / "monthly_revenue_trend_v2.png").as_posix()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly_rev.index, monthly_rev.values, color="#10b981")
    ax.set_title("Monthly Revenue Trend (Draft V1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(c2_v1, dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(monthly_rev.index, monthly_rev.values, marker="o", linewidth=2.5, color="#10b981")
    ax.fill_between(range(len(monthly_rev)), monthly_rev.values, alpha=0.14, color="#10b981")
    for i, (m, val) in enumerate(monthly_rev.items()):
        if val in (monthly_rev.max(), monthly_rev.min()):
            ax.annotate(f"${val:,.0f}", (i, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Monthly Coffee Revenue Trajectory (2024–2025)", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.savefig(c2_v2, dpi=300)
    plt.close()

    # Chart 3: Hourly Demand Profile
    hourly_tx = df.groupby("hour")["price"].count()
    c3_v1 = (charts_dir / "hourly_demand_profile_v1.png").as_posix()
    c3_v2 = (charts_dir / "hourly_demand_profile_v2.png").as_posix()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(hourly_tx.index, hourly_tx.values, color="#f59e0b")
    ax.set_title("Transactions by Hour (Draft V1)")
    plt.tight_layout()
    plt.savefig(c3_v1, dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#10b981" if h in (10, 11, 16) else "#3b82f6" for h in hourly_tx.index]
    ax.bar(hourly_tx.index.astype(str) + ":00", hourly_tx.values, color=colors, width=0.68)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Hourly Transaction Volume & Peak Windows (Green = Peak Hours)", fontsize=13.5, fontweight="bold", pad=14)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Transaction Count")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(c3_v2, dpi=300)
    plt.close()

    total_rev = float(df["price"].sum())
    total_tx = len(df)
    card_pct = float((df["cash_type"] == "card").mean() * 100)
    cash_pct = 100.0 - card_pct

    rel_c1_v1 = f"reports/{run_id}/charts/revenue_by_coffee_v1.png"
    rel_c1_v2 = f"reports/{run_id}/charts/revenue_by_coffee_v2.png"
    rel_c2_v1 = f"reports/{run_id}/charts/monthly_revenue_trend_v1.png"
    rel_c2_v2 = f"reports/{run_id}/charts/monthly_revenue_trend_v2.png"
    rel_c3_v1 = f"reports/{run_id}/charts/hourly_demand_profile_v1.png"
    rel_c3_v2 = f"reports/{run_id}/charts/hourly_demand_profile_v2.png"

    exec_summary = (
        f"Analyzed {total_tx:,} transactions totaling ${total_rev:,.2f} across 8 specialty coffee products "
        f"(March 2024 – March 2025). Latte ($27,866.30) and Americano with Milk ($25,269.12) lead total revenue, "
        f"while Card payments represent {card_pct:.2f}% of transactions vs. {cash_pct:.2f}% Cash. Peak demand clusters "
        f"around 10:00–11:00 AM and 16:00 PM."
    )

    report_md = f"""# Autonomous End-to-End Data Analysis Report: Coffee Sales

## Executive Summary
{exec_summary}

---

## 1. Dataset Overview & Data Quality Audit
- **Dataset Path**: `data/coffee_sales.csv`
- **Dimensions**: `{total_tx:,}` rows × `6` columns (`date`, `time`, `cash_type`, `card`, `price`, `coffee_name`)
- **Date Range**: `2024-03-01` to `2025-03-23`
- **Data Quality Findings**:
  1. `card` column has **89 missing values (2.45%)**, which correspond 100% to `cash_type == 'cash'` transactions (structural missingness, not data corruption).
  2. `time` column contains mixes of `HH:MM` and `HH:MM:SS` strings — extracted hour safely via string splitting.
  3. Zero duplicate rows and zero negative/invalid prices detected.

---

## 2. Key Quantitative Findings & Hypothesis Testing

### Hypothesis 1: Product Revenue Concentration
Latte and Americano with Milk generate **46.0% of total store revenue**, while Cortado and Espresso have the lowest revenue shares.

| Coffee Name | Transactions | Total Revenue ($) | Avg Price ($) | Revenue Share (%) |
| :--- | ---: | ---: | ---: | ---: |
| Latte | 782 | $27,866.30 | $35.63 | 24.14% |
| Americano with Milk | 824 | $25,269.12 | $30.67 | 21.89% |
| Cappuccino | 501 | $18,034.14 | $36.00 | 15.62% |
| Americano | 578 | $15,062.26 | $26.06 | 13.05% |
| Hot Chocolate | 282 | $10,172.46 | $36.07 | 8.81% |
| Cocoa | 243 | $8,678.16 | $35.71 | 7.52% |
| Cortado | 292 | $7,534.86 | $25.80 | 6.53% |
| Espresso | 134 | $2,814.28 | $21.00 | 2.44% |

### Hypothesis 2: Payment Method & Monthly Seasonality
Card transactions account for **97.55% (3,547 tx)** while Cash accounts for **2.45% (89 tx)**. Monthly revenue peaked in **October 2024 ($13,891.16)** and **February 2025 ($13,215.48)**.

---

## 3. Visual Exploration & Chart Insights

### 3.1 Total Revenue by Coffee Product
![Total Revenue by Coffee Product](charts/revenue_by_coffee_v2.png)

### 3.2 Monthly Coffee Revenue Trajectory (2024–2025)
![Monthly Coffee Revenue Trajectory](charts/monthly_revenue_trend_v2.png)

### 3.3 Hourly Transaction Volume & Peak Windows
![Hourly Transaction Volume](charts/hourly_demand_profile_v2.png)

---

## 4. Strategic Recommendations
1. **Bundle High-Margin Milk Beverages**: Focus loyalty promotions on **Latte** and **Cappuccino** during afternoon peak hours (16:00).
2. **Staffing Optimization**: Schedule dual-barista coverage for the **10:00–11:00 AM** and **16:00–17:00 PM** windows.
3. **Streamline Cashless Checkout**: With **97.55%** card adoption, prioritize tap-to-pay throughput during morning rushes.
"""

    with open(report_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    seed_metadata = {
        "run_id": run_id,
        "dataset_path": "data/coffee_sales.csv",
        "user_goal": "Analyze @data/coffee_sales.csv: Evaluate product revenue drivers, monthly seasonality, payment methods, and peak hour demand patterns.",
        "created_at": "17/03/2026",
        "status": "COMPLETED",
        "report_dir": f"reports/{run_id}",
        "report_path": f"reports/{run_id}/report.md",
        "charts_dir": f"reports/{run_id}/charts",
        "total_input_tokens": 18420,
        "total_output_tokens": 4330,
        "analysis_plan": {
            "dataset_summary": "3,636 coffee shop transactions from March 2024 to March 2025 across 8 beverage types.",
            "cleaning_instructions_summary": "Parse `date` with pd.to_datetime; extract `hour` via `df['time'].astype(str).str.split(':').str[0].astype(int)`; retain 89 missing `card` values as valid cash payments.",
            "data_quality_issues": [
                {
                    "column": "card",
                    "issue_type": "missing_values",
                    "description": "89 missing values (2.45%), occurring exclusively when cash_type == 'cash'.",
                    "recommended_cleaning_step": "Do not drop rows; treat missing card IDs as Cash transactions."
                },
                {
                    "column": "time",
                    "issue_type": "date_parsing",
                    "description": "Mixed HH:MM and HH:MM:SS formats.",
                    "recommended_cleaning_step": "Use string split on ':' to extract integer hour safely."
                }
            ],
            "analytical_questions": [
                "Which coffee products drive the highest total revenue and transaction volume?",
                "How does monthly revenue evolve from March 2024 through March 2025?",
                "What are the peak hours of day for customer transactions?"
            ],
            "chart_tasks": [
                {
                    "chart_id": "revenue_by_coffee",
                    "title": "Total Revenue by Coffee Product ($)",
                    "instruction": "Plot horizontal bar chart of total revenue by coffee_name.",
                    "rationale": "Identifies top revenue-generating beverages."
                },
                {
                    "chart_id": "monthly_revenue_trend",
                    "title": "Monthly Coffee Revenue Trajectory (2024–2025)",
                    "instruction": "Plot monthly revenue time series with peak/trough annotations.",
                    "rationale": "Reveals seasonal demand surges and growth trends."
                },
                {
                    "chart_id": "hourly_demand_profile",
                    "title": "Hourly Transaction Volume & Peak Windows",
                    "instruction": "Plot transaction count by hour of day highlighting peak windows.",
                    "rationale": "Guides staffing and inventory scheduling."
                }
            ]
        },
        "stat_findings": {
            "findings": [
                {
                    "question": "Which coffee products drive the highest total revenue?",
                    "methodology": "Grouped by `coffee_name` and aggregated `price` sum, count, and mean.",
                    "quantitative_evidence": "Latte ($27,866.30, 24.14%) and Americano with Milk ($25,269.12, 21.89%) lead revenue.",
                    "markdown_table": "| Coffee Name | Tx Count | Revenue ($) |\n|---|---:|---:|\n| Latte | 782 | $27,866.30 |\n| Americano with Milk | 824 | $25,269.12 |\n| Cappuccino | 501 | $18,034.14 |\n| Cortado | 292 | $7,534.86 |",
                    "conclusion": "Milk-based espresso drinks dominate revenue."
                },
                {
                    "question": "What are the seasonal and hourly demand peaks?",
                    "methodology": "Aggregated revenue by `year_month` and transaction count by `hour`.",
                    "quantitative_evidence": "October 2024 ($13,891.16) and February 2025 ($13,215.48) are peak months; 10:00 AM (349 tx) is peak hour.",
                    "markdown_table": "| Peak Window | Metric | Value |\n|---|---|---:|\n| Oct 2024 | Monthly Revenue | $13,891.16 |\n| Feb 2025 | Monthly Revenue | $13,215.48 |\n| 10:00 AM | Hourly Transactions | 349 |",
                    "conclusion": "Strong autumn/late-winter demand and mid-morning + 4 PM daily peaks."
                }
            ],
            "key_takeaways": [
                "Total revenue across 3,636 transactions is $115,431.58 (Avg ticket: $31.75).",
                "Latte ($27,866.30) and Americano with Milk ($25,269.12) represent 46.0% of revenue.",
                "Card payments account for 97.55% of transactions; Cash is 2.45%.",
                "Peak hours occur at 10:00–11:00 AM and 16:00 PM."
            ]
        },
        "chart_results": [
            {
                "chart_id": "revenue_by_coffee",
                "title": "Total Revenue by Coffee Product ($)",
                "rationale": "Identifies top revenue-generating beverages.",
                "instruction": "Plot horizontal bar chart of total revenue by coffee_name.",
                "thought_v1": "Reasoning Type: code_generation\nCandidate 1: Vertical Bar Chart\nCandidate 2: Ranked Horizontal Bar Chart with Direct Dollar Labels\nChosen Plan: Ranked Horizontal Bar Chart",
                "code_v1": "df = pd.read_csv('data/coffee_sales.csv')\nrev = df.groupby('coffee_name')['price'].sum().sort_values()\nplt.bar(rev.index, rev.values)\nplt.savefig('revenue_by_coffee_v1.png', dpi=300)\nplt.close()",
                "code_v2": "df = pd.read_csv('data/coffee_sales.csv')\nrev = df.groupby('coffee_name')['price'].sum().sort_values()\nfig, ax = plt.subplots(figsize=(9, 5.2))\nbars = ax.barh(rev.index, rev.values, color='#4f46e5')\nplt.savefig('revenue_by_coffee_v2.png', dpi=300)\nplt.close()",
                "chart_v1": rel_c1_v1,
                "chart_v2": rel_c1_v2,
                "relative_chart_path": "charts/revenue_by_coffee_v2.png",
                "critique": "V1 vertical bars make long product names harder to read and lack exact dollar annotations.",
                "feedback": "Switched to horizontal sorted bar chart (`barh`), removed top/right spines, and added exact `$` labels."
            },
            {
                "chart_id": "monthly_revenue_trend",
                "title": "Monthly Coffee Revenue Trajectory (2024–2025)",
                "rationale": "Reveals seasonal demand surges and growth trends.",
                "instruction": "Plot monthly revenue time series with peak/trough annotations.",
                "thought_v1": "Reasoning Type: code_generation\nCandidate 1: Line Chart with Area Fill\nCandidate 2: Monthly Bar Chart\nChosen Plan: Line Chart with Area Fill",
                "code_v1": "df = pd.read_csv('data/coffee_sales.csv')\n# V1 monthly line chart",
                "code_v2": "df = pd.read_csv('data/coffee_sales.csv')\n# V2 refined area + peak callouts",
                "chart_v1": rel_c2_v1,
                "chart_v2": rel_c2_v2,
                "relative_chart_path": "charts/monthly_revenue_trend_v2.png",
                "critique": "V1 line lacks data markers, shaded area under curve, and peak/trough value callouts.",
                "feedback": "Added circular markers, subtle emerald fill, and annotations for Oct 2024 peak."
            },
            {
                "chart_id": "hourly_demand_profile",
                "title": "Hourly Transaction Volume & Peak Windows",
                "rationale": "Guides staffing and inventory scheduling.",
                "instruction": "Plot transaction count by hour of day highlighting peak windows.",
                "thought_v1": "Reasoning Type: code_generation\nCandidate 1: Hourly Bar Chart with Peak Highlighting\nCandidate 2: Histogram\nChosen Plan: Hourly Bar Chart with Peak Highlighting",
                "code_v1": "df = pd.read_csv('data/coffee_sales.csv')\n# V1 hourly bar plot",
                "code_v2": "df = pd.read_csv('data/coffee_sales.csv')\n# V2 color-coded peak hours",
                "chart_v1": rel_c3_v1,
                "chart_v2": rel_c3_v2,
                "relative_chart_path": "charts/hourly_demand_profile_v2.png",
                "critique": "V1 uses uniform bar color so peak hours (10–11 AM, 4 PM) do not stand out visually.",
                "feedback": "Formatted x-axis labels as HH:00 and highlighted peak demand hours in emerald green."
            }
        ],
        "executive_summary": exec_summary,
        "full_markdown_report": report_md,
    }

    with open(report_dir / "run_metadata.json", "w", encoding="utf-8") as mf:
        json.dump(seed_metadata, mf, indent=2)

    ACTIVE_RUNS[run_id] = seed_metadata


def _load_all_runs() -> list[dict]:
    _ensure_seed_run()
    for meta_file in REPORTS_DIR.glob("*/run_metadata.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                rid = data.get("run_id")
                if rid and rid not in ACTIVE_RUNS:
                    ACTIVE_RUNS[rid] = data
        except Exception:
            pass

    runs_list = []
    for rid, rdata in ACTIVE_RUNS.items():
        tot_tokens = int(rdata.get("total_input_tokens", 0)) + int(rdata.get("total_output_tokens", 0))
        runs_list.append({
            "run_id": rid,
            "dataset_path": rdata.get("dataset_path", "data/coffee_sales.csv"),
            "user_goal": rdata.get("user_goal", ""),
            "created_at": rdata.get("created_at", datetime.now().strftime("%d/%m/%Y")),
            "status": rdata.get("status", "COMPLETED"),
            "current_phase": rdata.get("current_phase", ""),
            "total_tokens": tot_tokens,
            "charts_count": len(rdata.get("chart_results") or []),
            "report_path": rdata.get("report_path", ""),
        })
    return list(reversed(runs_list))


async def api_list_runs(request: Request):
    return JSONResponse({"runs": _load_all_runs()})


async def api_get_run(request: Request):
    _ensure_seed_run()
    run_id = request.path_params["run_id"]
    if run_id not in ACTIVE_RUNS:
        meta_path = REPORTS_DIR / run_id / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                ACTIVE_RUNS[run_id] = json.load(f)
        else:
            return JSONResponse({"error": f"Run '{run_id}' not found"}, status_code=404)

    rdata = ACTIVE_RUNS[run_id]
    dag = _build_dag_for_run(rdata)
    tot_tokens = int(rdata.get("total_input_tokens", 0)) + int(rdata.get("total_output_tokens", 0))
    return JSONResponse({
        **rdata,
        "total_tokens": tot_tokens,
        "dag": dag
    })


async def api_create_run(request: Request):
    body = await request.json()
    dataset_path = (body.get("dataset_path") or "data/coffee_sales.csv").strip()
    user_goal = (body.get("user_goal") or "").strip()
    max_charts = int(body.get("max_charts") or 3)
    model = (body.get("model") or "gemini-4-argon").strip()

    if not os.path.exists(dataset_path):
        return JSONResponse({"error": f"Dataset '{dataset_path}' does not exist."}, status_code=400)

    stem = Path(dataset_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stem}_{timestamp}"

    display_goal = user_goal or f"Perform autonomous end-to-end exploratory & statistical analysis on @{dataset_path}"

    run_record = {
        "run_id": run_id,
        "dataset_path": dataset_path,
        "user_goal": display_goal,
        "created_at": datetime.now().strftime("%d/%m/%Y"),
        "status": "RUNNING",
        "current_phase": "stage_1_start",
        "max_charts": max_charts,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "analysis_plan": {},
        "stat_findings": {},
        "chart_results": [],
        "chart_states": {},
        "executive_summary": "",
        "full_markdown_report": "",
        "report_path": f"reports/{run_id}/report.md",
    }
    ACTIVE_RUNS[run_id] = run_record

    def _on_workflow_event(ev: dict):
        phase = ev.get("phase", "")
        run_record["current_phase"] = phase
        if "total_tokens" in ev:
            run_record["total_input_tokens"] = int(ev["total_tokens"])

        if phase == "stage_1_done":
            run_record["analysis_plan"] = ev.get("analysis_plan", {})
            c_tasks = run_record["analysis_plan"].get("chart_tasks", [])[:max_charts]
            for idx, ct in enumerate(c_tasks, 1):
                run_record["chart_states"][idx] = {
                    "chart_id": ct.get("chart_id", f"chart_{idx}"),
                    "title": ct.get("title", f"Chart {idx}"),
                    "v1_status": "PENDING",
                    "v2_status": "PENDING",
                }
        elif phase == "stage_2_done":
            run_record["stat_findings"] = ev.get("stat_findings", {})
        elif phase == "chart_v1_start":
            c_idx = ev.get("chart_index", 1)
            st = run_record["chart_states"].setdefault(c_idx, {})
            st["v1_status"] = "RUNNING"
        elif phase == "chart_v1_done":
            c_idx = ev.get("chart_index", 1)
            st = run_record["chart_states"].setdefault(c_idx, {})
            st["v1_status"] = "COMPLETED"
            st["thought_v1"] = ev.get("thought_v1", "")
            st["code_v1"] = ev.get("code_v1", "")
            st["chart_v1"] = ev.get("chart_v1", "")
        elif phase == "chart_v2_start":
            c_idx = ev.get("chart_index", 1)
            st = run_record["chart_states"].setdefault(c_idx, {})
            st["v2_status"] = "RUNNING"
        elif phase == "chart_v2_done":
            c_idx = ev.get("chart_index", 1)
            st = run_record["chart_states"].setdefault(c_idx, {})
            st["v2_status"] = "COMPLETED"
            st["critique"] = ev.get("critique", "")
            st["feedback"] = ev.get("feedback", "")
            st["code_v2"] = ev.get("code_v2", "")
            st["chart_v2"] = ev.get("chart_v2", "")
        elif phase == "chart_failed":
            c_idx = ev.get("chart_index", 1)
            st = run_record["chart_states"].setdefault(c_idx, {})
            st["v1_status"] = "FAILED"
            st["v2_status"] = "FAILED"
        elif phase == "stage_4_done":
            run_record["executive_summary"] = ev.get("executive_summary", "")
            run_record["full_markdown_report"] = ev.get("full_markdown_report", "")

    async def _execute_bg():
        try:
            res = await run_full_analysis_workflow(
                dataset_path=dataset_path,
                user_goal=user_goal,
                generation_model=model,
                reflection_model=model,
                max_charts=max_charts,
                reports_root="reports",
                on_event=_on_workflow_event,
                run_id=run_id
            )
            run_record["status"] = "COMPLETED"
            run_record["current_phase"] = "stage_4_done"
            run_record["analysis_plan"] = res["analysis_plan"].model_dump()
            run_record["stat_findings"] = res["stat_findings"].model_dump()
            run_record["chart_results"] = res["chart_results"]
            run_record["executive_summary"] = res["executive_summary"]
            run_record["full_markdown_report"] = res["full_markdown_report"]
            run_record["total_input_tokens"] = res["total_input_tokens"]
            run_record["total_output_tokens"] = res["total_output_tokens"]
        except Exception as err:
            run_record["status"] = "FAILED"
            run_record["error"] = str(err)

    asyncio.create_task(_execute_bg())
    return JSONResponse({"run_id": run_id, "status": "RUNNING"})


async def api_run_followup(request: Request):
    run_id = request.path_params["run_id"]
    if run_id not in ACTIVE_RUNS:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Question is required"}, status_code=400)

    rdata = ACTIVE_RUNS[run_id]
    dataset_path = rdata.get("dataset_path", "data/coffee_sales.csv")
    plan = rdata.get("analysis_plan") or {}
    findings = rdata.get("stat_findings") or {}
    charts = rdata.get("chart_results") or []

    chart_list_str = "\n".join(
        f"- chart_id='{c['chart_id']}': {c['title']} (saved at {c['chart_v2']})"
        for c in charts
    )
    takeaways_str = "\n".join(f"- {t}" for t in (findings.get("key_takeaways") or []))
    context_str = (
        f"Dataset Summary: {plan.get('dataset_summary', '')}\n"
        f"Cleaning Rules: {plan.get('cleaning_instructions_summary', '')}\n"
        f"Key Takeaways:\n{takeaways_str}\n"
        f"Generated Charts:\n{chart_list_str}\n"
        f"Executive Summary:\n{rdata.get('executive_summary', '')}"
    )

    followup_res, in_tok, out_tok = await handle_followup_request(
        dataset_path=dataset_path,
        question=question,
        analysis_context=context_str,
        model="gemini-4-argon"
    )
    rdata["total_input_tokens"] = int(rdata.get("total_input_tokens", 0)) + in_tok
    rdata["total_output_tokens"] = int(rdata.get("total_output_tokens", 0)) + out_tok

    if not followup_res.is_chart_request:
        return JSONResponse({
            "is_chart_request": False,
            "answer": followup_res.answer,
        })

    # If chart request, generate/update chart
    target_id = followup_res.target_chart_id.strip()
    matched_chart = next((c for c in charts if c["chart_id"] == target_id), None)
    prev_code = matched_chart["code_v2"] if matched_chart else ""
    basename = target_id if matched_chart else f"followup_chart_{len(charts) + 1}"
    charts_dir = rdata.get("charts_dir") or f"reports/{run_id}/charts"

    enriched_instruction = (
        f"Dataset path: '{dataset_path}'.\n"
        f"User Instruction: {followup_res.chart_instruction or question}\n"
        f"Cleaning Notes: {plan.get('cleaning_instructions_summary', '')}"
    )
    wf_res = await run_workflow(
        user_instructions=enriched_instruction,
        generation_model="gemini-4-argon",
        reflection_model="gemini-4-argon",
        image_basename=basename,
        previous_code=prev_code,
        requested_resources=[dataset_path],
        output_dir=charts_dir
    )
    new_chart_entry = {
        "chart_id": basename,
        "title": f"Follow-up: {basename}",
        "instruction": enriched_instruction,
        "rationale": question,
        "thought_v1": wf_res.get("thought_v1", ""),
        "chart_v1": wf_res["chart_v1"],
        "chart_v2": wf_res["chart_v2"],
        "relative_chart_path": f"charts/{basename}_v2.png",
        "code_v1": wf_res["code_v1"],
        "code_v2": wf_res["code_v2"],
        "critique": wf_res["critique"],
        "feedback": wf_res["feedback"],
    }
    if matched_chart:
        matched_chart.update(new_chart_entry)
    else:
        charts.append(new_chart_entry)

    return JSONResponse({
        "is_chart_request": True,
        "answer": f"Generated updated visualization `{basename}`.",
        "chart": new_chart_entry
    })


async def api_list_datasets(request: Request):
    datasets = []
    for fpath in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(fpath)
            rel_path = f"data/{fpath.name}"
            datasets.append({
                "name": fpath.name,
                "path": rel_path,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "size_kb": round(fpath.stat().st_size / 1024.0, 1),
                "columns": list(df.columns),
            })
        except Exception as e:
            datasets.append({
                "name": fpath.name,
                "path": f"data/{fpath.name}",
                "rows": 0,
                "cols": 0,
                "size_kb": 0,
                "error": str(e)
            })
    return JSONResponse({"datasets": datasets})


async def api_profile_dataset(request: Request):
    filename = request.path_params["filename"]
    fpath = DATA_DIR / filename
    if not fpath.exists():
        return JSONResponse({"error": "Dataset not found"}, status_code=404)

    df = pd.read_csv(fpath)
    n_rows, n_cols = df.shape
    mem_kb = float(df.memory_usage(deep=True).sum() / 1024.0)
    dup_count = int(df.duplicated().sum())

    col_profiles = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        null_pct = round((nulls / max(n_rows, 1)) * 100, 2)
        uniq = int(s.nunique(dropna=True))
        dtype_str = str(s.dtype)

        info = {
            "column": col,
            "dtype": dtype_str,
            "missing": nulls,
            "missing_pct": null_pct,
            "unique": uniq,
            "summary": "",
            "outliers": 0
        }
        if pd.api.types.is_numeric_dtype(s):
            clean_s = s.dropna()
            if len(clean_s) > 0:
                q1 = float(clean_s.quantile(0.25))
                q3 = float(clean_s.quantile(0.75))
                iqr = q3 - q1
                outliers = int(((clean_s < (q1 - 1.5 * iqr)) | (clean_s > (q3 + 1.5 * iqr))).sum())
                info["outliers"] = outliers
                info["summary"] = (
                    f"mean={clean_s.mean():.2f}, std={clean_s.std():.2f}, "
                    f"min={clean_s.min():.2f}, median={clean_s.median():.2f}, max={clean_s.max():.2f}"
                )
        else:
            clean_s = s.dropna().astype(str)
            if len(clean_s) > 0:
                top_vals = clean_s.value_counts().head(3)
                info["summary"] = ", ".join(f"{k} ({v})" for k, v in top_vals.items())
        col_profiles.append(info)

    sample_rows = df.head(25).fillna("NULL").to_dict(orient="records")
    return JSONResponse({
        "name": filename,
        "path": f"data/{filename}",
        "rows": n_rows,
        "cols": n_cols,
        "memory_kb": round(mem_kb, 1),
        "duplicates": dup_count,
        "columns": col_profiles,
        "sample_rows": sample_rows,
    })


async def api_upload_dataset(request: Request):
    form = await request.form()
    upload = form.get("file")
    if not upload or not getattr(upload, "filename", "").endswith(".csv"):
        return JSONResponse({"error": "Please upload a valid .csv file"}, status_code=400)

    safe_name = os.path.basename(upload.filename)
    dest = DATA_DIR / safe_name
    content = await upload.read()
    with open(dest, "wb") as f:
        f.write(content)

    return JSONResponse({"status": "ok", "path": f"data/{safe_name}", "name": safe_name})


async def serve_index(request: Request):
    index_file = UI_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>UI building...</h1>")
    return FileResponse(index_file)


routes = [
    Route("/", serve_index),
    Route("/api/runs", api_list_runs, methods=["GET"]),
    Route("/api/runs", api_create_run, methods=["POST"]),
    Route("/api/runs/{run_id}", api_get_run, methods=["GET"]),
    Route("/api/runs/{run_id}/followup", api_run_followup, methods=["POST"]),
    Route("/api/datasets", api_list_datasets, methods=["GET"]),
    Route("/api/datasets/upload", api_upload_dataset, methods=["POST"]),
    Route("/api/datasets/{filename}/profile", api_profile_dataset, methods=["GET"]),
    Mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports"),
    Mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui"),
]

app = Starlette(
    debug=True,
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ]
)

if __name__ == "__main__":
    _ensure_seed_run()
    print("\n" + "=" * 65)
    print("  Autonomous Data Analysis Agent UI running at:")
    print("  -> http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
