"""
app.py — Streamlit sandbox for the Redrob Ranker (submission_spec.md
Section 10.5).

Accepts a small candidate sample (<=100 candidates) as a .jsonl or .json
upload, runs the exact same ranking logic as rank.py end-to-end on CPU,
and lets you download the ranked CSV. This is the "sandbox / demo" that
organizers use as a fast sanity check before the full Stage-3
reproduction of rank.py against the real 100K pool.

Deploy target: Streamlit Community Cloud or HuggingFace Spaces
(Streamlit SDK). Both are free-tier compatible — this app has no GPU
requirement and makes no external network calls.

Local run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import json
import time

import streamlit as st

from features import score_candidate
from reasoning import build_reasoning

st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")

st.title("Redrob Ranker — Sandbox")
st.caption(
    "Upload a small candidate sample (≤100 candidates, .jsonl or .json) "
    "and run the same rule-based ranker used to produce the full "
    "top-100 submission. CPU-only, no network calls, no GPU."
)

with st.expander("What this sandbox checks", expanded=False):
    st.markdown(
        "- The ranking logic in `features.py` / `reasoning.py` runs "
        "end-to-end on a sample\n"
        "- Output matches the required submission schema: "
        "`candidate_id, rank, score, reasoning`\n"
        "- Runtime and honeypot flags are shown for transparency\n\n"
        "This sandbox does **not** need to handle the full 100K pool — "
        "that full reproduction happens separately via `rank.py` "
        "against `candidates.jsonl` (see README.md for the single "
        "reproduce command)."
    )

uploaded = st.file_uploader(
    "Candidate sample (.jsonl — one JSON object per line, or .json — a list of objects)",
    type=["jsonl", "json"],
)

sample_note = st.empty()

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    candidates = []
    parse_error = None
    try:
        if uploaded.name.endswith(".json"):
            data = json.loads(raw)
            candidates = data if isinstance(data, list) else [data]
        else:
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    except json.JSONDecodeError as e:
        parse_error = str(e)

    if parse_error:
        st.error(f"Could not parse uploaded file as JSON/JSONL: {parse_error}")
    elif len(candidates) == 0:
        st.warning("No candidate records found in the uploaded file.")
    else:
        if len(candidates) > 100:
            st.warning(
                f"Uploaded {len(candidates)} candidates — sandbox is scoped to "
                f"≤100 per the spec. Using the first 100."
            )
            candidates = candidates[:100]

        sample_note.info(f"Loaded {len(candidates)} candidate(s). Running ranker...")

        t0 = time.time()
        results = []
        n_honeypot = 0
        errors = []
        for c in candidates:
            try:
                r = score_candidate(c)
                if r["is_honeypot"]:
                    n_honeypot += 1
                results.append((c.get("candidate_id", "UNKNOWN"), r["score"], c, r))
            except Exception as e:  # noqa: BLE001 — surface bad records to the user, don't crash the demo
                errors.append((c.get("candidate_id", "UNKNOWN"), str(e)))
        elapsed = time.time() - t0

        results.sort(key=lambda x: (-x[1], x[0]))

        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates scored", len(results))
        col2.metric("Honeypots flagged", n_honeypot)
        col3.metric("Runtime", f"{elapsed:.3f}s")

        if errors:
            st.error(f"{len(errors)} record(s) failed to score (likely missing required fields):")
            for cid, msg in errors[:10]:
                st.text(f"  {cid}: {msg}")

        rows = []
        for rank, (cid, score, candidate, result) in enumerate(results, start=1):
            reasoning = build_reasoning(candidate, result)
            rows.append({
                "candidate_id": cid,
                "rank": rank,
                "score": round(score, 6),
                "reasoning": reasoning,
                "honeypot": result["is_honeypot"],
            })

        st.subheader("Ranked output")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Build downloadable CSV matching the exact submission schema
        # (candidate_id, rank, score, reasoning — no honeypot column).
        csv_buf = io.StringIO()
        csv_buf.write("candidate_id,rank,score,reasoning\n")
        import csv as csv_module
        writer = csv_module.writer(csv_buf)
        for r in rows:
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.6f}", r["reasoning"]])

        st.download_button(
            "Download ranked CSV",
            data=csv_buf.getvalue(),
            file_name="sandbox_ranking.csv",
            mime="text/csv",
        )
else:
    st.info(
        "No file uploaded yet. You can test with the first 50 rows of "
        "`sample_candidates.json` from the hackathon bundle."
    )
