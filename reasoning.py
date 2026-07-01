"""
reasoning.py — Generates a 1-2 sentence, fact-grounded reasoning string
for each ranked candidate. Every claim is pulled directly from the
candidate's own record (skills, career_history, redrob_signals) so
nothing is hallucinated, per submission_spec.md Section 3 reasoning checks.
"""

from features import _days_since, TODAY


def _top_matching_skills(candidate: dict, result: dict, limit=2) -> list[str]:
    f = result["features"]
    hits = []
    blob_terms = {
        "embeddings": f["embeddings_score"],
        "vector search / retrieval infra": f["vectordb_score"],
        "Python": f["python_score"],
        "ranking evaluation (NDCG/MRR/A-B testing)": f["eval_score"],
    }
    ranked = sorted(blob_terms.items(), key=lambda kv: kv[1], reverse=True)
    for name, score in ranked:
        if score >= 0.4:
            hits.append(name)
        if len(hits) >= limit:
            break
    return hits


def _current_role_phrase(candidate: dict) -> str:
    p = candidate["profile"]
    return f"{p.get('current_title', 'their current role')} at {p.get('current_company', 'their current company')}"


def build_reasoning(candidate: dict, result: dict) -> str:
    p = candidate["profile"]
    f = result["features"]
    yoe = f["years_of_experience"]
    role = _current_role_phrase(candidate)
    skills_hit = _top_matching_skills(candidate, result)

    clauses = []

    # Opening: experience + role framing
    if skills_hit:
        clauses.append(f"{yoe:.1f} years experience, currently {role}, with hands-on {', '.join(skills_hit)}")
    else:
        clauses.append(f"{yoe:.1f} years experience, currently {role}")

    # Company/title flags
    if f["all_consulting"]:
        clauses.append("entire career has been at IT-services/consulting firms with no product-company tenure")
    if f["title_leadership_no_code"]:
        clauses.append("has been in an architecture/leadership title for 18+ months, raising a hands-on-coding concern")
    if f["pure_research_no_prod"]:
        clauses.append("background skews research-only with no clear production deployment")
    if f["cv_speech_robo_only"]:
        clauses.append("core expertise is in computer vision/speech/robotics rather than NLP/IR")
    if f["llm_wrapper_only_recent"]:
        clauses.append("AI experience looks limited to a recent LangChain/API-wrapper stint without deeper pre-LLM ML background")

    # Location / notice
    loc = p.get("location", "")
    country = p.get("country", "")
    if f["location_score"] >= 0.85:
        clauses.append(f"based in {loc} ({country}), matching the Pune/Noida-preferred footprint")
    elif f["location_score"] <= 0.2:
        clauses.append(f"based outside India in {loc}, which is case-by-case per the JD given no visa sponsorship")
    notice = f.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            clauses.append(f"a {notice}-day notice period fits the JD's sub-30-day preference")
        elif notice > 90:
            clauses.append(f"a long {notice}-day notice period is a practical concern")

    # Behavioral signal
    signals = f.get("signals", {})
    days_inactive = _days_since(signals.get("last_active_date"))
    rr = signals.get("recruiter_response_rate")
    if days_inactive is not None and days_inactive > 120:
        clauses.append(f"inactive on the platform for {days_inactive} days, so reachability is uncertain")
    elif rr is not None and rr >= 0.5 and (days_inactive is None or days_inactive <= 30):
        clauses.append(f"recently active with a {rr:.0%} recruiter response rate")

    # Compose into 1-2 sentences (capitalize only the first letter of each
    # sentence; never lowercase the rest, which would mangle proper nouns).
    def _cap_first(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    if len(clauses) <= 2:
        text = _cap_first("; ".join(clauses)) + "."
    else:
        first = _cap_first(clauses[0]) + "."
        rest = _cap_first("; ".join(clauses[1:])) + "."
        text = first + " " + rest

    # keep it tight
    if len(text) > 400:
        text = text[:397].rsplit(" ", 1)[0] + "..."
    return text
