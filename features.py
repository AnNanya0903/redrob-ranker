"""
features.py — Feature extraction, honeypot detection, and scoring logic
for the Redrob "Intelligent Candidate Discovery & Ranking Challenge".

Pure stdlib. CPU-only. No network. No GPU. Designed to run over 100K
candidates well within a 5-minute / 16GB budget (typical run: ~15-25s).
"""

from __future__ import annotations
import re
from datetime import date, datetime
from math import exp

TODAY = date(2026, 7, 1)  # dataset "as of" reference date

# ---------------------------------------------------------------------------
# JD-derived keyword sets
# ---------------------------------------------------------------------------

EMBEDDING_TERMS = [
    "sentence-transformers", "sentence transformers", "sbert",
    "openai embeddings", "text-embedding", "bge", "e5", "embedding",
    "embeddings", "dense retrieval", "semantic search",
]

VECTOR_DB_TERMS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector db",
    "hybrid search", "ann search", "approximate nearest neighbor",
]

EVAL_TERMS = [
    "ndcg", "mrr", "map@", "mean average precision", "precision@",
    "recall@", "a/b test", "a/b testing", "ab testing",
    "offline-to-online", "offline evaluation", "evaluation framework",
    "learning to rank", "learning-to-rank", "ltr",
]

RANKING_RETRIEVAL_TERMS = [
    "ranking", "retrieval", "recommendation", "recommender",
    "search relevance", "matching system", "candidate matching",
    "information retrieval", "re-ranking", "reranking", "bm25",
]

LLM_TERMS = ["llm", "large language model", "gpt", "fine-tun", "lora",
             "qlora", "peft", "prompt engineering", "rag"]

WRAPPER_ONLY_TERMS = ["langchain", "openai api", "chatgpt", "gpt wrapper",
                       "llamaindex", "llama-index"]

PRE_LLM_ML_TERMS = ["xgboost", "gradient boost", "collaborative filtering",
                     "click-through", "ctr prediction", "feature engineering",
                     "recommendation engine", "search ranking", "bm25",
                     "tf-idf", "word2vec", "svd", "matrix factorization"]

PYTHON_TERM = "python"

RESEARCH_ONLY_SIGNALS = ["research scientist", "postdoc", "post-doc",
                          "phd researcher", "research fellow"]
RESEARCH_ORG_TERMS = ["university", "institute", "research lab",
                       "laboratory", "academy"]
PRODUCTION_SIGNALS = ["deployed", "production", "shipped", "real users",
                       "scale", "live traffic", "users", "in prod"]

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra", "mindtree",
    "l&t infotech", "ltimindtree", "mphasis", "birlasoft", "persistent systems",
}

TITLE_LEADERSHIP_NO_CODE = [
    "architect", "engineering manager", "head of", "director",
    "vp of engineering", "vp engineering", "chief technology officer", "cto",
]
TITLE_IC_HINT = ["staff engineer", "principal engineer", "tech lead",
                  "senior engineer", "software engineer", "ml engineer",
                  "machine learning engineer", "applied scientist"]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "asr", "text-to-speech", "tts", "robotics",
    "slam", "autonomous", "lidar", "sensor fusion",
]
NLP_IR_TERMS = ["nlp", "natural language", "retrieval", "search", "ranking",
                "embeddings", "information retrieval", "text classification",
                "recommendation", "matching"]

TITLE_NON_TECH = [
    "marketing", "sales", "recruiter", "hr ", "human resources",
    "account manager", "business development", "customer success",
    "operations manager", "product marketing", "content", "finance",
]

# Locations
PREFERRED_CITIES = {"pune", "noida"}
WELCOME_INDIA_CITIES = {
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bangalore",
    "bengaluru", "pune", "noida",
}
TIER1_INDIA_CITIES = WELCOME_INDIA_CITIES | {"chennai", "kolkata"}


def _text_blob(candidate: dict) -> str:
    """Concatenate all free-text fields (lowercased) for keyword scanning."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts).lower()


def _skill_index(candidate: dict) -> dict:
    idx = {}
    for s in candidate.get("skills", []):
        idx[s["name"].strip().lower()] = s
    return idx


def _any_term_in(terms, blob) -> bool:
    return any(t in blob for t in terms)


def _count_terms_in(terms, blob) -> int:
    return sum(1 for t in terms if t in blob)


def _skill_strength(skill_idx: dict, terms: list) -> float:
    """
    Score 0-1 for how strongly a set of related skill terms is backed by
    real (non-stuffed) evidence: proficiency + endorsements + duration.
    Falls back to a weaker text-mention signal if no matching skill entry.
    """
    best = 0.0
    for name, s in skill_idx.items():
        if any(t in name for t in terms):
            prof_weight = {"beginner": 0.25, "intermediate": 0.5,
                           "advanced": 0.8, "expert": 1.0}.get(s.get("proficiency"), 0.4)
            duration = s.get("duration_months", 0) or 0
            endorsements = s.get("endorsements", 0) or 0
            # Trust discount: high proficiency claimed with ~no duration/endorsements
            # looks like keyword stuffing.
            trust = 1.0
            if prof_weight >= 0.8 and duration == 0 and endorsements == 0:
                trust = 0.25
            elif duration == 0 and endorsements == 0:
                trust = 0.6
            duration_bonus = min(duration / 24.0, 1.0)  # saturates at 2 years
            endorsement_bonus = min(endorsements / 10.0, 1.0)
            score = trust * (0.5 * prof_weight + 0.3 * duration_bonus + 0.2 * endorsement_bonus)
            best = max(best, score)
    return best


def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Rule-based honeypot detection using internal profile inconsistencies.
    Returns (is_honeypot, reasons).
    Calibrated against the full 100K pool: three independent heuristics
    below flag ~70 candidates with zero overlap between heuristics,
    closely matching the "~80 honeypots" the spec describes.
    """
    reasons = []
    profile = candidate["profile"]
    ch = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0) or 0

    total_months = sum(job.get("duration_months", 0) or 0 for job in ch)
    if yoe > 0 and total_months > yoe * 12 * 1.3:
        reasons.append("career-history duration far exceeds stated years_of_experience")

    starts = []
    for job in ch:
        try:
            starts.append(date.fromisoformat(job["start_date"]))
        except (ValueError, TypeError, KeyError):
            continue
    if starts:
        first_start = min(starts)
        career_span_years = (TODAY - first_start).days / 365.25
        if yoe > career_span_years + 0.5:
            reasons.append("claimed experience predates earliest documented role "
                            "(implies employment before the role/company existed)")

    expert_zero = sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") in ("expert", "advanced") and (s.get("duration_months", 0) or 0) == 0
    )
    if expert_zero >= 3:
        reasons.append(f"{expert_zero} skills claimed at expert/advanced level with 0 months of use")

    return (len(reasons) > 0, reasons)


def extract_features(candidate: dict) -> dict:
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    blob = _text_blob(candidate)
    skill_idx = _skill_index(candidate)
    ch = candidate.get("career_history", [])

    f = {}

    # --- Required skills (JD "absolutely need") ---
    f["embeddings_score"] = max(_skill_strength(skill_idx, EMBEDDING_TERMS),
                                 0.3 if _any_term_in(EMBEDDING_TERMS, blob) else 0.0)
    f["vectordb_score"] = max(_skill_strength(skill_idx, VECTOR_DB_TERMS),
                               0.3 if _any_term_in(VECTOR_DB_TERMS, blob) else 0.0)
    f["python_score"] = _skill_strength(skill_idx, [PYTHON_TERM])
    f["eval_score"] = max(_skill_strength(skill_idx, EVAL_TERMS),
                           0.3 if _any_term_in(EVAL_TERMS, blob) else 0.0)
    f["ranking_retrieval_hits"] = _count_terms_in(RANKING_RETRIEVAL_TERMS, blob)

    # --- Nice-to-haves ---
    f["finetune_score"] = _skill_strength(skill_idx, ["lora", "qlora", "peft", "fine-tun"])
    f["ltr_score"] = _skill_strength(skill_idx, ["xgboost", "learning to rank", "learning-to-rank"])
    f["hrtech_bonus"] = 1.0 if _any_term_in(
        ["hr-tech", "hr tech", "recruiting", "talent", "marketplace", "job board"], blob) else 0.0
    f["opensource_bonus"] = 1.0 if _any_term_in(
        ["open source", "open-source", "github.com", "published paper", "conference talk"], blob) else 0.0

    # --- Title relevance ---
    title = (profile.get("current_title") or "").lower()
    f["title_non_tech"] = _any_term_in(TITLE_NON_TECH, title)
    f["title_leadership_no_code"] = False
    if ch:
        cur = next((j for j in ch if j.get("is_current")), ch[0])
        cur_title = (cur.get("title") or "").lower()
        cur_duration = cur.get("duration_months", 0) or 0
        if _any_term_in(TITLE_LEADERSHIP_NO_CODE, cur_title) and cur_duration >= 18 \
                and not _any_term_in(TITLE_IC_HINT, cur_title):
            f["title_leadership_no_code"] = True

    # --- Experience years (JD sweet spot 5-9, soft falloff outside) ---
    yoe = profile.get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        f["experience_fit"] = 1.0
    elif yoe < 5:
        f["experience_fit"] = max(0.0, 1.0 - (5 - yoe) * 0.22)
    else:
        f["experience_fit"] = max(0.0, 1.0 - (yoe - 9) * 0.10)
    f["years_of_experience"] = yoe

    # --- Company type: consulting-only vs product-company experience ---
    companies = [(job.get("company") or "").strip().lower() for job in ch]
    f["all_consulting"] = bool(companies) and all(
        any(cf in c for cf in CONSULTING_FIRMS) for c in companies
    )
    f["any_consulting"] = any(any(cf in c for cf in CONSULTING_FIRMS) for c in companies)

    # --- Pure research without production deployment ---
    research_titles = any(_any_term_in(RESEARCH_ONLY_SIGNALS, (job.get("title") or "").lower()) for job in ch)
    research_orgs = any(_any_term_in(RESEARCH_ORG_TERMS, (job.get("company") or "").lower()) for job in ch)
    has_production_signal = _any_term_in(PRODUCTION_SIGNALS, blob)
    f["pure_research_no_prod"] = (research_titles or research_orgs) and not has_production_signal

    # --- Recent LLM-wrapper-only experience (<12mo, no pre-LLM ML production exp) ---
    wrapper_only = _any_term_in(WRAPPER_ONLY_TERMS, blob) and not _any_term_in(PRE_LLM_ML_TERMS, blob)
    recent_ai_short = False
    for job in ch:
        title_l = (job.get("title") or "").lower()
        desc_l = (job.get("description") or "").lower()
        if _any_term_in(LLM_TERMS, title_l + " " + desc_l) and (job.get("duration_months", 0) or 0) < 12:
            recent_ai_short = True
    f["llm_wrapper_only_recent"] = wrapper_only and recent_ai_short and yoe < 3

    # --- Domain adjacency (CV/speech/robotics without NLP/IR) ---
    has_cv_speech_robo = _any_term_in(CV_SPEECH_ROBOTICS_TERMS, blob)
    has_nlp_ir = _any_term_in(NLP_IR_TERMS, blob)
    f["cv_speech_robo_only"] = has_cv_speech_robo and not has_nlp_ir

    # --- Location ---
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    city_token = location.split(",")[0].strip()
    if country not in ("india", ""):
        f["location_score"] = 0.15
    elif city_token in PREFERRED_CITIES:
        f["location_score"] = 1.0
    elif city_token in WELCOME_INDIA_CITIES:
        f["location_score"] = 0.85
    elif city_token in TIER1_INDIA_CITIES:
        f["location_score"] = 0.7
    else:
        f["location_score"] = 0.45  # other India city; relocation possible but unconfirmed tier-1 origin

    # --- Notice period ---
    notice = signals.get("notice_period_days")
    if notice is None:
        f["notice_score"] = 0.6
    elif notice <= 30:
        f["notice_score"] = 1.0
    elif notice <= 60:
        f["notice_score"] = 0.7
    elif notice <= 90:
        f["notice_score"] = 0.5
    else:
        f["notice_score"] = 0.3
    f["notice_period_days"] = notice

    # --- Behavioral / Redrob signal modifier ---
    f["behavior_modifier"] = _behavior_modifier(signals)
    f["signals"] = signals

    return f


def _days_since(date_str: str | None) -> float | None:
    if not date_str:
        return None
    try:
        d = date.fromisoformat(date_str)
    except ValueError:
        return None
    return (TODAY - d).days


def _behavior_modifier(signals: dict) -> float:
    """
    Multiplicative modifier in roughly [0.5, 1.15] reflecting how
    "actually reachable / hirable" the candidate looks right now.
    """
    m = 1.0

    # Recency of activity — most important behavioral signal per redrob_signals_doc.
    days_inactive = _days_since(signals.get("last_active_date"))
    if days_inactive is not None:
        if days_inactive <= 14:
            m *= 1.10
        elif days_inactive <= 30:
            m *= 1.03
        elif days_inactive <= 90:
            m *= 0.90
        elif days_inactive <= 180:
            m *= 0.70
        else:
            m *= 0.50

    if signals.get("open_to_work_flag"):
        m *= 1.05

    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        m *= (0.85 + 0.30 * rr)  # 0.85x at rr=0 .. 1.15x at rr=1

    icr = signals.get("interview_completion_rate")
    if icr is not None:
        m *= (0.90 + 0.15 * icr)

    oar = signals.get("offer_acceptance_rate")
    if oar is not None and oar >= 0:
        m *= (0.92 + 0.10 * oar)

    completeness = signals.get("profile_completeness_score")
    if completeness is not None:
        m *= (0.92 + 0.0008 * completeness)  # up to ~1.0 at 100

    if signals.get("verified_email") and signals.get("verified_phone"):
        m *= 1.02

    return m


def score_candidate(candidate: dict) -> dict:
    is_honeypot, honeypot_reasons = detect_honeypot(candidate)
    f = extract_features(candidate)

    skills_component = (
        0.32 * f["embeddings_score"] +
        0.28 * f["vectordb_score"] +
        0.18 * f["python_score"] +
        0.22 * f["eval_score"]
    )
    skills_component += 0.05 * min(f["ranking_retrieval_hits"] / 3.0, 1.0)
    skills_component += 0.03 * f["finetune_score"] + 0.03 * f["ltr_score"]
    skills_component += 0.02 * f["hrtech_bonus"] + 0.02 * f["opensource_bonus"]
    skills_component = min(skills_component, 1.15)

    title_component = 1.0
    if f["title_non_tech"]:
        title_component = 0.05
    elif f["title_leadership_no_code"]:
        title_component = 0.55

    company_component = 1.0
    if f["all_consulting"]:
        company_component = 0.35
    elif f["any_consulting"]:
        company_component = 0.85

    if f["pure_research_no_prod"]:
        company_component *= 0.30
    if f["llm_wrapper_only_recent"]:
        skills_component *= 0.35
    if f["cv_speech_robo_only"]:
        skills_component *= 0.45

    composite = (
        0.34 * skills_component +
        0.16 * title_component +
        0.14 * company_component +
        0.14 * f["experience_fit"] +
        0.12 * f["location_score"] +
        0.10 * f["notice_score"]
    )

    composite *= f["behavior_modifier"]

    if is_honeypot:
        composite = min(composite, 0.02)

    return {
        "score": round(composite, 6),
        "is_honeypot": is_honeypot,
        "honeypot_reasons": honeypot_reasons,
        "features": f,
        "skills_component": skills_component,
        "title_component": title_component,
        "company_component": company_component,
    }
