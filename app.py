from dotenv import load_dotenv

import asyncio
import io
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
import pandas as pd
import streamlit as st
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, TypeAdapter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
# Load GEMINI_API_KEY from a .env file sitting next to this script.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

APP_TITLE = "Job Search Utility Tool"
MODEL_NAME = "gemini-3.8-flash"
MAX_MARKDOWN_CHARS = 12000  # per crawled page, enforced with a slice
MAX_RESUME_CHARS = 8000
MAX_CANDIDATE_LINKS = 25
PAGE_TIMEOUT_MS = 45000
GEMINI_MAX_ATTEMPTS = 3

# Links that mention "careers" but are not job pages (logins, demos, help...).
SKIP_LINK_WORDS = (
    "login",
    "log-in",
    "signin",
    "sign-in",
    "signup",
    "sign-up",
    "/demo",
    "/help",
    "help.",
    "support.",
    "account.",
    "/privacy",
    "/terms",
    "/legal",
    "/blog",
    "/benefits",
)
ATS_DOMAINS = (
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "workable.com",
    "smartrecruiters.com",
    "bamboohr.com",
)
JOB_LIST_WORDS = (
    "open-positions",
    "open-roles",
    "openings",
    "all-jobs",
    "search-jobs",
    "/jobs",
    "/positions",
    "/roles",
)
LOCALE_SEGMENT = re.compile(r"^[a-z]{2}([-_][a-z]{2})?$")
ANTI_BOT_MARKERS = ("anti-bot", "cloudflare", "captcha", "http 403", "access denied")
BLOCKS_BEFORE_SKIPPING_SITE = 2

# Public job-board APIs (no key needed). Reading these gives the complete list
# of open roles instead of whatever fits in a 12,000-character page crawl.
GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
LEVER_API = "https://api.lever.co/v0/postings/{token}?mode=json"
ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/{token}"
ATS_BOARD_PATTERNS = (
    (
        "Greenhouse",
        re.compile(
            r"greenhouse\.io/(?:embed/job_board(?:/js)?\?for=|embed/job_app\?for=)?([A-Za-z0-9_-]+)",
            re.IGNORECASE,
        ),
    ),
    ("Lever", re.compile(r"jobs\.(?:eu\.)?lever\.co/([A-Za-z0-9_.-]+)", re.IGNORECASE)),
    ("Ashby", re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_.%-]+)", re.IGNORECASE)),
)
NOT_BOARD_TOKENS = {"embed", "v1", "boards", "jobs", "api", "static", "assets", "js", "css"}
MAX_BOARDS_PER_COMPANY = 2
MAX_MATCHED_ROLES = 40
HTTP_TIMEOUT_SECONDS = 20
# Spelling variants folded together before titles are compared.
TITLE_EQUIVALENTS = (
    (r"full[\s-]+stack", "fullstack"),
    (r"back[\s-]+end", "backend"),
    (r"front[\s-]+end", "frontend"),
    (r"dev[\s-]+ops", "devops"),
    (r"\bsr\b\.?", "senior"),
    (r"\bjr\b\.?", "junior"),
    (r"\bswe\b", "software engineer"),
    (r"\bsde\b", "software development engineer"),
)
US_STATE_CODES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split()
US_LOCATION_PATTERN = re.compile(
    r"united states|\busa\b|\bu\.s\.|\bus\b|, (?:" + "|".join(US_STATE_CODES) + r")\b",
    re.IGNORECASE,
)

# The app never passes Python functions as tools, so the SDK's automatic
# function calling is switched off (this also silences its AFC warning).
NO_AUTO_FUNCTION_CALLING = types.AutomaticFunctionCallingConfig(disable=True)

# Public H-1B record sources
H1B_RECORDS_CONCURRENCY = 2  # be polite to the third-party lookup sites
MAX_SEARCH_CHARS = 6000
H1BDATA_URL = "https://www.h1bdata.info/index.php?em={query}&job=&city=&year=all+years"
MYVISAJOBS_URL = "https://www.myvisajobs.com/employer/{slug}/"
DOL_PERFORMANCE_URL = "https://www.dol.gov/agencies/eta/foreign-labor/performance"
DOL_COLUMNS = (
    "CASE_STATUS",
    "DECISION_DATE",
    "VISA_CLASS",
    "JOB_TITLE",
    "EMPLOYER_NAME",
    "WORKSITE_STATE",
)
LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "lp",
    "llp",
    "pbc",
}

CAREER_KEYWORDS = (
    "career",
    "jobs",
    "job-openings",
    "open-roles",
    "open-positions",
    "openings",
    "join-us",
    "joinus",
    "work-with-us",
    "we-are-hiring",
    "hiring",
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "workable.com",
    "smartrecruiters.com",
    "bamboohr.com",
)


# ---------------------------------------------------------------------------
# 2. PYDANTIC SCHEMAS FOR STRUCTURED GEMINI OUTPUT
# ---------------------------------------------------------------------------
class TargetCompany(BaseModel):
    """Stage A: one cleaned company extracted from unstructured notes."""

    company_name: str = Field(description="Official name of the company.")
    inferred_url: str = Field(
        description=(
            "Best guess at the company's careers page or main website, "
            "as a full https:// URL."
        )
    )
    context_notes: str = Field(
        description="Any relevant notes from the raw input about this company."
    )


class RelatedTitles(BaseModel):
    """Stage A0: job titles that count as the same kind of role."""

    titles: List[str] = Field(
        description="Equivalent or closely related job titles as written in US job postings."
    )


class FitEvaluation(BaseModel):
    """Stage B: evaluation of crawled site text against the role and resume."""

    is_match: bool = Field(
        description="True if the company plausibly has roles fitting the candidate."
    )
    sponsorship_status: str = Field(
        description=(
            "One of: 'Proven sponsor (recent public records)', "
            "'Past sponsor (older records only)', "
            "'Sponsors (stated on site, no records found)', "
            "'Likely sponsors (indirect signals)', "
            "'Records show sponsorship, but site says no sponsorship', "
            "'Does not sponsor (stated on site)', 'No evidence found'."
        )
    )
    h1b_record_summary: str = Field(
        description=(
            "One or two sentences with the concrete numbers, fiscal years and "
            "job titles found in the public H-1B records, naming each source. "
            "'No public records found' if there are none."
        )
    )
    extracted_job_links: List[str] = Field(
        description="Full URLs of relevant job postings or careers pages."
    )
    summary: str = Field(
        description="2-4 sentence rationale covering technical fit and sponsorship evidence."
    )
    us_roles: str = Field(
        description=(
            "Whether relevant roles are located in the United States (including "
            "US-remote). One of: 'Yes', 'No', 'Unclear'."
        )
    )
    job_locations: List[str] = Field(
        description="Locations listed for the relevant roles, e.g. 'Austin, TX', 'Remote (US)'."
    )


# ---------------------------------------------------------------------------
# 3. RESUME PROCESSING
# ---------------------------------------------------------------------------
def extract_resume_text(uploaded_file) -> str:
    """Return plain text from an uploaded .txt or .pdf resume."""
    if uploaded_file is None:
        return ""

    file_name = (uploaded_file.name or "").lower()
    raw_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        page_texts = []
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")
        return "\n".join(page_texts).strip()

    try:
        return raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="ignore").strip()


# ---------------------------------------------------------------------------
# 4. SMALL HELPERS
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Make sure a model-inferred URL is usable by the crawler."""
    cleaned = (url or "").strip().strip("<>\"'")
    if not cleaned:
        return ""
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = "https://" + cleaned.lstrip("/")
    return cleaned


def normalize_us_roles(value: str) -> str:
    """Coerce the model's answer to exactly Yes / No / Unclear."""
    cleaned = (value or "").strip().lower()
    if cleaned.startswith("yes"):
        return "Yes"
    if cleaned.startswith("no"):
        return "No"
    return "Unclear"


def short_error(prefix: str, exc: Exception) -> str:
    """One-line, CSV-friendly error text."""
    message = " ".join(str(exc).split())
    if "Error: " in message:
        message = message.split("Error: ", 1)[1]
    return f"{prefix}: {message}"[:200]


def markdown_to_text(markdown_obj) -> str:
    """crawl4ai returns either a str subclass or an object with .raw_markdown."""
    if markdown_obj is None:
        return ""
    raw = getattr(markdown_obj, "raw_markdown", None)
    if raw:
        return str(raw)
    return str(markdown_obj)


def page_key(url: str) -> str:
    """Identity of a page ignoring www, locale prefixes, query and trailing slash.

    'https://www.site.com/fr/careers/?x=1' and 'https://site.com/careers'
    share a key, so translated copies of a page are not crawled twice.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    segments = [
        segment
        for segment in parsed.path.lower().split("/")
        if segment and not LOCALE_SEGMENT.match(segment)
    ]
    return host + "/" + "/".join(segments)


def find_career_links(result, base_url: str) -> List[str]:
    """Pull likely careers / job-board links out of a crawl result.

    Best candidates first: applicant-tracking boards, then job listing pages,
    then other careers pages.
    """
    links = getattr(result, "links", None) or {}
    seen = {page_key(base_url)}
    ranked: List[Tuple[int, str]] = []

    for bucket in ("internal", "external"):
        for item in links.get(bucket, []) or []:
            href = item.get("href", "") if isinstance(item, dict) else str(item)
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue

            # Judge the host and path only: tracking parameters such as
            # '?referer=/careers' must not make a login page look like a job page.
            target = (parsed.netloc + parsed.path).lower()
            if any(word in target for word in SKIP_LINK_WORDS):
                continue
            if not any(keyword in target for keyword in CAREER_KEYWORDS):
                continue

            key = page_key(absolute)
            if key in seen:
                continue
            seen.add(key)

            if any(domain in target for domain in ATS_DOMAINS):
                rank = 0
            elif any(word in target for word in JOB_LIST_WORDS):
                rank = 1
            else:
                rank = 2
            ranked.append((rank, absolute))

    ranked.sort(key=lambda pair: pair[0])
    return [link for _, link in ranked][:MAX_CANDIDATE_LINKS]


def parse_structured(response, expected_type):
    """Use the SDK's parsed object, falling back to validating the raw JSON."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed
    return TypeAdapter(expected_type).validate_json(response.text or "")


def core_company_name(name: str) -> str:
    """'Stripe, Inc.' -> 'Stripe'. Used to query employer-name databases."""
    cleaned = re.sub(r"\(.*?\)", " ", name or "")
    cleaned = re.sub(r"[.'\u2019]", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z&]+", " ", cleaned)
    tokens = cleaned.split()
    while len(tokens) > 1 and tokens[-1].lower() in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def page_mentions_company(page_text: str, company_name: str) -> bool:
    """Guard against lookup sites answering with a default / unrelated page."""
    tokens = core_company_name(company_name).lower().split()
    if not tokens:
        return False
    return tokens[0] in (page_text or "").lower()


def clean_role(role: str) -> str:
    """'Senior Backend Engineer (Python)' -> 'Senior Backend Engineer'."""
    return " ".join(re.sub(r"\(.*?\)", " ", role or "").split())


def split_titles(text: str) -> List[str]:
    """Split on new lines, commas, semicolons, tabs or '|' outside parentheses.

    'Backend Engineer (Python, Go), Full Stack Developer' gives two titles, so
    commas inside a parenthetical note do not break a title apart.
    """
    pieces: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text or "":
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        if depth == 0 and char in "\n\r,;|\t":
            pieces.append("".join(current))
            current = []
        else:
            current.append(char)
    pieces.append("".join(current))
    return pieces


def parse_titles(text: str) -> List[str]:
    """Titles from pasted text; bullets, numbering and duplicates removed."""
    titles: List[str] = []
    seen = set()
    for piece in split_titles(text):
        title = re.sub(r"^\s*(?:[-*\u2022\u00b7]|\d+[.)])\s*", "", piece)
        title = " ".join(title.split())
        if title and title.lower() not in seen:
            seen.add(title.lower())
            titles.append(title)
    return titles


def target_titles(target_role: str) -> List[str]:
    """Target roles as plain titles for matching ('(Python)' notes removed)."""
    titles: List[str] = []
    for title in parse_titles(target_role):
        cleaned = clean_role(title)
        if cleaned and cleaned.lower() not in {existing.lower() for existing in titles}:
            titles.append(cleaned)
    return titles


def normalize_title(title: str) -> str:
    lowered = (title or "").lower()
    for pattern, replacement in TITLE_EQUIVALENTS:
        lowered = re.sub(pattern, replacement, lowered)
    lowered = re.sub(r"[^a-z0-9+#]+", " ", lowered)
    return " ".join(lowered.split())


def title_matches(job_title: str, title_token_sets: List[set]) -> bool:
    """True when every word of some target title appears in the job title.

    'Software Developer' therefore matches 'Senior Software Developer, Payments'.
    """
    job_tokens = set(normalize_title(job_title).split())
    return any(tokens and tokens <= job_tokens for tokens in title_token_sets)


def looks_like_us_location(location: str) -> bool:
    return bool(US_LOCATION_PATTERN.search(location or ""))


def find_ats_boards(text: str) -> List[Tuple[str, str]]:
    """Find (ats, board token) pairs referenced in page HTML or links."""
    boards: List[Tuple[str, str]] = []
    for ats, pattern in ATS_BOARD_PATTERNS:
        for token in pattern.findall(text or ""):
            if token.lower() in NOT_BOARD_TOKENS:
                continue
            pair = (ats, token)
            if pair not in boards:
                boards.append(pair)
    return boards


def guess_board_tokens(company_name: str) -> List[str]:
    """Board names companies usually pick: 'grafanalabs', 'grafana-labs'."""
    words = re.sub(r"[^a-z0-9 ]+", "", core_company_name(company_name).lower()).split()
    guesses = ["".join(words), "-".join(words)] if words else []
    return [guess for index, guess in enumerate(guesses) if guess and guess not in guesses[:index]]


def join_location(*parts) -> str:
    return " / ".join(str(part) for part in parts if part)


async def fetch_board_jobs(http: httpx.AsyncClient, ats: str, token: str) -> Optional[List[dict]]:
    """Every open role on a public job board as {title, location, url}.

    Returns None when the board does not exist.
    """
    if ats == "Greenhouse":
        response = await http.get(GREENHOUSE_API.format(token=token))
    elif ats == "Lever":
        response = await http.get(LEVER_API.format(token=token))
    else:
        response = await http.get(ASHBY_API.format(token=token))
    if response.status_code != 200:
        return None
    data = response.json()

    jobs: List[dict] = []
    if ats == "Greenhouse":
        for job in data.get("jobs", []) or []:
            location = (job.get("location") or {}).get("name", "")
            jobs.append({"title": job.get("title", ""), "location": location, "url": job.get("absolute_url", "")})
    elif ats == "Lever":
        postings = data if isinstance(data, list) else data.get("data", []) or []
        for posting in postings:
            categories = posting.get("categories") or {}
            locations = categories.get("allLocations") or [categories.get("location")]
            workplace = posting.get("workplaceType")
            jobs.append(
                {
                    "title": posting.get("text", ""),
                    "location": join_location(
                        ", ".join(location for location in locations if location),
                        workplace if workplace and workplace != "unspecified" else "",
                        posting.get("country"),
                    ),
                    "url": posting.get("hostedUrl", ""),
                }
            )
    else:
        for job in data.get("jobs", []) or []:
            if job.get("isListed") is False:
                continue
            secondary = [item.get("location") for item in job.get("secondaryLocations") or []]
            country = (((job.get("address") or {}).get("postalAddress") or {}).get("addressCountry"))
            jobs.append(
                {
                    "title": job.get("title", ""),
                    "location": join_location(
                        ", ".join(location for location in [job.get("location"), *secondary] if location),
                        "Remote" if job.get("isRemote") else "",
                        country,
                    ),
                    "url": job.get("jobUrl", ""),
                }
            )
    return [job for job in jobs if job["title"]]


async def find_matching_roles(
    http: httpx.AsyncClient,
    company_name: str,
    discovered_boards: List[Tuple[str, str]],
    related_titles: List[str],
    usa_only: bool,
) -> Tuple[str, str, List[str]]:
    """Read the company's public job board(s) and keep roles matching the titles.

    Returns (block for Gemini, short summary for the table, matching job URLs).
    """
    candidates = [(ats, token, False) for ats, token in discovered_boards]
    if not candidates:
        # No board linked from the site: try the names companies usually use.
        for token in guess_board_tokens(company_name):
            for ats, _ in ATS_BOARD_PATTERNS:
                candidates.append((ats, token, True))

    title_token_sets = [set(normalize_title(title).split()) for title in related_titles]
    boards_read: List[str] = []
    all_jobs: List[dict] = []
    seen_urls = set()

    for ats, token, guessed in candidates:
        if len(boards_read) >= MAX_BOARDS_PER_COMPANY:
            break
        try:
            jobs = await fetch_board_jobs(http, ats, token)
        except Exception:
            continue
        if not jobs:
            continue
        boards_read.append(f"{ats} '{token}'" + (" (guessed name)" if guessed else ""))
        for job in jobs:
            if job["url"] not in seen_urls:
                seen_urls.add(job["url"])
                all_jobs.append(job)
        if guessed:
            break  # One guessed board is enough; more would risk another company's.

    if not boards_read:
        return "", "No public job board found", []

    matched = [job for job in all_jobs if title_matches(job["title"], title_token_sets)]
    if usa_only:
        matched.sort(key=lambda job: not looks_like_us_location(job["location"]))
    shown = matched[:MAX_MATCHED_ROLES]

    source = ", ".join(boards_read)
    summary = f"{len(matched)} of {len(all_jobs)} open roles match ({source})"
    lines = [
        f"Source: {source}. {len(all_jobs)} open roles in total; "
        f"{len(matched)} have a title matching the target titles."
    ]
    if shown:
        lines.append("Matching roles (title | location | url):")
        lines.extend(f"- {job['title']} | {job['location'] or 'location not stated'} | {job['url']}" for job in shown)
        if len(matched) > len(shown):
            lines.append(f"(and {len(matched) - len(shown)} more matching roles not listed)")
    return "\n".join(lines), summary, [job["url"] for job in matched if job["url"]]


# ---------------------------------------------------------------------------
# 5. PUBLIC H-1B RECORDS (LCA disclosures and sponsor databases)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dol_lca_file(path: str, modified_time: float) -> pd.DataFrame:
    """Load a DOL OFLC LCA disclosure file (.xlsx or .csv), keeping key columns.

    `modified_time` is only part of the cache key so an updated file reloads.
    """

    def wanted(column_name) -> bool:
        return str(column_name).strip().upper() in DOL_COLUMNS

    if path.lower().endswith(".csv"):
        frame = pd.read_csv(path, usecols=wanted, dtype=str, low_memory=False)
    else:
        frame = pd.read_excel(path, usecols=wanted, dtype=str)

    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if "EMPLOYER_NAME" not in frame.columns:
        raise ValueError("the file has no EMPLOYER_NAME column")

    frame["EMPLOYER_NAME"] = frame["EMPLOYER_NAME"].fillna("")
    keys = {name: core_company_name(name).upper() for name in frame["EMPLOYER_NAME"].unique()}
    frame["EMPLOYER_KEY"] = frame["EMPLOYER_NAME"].map(keys)
    return frame


def top_values(series: pd.Series, limit: int) -> str:
    counts = series.dropna().astype(str).str.strip().value_counts().head(limit)
    return ", ".join(f"{value} ({count})" for value, count in counts.items())


def summarize_dol_records(frame: pd.DataFrame, company_name: str) -> Tuple[str, int]:
    """Summarise the DOL LCA rows whose employer name matches the company."""
    key = core_company_name(company_name).upper()
    if not key:
        return "", 0

    employer_keys = frame["EMPLOYER_KEY"]
    matched = frame[(employer_keys == key) | employer_keys.str.startswith(key + " ")]
    if matched.empty:
        return "", 0

    lines = [f"Total LCA filings matched: {len(matched)}"]
    lines.append("Employer names matched: " + top_values(matched["EMPLOYER_NAME"], 6))
    if "VISA_CLASS" in matched.columns:
        lines.append("By visa class: " + top_values(matched["VISA_CLASS"], 5))
    if "CASE_STATUS" in matched.columns:
        lines.append("By case status: " + top_values(matched["CASE_STATUS"], 5))
    if "DECISION_DATE" in matched.columns:
        years = pd.to_datetime(matched["DECISION_DATE"], errors="coerce").dt.year
        years = years.dropna().astype(int)
        if not years.empty:
            lines.append("By decision year: " + top_values(years, 6))
    if "JOB_TITLE" in matched.columns:
        lines.append("Top job titles: " + top_values(matched["JOB_TITLE"], 10))
    if "WORKSITE_STATE" in matched.columns:
        lines.append("Top worksite states: " + top_values(matched["WORKSITE_STATE"], 6))
    return "\n".join(lines), len(matched)


async def search_h1b_records(client, company_name: str) -> Tuple[str, List[str]]:
    """Ask Gemini to run live Google searches for the employer's H-1B history.

    Returns (findings text, source site names). The text is discarded unless
    the answer is actually backed by search results.
    """
    prompt = (
        f"Use Google Search to find public H-1B sponsorship records for the US "
        f"employer '{company_name}'.\n"
        "Look for: Labor Condition Application (LCA) filings from Department of "
        "Labor disclosure data; USCIS H-1B Employer Data Hub approvals and "
        "denials; PERM / green card filings; and sponsor profiles on sites such "
        "as h1bgrader.com, myvisajobs.com, h1bdata.info, h1bscan.com and "
        "similar public trackers.\n"
        "Report only what the search results state: counts by fiscal year, "
        "approval or denial numbers, typical job titles and salaries, and the "
        "legal employer names the filings are under. Name the site each figure "
        "comes from. If the results show nothing for this employer, say so "
        "plainly. Do not answer from memory and do not estimate."
    )
    response = await client.aio.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
            automatic_function_calling=NO_AUTO_FUNCTION_CALLING,
        ),
    )

    sources: List[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            title = getattr(web, "title", None)
            if title and title not in sources:
                sources.append(title)

    if not sources:
        return "", []
    return (response.text or "").strip()[:MAX_SEARCH_CHARS], sources


async def gather_h1b_evidence(
    company: "TargetCompany",
    crawler,
    settings: "CrawlSettings",
    client,
    records_semaphore: asyncio.Semaphore,
    use_lookup_sites: bool,
    use_search: bool,
    dol_frame: Optional[pd.DataFrame],
) -> Tuple[str, List[str], List[str]]:
    """Collect public H-1B records for one employer from every enabled source.

    Returns (evidence text for Gemini, per-source outcome notes, lookup URLs).
    Never raises: a failing source is just reported as such.
    """
    name = company.company_name
    core_name = core_company_name(name)
    sections: List[str] = []
    outcomes: List[str] = []
    lookup_links: List[str] = []

    if use_lookup_sites and core_name:
        slug = re.sub(r"[^a-z0-9]+", "-", core_name.lower()).strip("-")
        lookups = [
            ("h1bdata.info", H1BDATA_URL.format(query=quote_plus(core_name))),
            ("myvisajobs.com", MYVISAJOBS_URL.format(slug=slug)),
        ]
        for site, lookup_url in lookups:
            lookup_links.append(lookup_url)
            async with records_semaphore:
                # Checked while holding the semaphore so waiting tasks see blocks
                # recorded by the ones ahead of them.
                if settings.blocked_counts.get(site, 0) >= BLOCKS_BEFORE_SKIPPING_SITE:
                    outcomes.append(f"{site}: skipped (site blocks automated access)")
                    continue
                try:
                    page_text, _, _ = await crawl_page(crawler, lookup_url, settings.stealth)
                except Exception as exc:
                    if is_anti_bot_error(exc):
                        settings.blocked_counts[site] = settings.blocked_counts.get(site, 0) + 1
                        outcomes.append(f"{site}: blocked (anti-bot)")
                    else:
                        outcomes.append(f"{site}: unreachable")
                    continue
                settings.blocked_counts[site] = 0
                await asyncio.sleep(1)
            if page_text.strip() and page_mentions_company(page_text, name):
                sections.append(f"### SOURCE: {site} ({lookup_url})\n{page_text}")
                outcomes.append(f"{site}: page found")
            else:
                outcomes.append(f"{site}: no page for this employer")

    if use_search:
        try:
            findings, search_sources = await search_h1b_records(client, name)
        except Exception:
            outcomes.append("Google Search: failed")
        else:
            if findings:
                sections.append(
                    "### SOURCE: Google Search results (via "
                    + ", ".join(search_sources[:8])
                    + ")\n"
                    + findings
                )
                outcomes.append(f"Google Search: {len(search_sources)} sources")
            else:
                outcomes.append("Google Search: no results")

    if dol_frame is not None:
        try:
            dol_text, dol_count = await asyncio.to_thread(summarize_dol_records, dol_frame, name)
        except Exception:
            outcomes.append("DOL file: lookup failed")
        else:
            if dol_count:
                sections.append(
                    "### SOURCE: DOL OFLC LCA disclosure file (official)\n" + dol_text
                )
                outcomes.append(f"DOL file: {dol_count} LCAs")
            else:
                outcomes.append("DOL file: 0 LCAs")

    return "\n\n".join(sections), outcomes, lookup_links


# ---------------------------------------------------------------------------
# 6. GEMINI CALLS
# ---------------------------------------------------------------------------
async def generate_structured(client, prompt: str, schema, expected_type):
    """Call Gemini with an enforced response_schema, with simple retry/backoff."""
    last_error: Optional[Exception] = None

    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,
                    automatic_function_calling=NO_AUTO_FUNCTION_CALLING,
                ),
            )
            return parse_structured(response, expected_type)
        except Exception as exc:
            last_error = exc
            if attempt < GEMINI_MAX_ATTEMPTS - 1:
                await asyncio.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Gemini request failed: {last_error}")


async def stage_a_clean_targets(client, raw_notes: str, target_role: str) -> List[TargetCompany]:
    """Stage A: unstructured notes -> list of TargetCompany."""
    prompt = (
        "You are a data-cleaning assistant for a job search.\n"
        "The text below is a messy dump of notes about companies. Extract every "
        "distinct company mentioned.\n"
        "For each company return:\n"
        "- company_name: the official company name.\n"
        "- inferred_url: if the notes contain a URL for the company, use it. "
        "Otherwise infer the most likely careers page (or main website if unsure) "
        "as a full https:// URL.\n"
        "- context_notes: a short summary of whatever the notes say about it "
        "(empty string if nothing).\n"
        "Do not invent companies that are not in the notes. Deduplicate.\n\n"
        f"Target role (for context only): {target_role}\n\n"
        "RAW NOTES:\n"
        f"{raw_notes}"
    )
    companies = await generate_structured(
        client, prompt, list[TargetCompany], List[TargetCompany]
    )
    return list(companies or [])


async def expand_role_titles(client, target_role: str, resume_text: str) -> List[str]:
    """Stage A0: 'Software Developer' -> Full Stack Engineer, Software Engineer, ..."""
    resume_block = resume_text[:3000] if resume_text else "(no resume provided)"
    prompt = (
        "You help a job seeker search company job boards.\n"
        f"Target role(s): {target_role}\n\n"
        "Candidate resume excerpt:\n"
        f"{resume_block}\n\n"
        "List 8 to 15 job titles that US employers use for the same kind of role, "
        "including the target role(s) themselves, common synonyms and adjacent titles "
        "the candidate is qualified for (for 'Software Developer': Software "
        "Engineer, Full Stack Engineer, Full Stack Developer, Backend Engineer, "
        "Application Developer, ...).\n"
        "Rules: write each title without seniority words (no Senior, Staff, "
        "Lead, I, II) because seniority is matched separately; keep titles short "
        "(2-4 words); do not include unrelated roles such as management, sales "
        "or recruiting unless the target role is one."
    )
    result = await generate_structured(client, prompt, RelatedTitles, RelatedTitles)
    titles = parse_titles("\n".join([*target_titles(target_role), *(result.titles or [])]))
    return titles[:25]


def suggest_titles_now(api_key: str, target_role: str, resume_text: str) -> List[str]:
    """Synchronous wrapper used by the 'Suggest related titles' button."""

    async def run() -> List[str]:
        client = genai.Client(api_key=api_key)
        return await expand_role_titles(client, target_role, resume_text)

    return asyncio.run(run())


async def stage_b_evaluate_fit(
    client,
    company: TargetCompany,
    target_role: str,
    resume_text: str,
    crawled_text: str,
    candidate_links: List[str],
    usa_only: bool,
    h1b_evidence: str,
    related_titles: List[str],
    job_board_block: str,
) -> FitEvaluation:
    """Stage B: crawled text + public H-1B records -> FitEvaluation."""
    records_block = h1b_evidence if h1b_evidence else "(no public records were found or checked)"
    links_block = "\n".join(candidate_links) if candidate_links else "(none discovered)"
    resume_block = resume_text[:MAX_RESUME_CHARS] if resume_text else "(no resume provided)"

    prompt = (
        "You are a technical recruiter and immigration-aware career advisor.\n"
        "Evaluate the company below for a candidate, using ONLY the crawled "
        "website text, the links and the public H-1B records provided.\n\n"
        f"TARGET ROLE: {target_role}\n"
        "TITLES THAT COUNT AS THE TARGET ROLE (any seniority): "
        f"{', '.join(related_titles) if related_titles else target_role}\n\n"
        f"COMPANY: {company.company_name}\n"
        f"USER NOTES: {company.context_notes}\n\n"
        "CANDIDATE RESUME:\n"
        f"{resume_block}\n\n"
        "LINKS DISCOVERED ON THE SITE:\n"
        f"{links_block}\n\n"
        "COMPANY JOB BOARD (complete list of open roles, filtered to the titles above):\n"
        f"{job_board_block or '(no public job board was found; rely on the website text)'}\n\n"
        "CRAWLED WEBSITE TEXT (truncated):\n"
        f"{crawled_text}\n\n"
        "PUBLIC H-1B RECORDS (LCA disclosures, sponsor databases, search results):\n"
        f"{records_block}\n\n"
        "Instructions:\n"
        "- is_match: true if an open role whose title matches any of the titles "
        "that count as the target role (any seniority) is listed on the job "
        "board or the website and plausibly fits the resume. Without a job "
        "list, true only if the site shows teams or a tech stack that plausibly "
        "fit the target role and the resume.\n"
        "- sponsorship_status: weigh two kinds of evidence. (1) The public H-1B "
        "records: count only filings whose employer name is this company or one "
        "of its legal entities, and ignore any record page that is about a "
        "different employer. An LCA is filed by the employer before an H-1B "
        "petition, so certified LCAs show intent to sponsor, not approved "
        "visas. (2) The company's own site: markers such as 'visa sponsorship', "
        "'H-1B', 'work authorization', 'immigration support', or exclusions "
        "such as 'must be authorized to work without sponsorship', 'US citizens "
        "only', 'security clearance required'. Answer with exactly one of: "
        "'Proven sponsor (recent public records)' when records show filings in "
        "the last three fiscal years; 'Past sponsor (older records only)'; "
        "'Sponsors (stated on site, no records found)'; 'Likely sponsors "
        "(indirect signals)'; 'Records show sponsorship, but site says no "
        "sponsorship'; 'Does not sponsor (stated on site)'; 'No evidence "
        "found'. Never guess beyond the evidence.\n"
        "- h1b_record_summary: the concrete numbers, fiscal years and job titles "
        "from the public records, naming each source, and whether any filed job "
        "titles resemble the target role. 'No public records found' if none.\n"
        "- extracted_job_links: full URLs to relevant postings or careers pages, "
        "preferring the best-fitting roles from the job board list. "
        "Only use URLs that appear in the discovered links or the crawled text; "
        "never fabricate URLs.\n"
        "- summary: 2-4 sentences citing the specific evidence for fit and "
        "sponsorship.\n"
        "- job_locations: the locations stated for the relevant roles (city/state, "
        "country, or remote region). Empty list if the site does not say.\n"
        "- us_roles: 'Yes' if at least one relevant role is based in the United "
        "States or is remote and open to US residents; 'No' if the site states "
        "locations and none are in the United States; 'Unclear' if locations are "
        "not stated. Never guess from the company's headquarters alone."
    )
    if usa_only:
        prompt += (
            "\n\nLOCATION FILTER - USA ONLY: the candidate only wants roles based "
            "in the United States (US-remote counts). Ignore roles located "
            "elsewhere: is_match must be false unless a fitting role is US-based "
            "or its location is not stated, and extracted_job_links must not "
            "include postings that are clearly outside the United States."
        )
    return await generate_structured(client, prompt, FitEvaluation, FitEvaluation)


# ---------------------------------------------------------------------------
# 7. ASYNC CRAWLING (crawl4ai)
# ---------------------------------------------------------------------------
class CrawlSettings(NamedTuple):
    """Crawl configs plus per-run state shared by every company task."""

    normal: CrawlerRunConfig
    stealth: CrawlerRunConfig
    blocked_counts: Dict[str, int]  # lookup site -> anti-bot blocks so far


def is_anti_bot_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ANTI_BOT_MARKERS)


async def crawl_with_fallback(
    crawler, url: str, settings: CrawlSettings
) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Crawl normally; if the site's bot protection blocks it, retry in stealth mode."""
    try:
        return await crawl_page(crawler, url, settings.normal)
    except Exception as exc:
        if not is_anti_bot_error(exc):
            raise
    return await crawl_page(crawler, url, settings.stealth)


async def crawl_page(
    crawler, url: str, run_config
) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Crawl one page.

    Returns (truncated markdown, career-like links, job boards referenced).
    """
    result = await crawler.arun(url=url, config=run_config)

    if not getattr(result, "success", False):
        message = getattr(result, "error_message", None) or "crawl was not successful"
        raise RuntimeError(str(message))

    markdown_text = markdown_to_text(getattr(result, "markdown", None))
    markdown_text = markdown_text[:12000]
    career_links = find_career_links(result, url)

    # Job boards are often embedded via iframe/script, so scan the raw HTML too.
    links = getattr(result, "links", None) or {}
    hrefs = [
        item.get("href", "") if isinstance(item, dict) else str(item)
        for bucket in ("internal", "external")
        for item in links.get(bucket, []) or []
    ]
    boards = find_ats_boards("\n".join(hrefs) + "\n" + (getattr(result, "html", "") or ""))
    return markdown_text, career_links, boards


async def process_single_target(
    company: TargetCompany,
    crawler,
    settings: CrawlSettings,
    client,
    target_role: str,
    resume_text: str,
    usa_only: bool,
    semaphore: asyncio.Semaphore,
    records_semaphore: asyncio.Semaphore,
    use_lookup_sites: bool,
    use_search: bool,
    dol_frame: Optional[pd.DataFrame],
    http: httpx.AsyncClient,
    related_titles: List[str],
) -> dict:
    """Crawl + evaluate one company. Never raises; errors land in the row."""
    url = normalize_url(company.inferred_url)
    row = {
        "Company": company.company_name,
        "URL": url,
        "Match": False,
        "Sponsorship Status": "Unknown",
        "Matching Openings": "",
        "H-1B Records": "",
        "H-1B Sources": "",
        "US Roles": "Unclear",
        "Locations": "",
        "Job Links": "",
        "Summary": "",
        "H-1B Lookup Links": "",
        "Notes": company.context_notes,
        "Status": "OK",
    }

    # --- Company website -----------------------------------------------------
    crawled_sections: List[str] = []
    career_links: List[str] = []
    boards: List[Tuple[str, str]] = []
    site_problem = ""

    if not url:
        site_problem = "No URL could be inferred"
    else:
        async with semaphore:
            try:
                home_markdown, career_links, boards = await crawl_with_fallback(
                    crawler, url, settings
                )
            except Exception as exc:
                site_problem = short_error("Site crawl failed", exc)
            else:
                crawled_sections.append(f"### PAGE: {url}\n{home_markdown}")

                # Follow one careers / job-board link if the landing page has one.
                follow_url = next(
                    (link for link in career_links if link.rstrip("/") != url.rstrip("/")),
                    None,
                )
                if follow_url:
                    try:
                        careers_markdown, more_links, more_boards = await crawl_with_fallback(
                            crawler, follow_url, settings
                        )
                    except Exception:
                        pass  # The landing page alone is still usable.
                    else:
                        crawled_sections.append(f"### PAGE: {follow_url}\n{careers_markdown}")
                        for link in more_links:
                            if link not in career_links:
                                career_links.append(link)
                        for board in more_boards:
                            if board not in boards:
                                boards.append(board)

    # The careers link itself may point at a job board (e.g. boards.greenhouse.io/x).
    for board in find_ats_boards("\n".join(career_links)):
        if board not in boards:
            boards.append(board)

    # --- Complete list of open roles from the public job board ---------------
    try:
        job_board_block, openings_summary, matched_urls = await find_matching_roles(
            http, company.company_name, boards, related_titles, usa_only
        )
    except Exception as exc:
        job_board_block, openings_summary, matched_urls = "", short_error("Job board failed", exc), []
    row["Matching Openings"] = openings_summary

    crawled_text = "\n\n".join(crawled_sections)
    if not crawled_text.strip() and not site_problem:
        site_problem = "Site crawl returned no text"

    # --- Public H-1B records (independent of the company website) ------------
    h1b_evidence, outcomes, lookup_links = await gather_h1b_evidence(
        company,
        crawler,
        settings,
        client,
        records_semaphore,
        use_lookup_sites,
        use_search,
        dol_frame,
    )
    row["H-1B Sources"] = " | ".join(outcomes)
    row["H-1B Lookup Links"] = " | ".join(lookup_links)

    if site_problem and not h1b_evidence and not job_board_block:
        row["Status"] = site_problem
        return row

    # --- Gemini evaluation -----------------------------------------------------
    try:
        evaluation = await stage_b_evaluate_fit(
            client,
            company,
            target_role,
            resume_text,
            crawled_text if crawled_text.strip() else "(the company website could not be crawled)",
            career_links[:MAX_CANDIDATE_LINKS],
            usa_only,
            h1b_evidence,
            related_titles,
            job_board_block,
        )
    except Exception as exc:
        row["Status"] = short_error("Evaluation failed", exc)
        return row

    row["Match"] = bool(evaluation.is_match)
    row["Sponsorship Status"] = evaluation.sponsorship_status
    row["H-1B Records"] = evaluation.h1b_record_summary
    job_links = list(evaluation.extracted_job_links or []) or matched_urls[:5]
    row["Job Links"] = " | ".join(job_links)
    row["Summary"] = evaluation.summary
    row["US Roles"] = normalize_us_roles(evaluation.us_roles)
    row["Locations"] = " | ".join(evaluation.job_locations or [])
    if site_problem:
        row["Status"] = site_problem + " (H-1B records only)"
    return row


async def process_all_targets(
    api_key: str,
    raw_notes: str,
    target_role: str,
    resume_text: str,
    usa_only: bool,
    max_concurrency: int,
    use_lookup_sites: bool,
    use_search: bool,
    dol_frame: Optional[pd.DataFrame],
    related_titles: List[str],
    on_status: Callable[[str], None],
    on_progress: Callable[[int, int, str], None],
) -> Tuple[List[dict], List[str]]:
    """Full pipeline: titles -> Stage A cleanup -> concurrent crawl -> Stage B.

    Returns (result rows, the titles that were matched against).
    """
    # Created inside the running loop so the async client binds to it.
    client = genai.Client(api_key=api_key)

    titles = list(related_titles)
    if not titles:
        on_status("Expanding your target role into related job titles...")
        try:
            titles = await expand_role_titles(client, target_role, resume_text)
        except Exception:
            titles = target_titles(target_role)
    else:
        # Every target role is always matched, even if it is not in the box.
        existing = {title.lower() for title in titles}
        missing = [title for title in target_titles(target_role) if title.lower() not in existing]
        titles = missing + titles

    on_status("Stage A: structuring your company notes with Gemini...")
    companies = await stage_a_clean_targets(client, raw_notes, target_role)
    if not companies:
        return [], titles

    on_status(
        f"Stage B: crawling {len(companies)} companies and checking public H-1B records..."
    )
    browser_config = BrowserConfig(headless=True, verbose=False)
    settings = CrawlSettings(
        normal=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=PAGE_TIMEOUT_MS,
            verbose=False,
        ),
        # Used for sites that challenge automated browsers.
        stealth=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=PAGE_TIMEOUT_MS,
            verbose=False,
            magic=True,
            simulate_user=True,
            override_navigator=True,
            delay_before_return_html=3.0,
        ),
        blocked_counts={},
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    records_semaphore = asyncio.Semaphore(H1B_RECORDS_CONCURRENCY)
    total = len(companies)
    completed = 0
    rows: List[dict] = []

    http = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh) JobSearchUtilityTool/1.0"},
    )
    async with http, AsyncWebCrawler(config=browser_config) as crawler:
        tasks = [
            asyncio.create_task(
                process_single_target(
                    company,
                    crawler,
                    settings,
                    client,
                    target_role,
                    resume_text,
                    usa_only,
                    semaphore,
                    records_semaphore,
                    use_lookup_sites,
                    use_search,
                    dol_frame,
                    http,
                    titles,
                )
            )
            for company in companies
        ]
        for finished in asyncio.as_completed(tasks):
            row = await finished
            rows.append(row)
            completed += 1
            on_progress(completed, total, row["Company"])

    # Restore the order the companies were extracted in.
    order = {company.company_name: index for index, company in enumerate(companies)}
    rows.sort(key=lambda item: order.get(item["Company"], len(order)))
    return rows, titles


# ---------------------------------------------------------------------------
# 8. STREAMLIT UI
# ---------------------------------------------------------------------------
def suggest_titles_callback(api_key: str) -> None:
    """Fill the related-titles box. Runs before the rerun, so it may set widget state."""
    role = "; ".join(parse_titles(st.session_state.get("target_role", "")))
    if not api_key or not role:
        st.session_state["titles_error"] = "Enter a target role and a Gemini API key first."
        return
    resume_text = ""
    resume_file = st.session_state.get("resume_file")
    if resume_file is not None:
        try:
            resume_text = extract_resume_text(resume_file)
        except Exception:
            resume_text = ""
    try:
        titles = suggest_titles_now(api_key, role, resume_text)
    except Exception as exc:
        st.session_state["titles_error"] = short_error("Could not suggest titles", exc)
        return
    st.session_state["related_titles"] = "\n".join(titles)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Paste messy company notes, upload a resume, and let the pipeline crawl "
        "each company for technical fit, then check public H-1B records for "
        "its sponsorship history."
    )

    # --- Sidebar: API key + settings -------------------------------------
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    with st.sidebar:
        st.header("Settings")
        if not api_key:
            st.warning("GEMINI_API_KEY not found in environment or .env")
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Used for this session only; it is not written to disk.",
            ).strip()
        st.text_input("Model", value=MODEL_NAME, disabled=True)
        max_concurrency = st.slider(
            "Concurrent crawls",
            min_value=1,
            max_value=8,
            value=3,
            help="Higher is faster but uses more memory and may hit rate limits.",
        )

        st.subheader("H-1B evidence sources")
        use_lookup_sites = st.checkbox(
            "Sponsor databases (h1bdata.info, myvisajobs.com)",
            value=True,
            help="Crawls each employer's page on these public LCA databases.",
        )
        use_search = st.checkbox(
            "Live Google Search (via Gemini)",
            value=True,
            help=(
                "Gemini searches the web for the employer's LCA, USCIS and PERM "
                "records on any public site (h1bgrader, h1bscan and others). "
                "Uses Gemini's Google Search tool, which is billed separately."
            ),
        )
        dol_path = st.text_input(
            "DOL LCA disclosure file (optional)",
            placeholder="/Users/you/Downloads/LCA_Disclosure_Data_FY2026_Q3.xlsx",
            help=(
                "Full path to an official LCA disclosure file (.xlsx or .csv) "
                "downloaded from the Department of Labor. The first load of an "
                ".xlsx can take a few minutes."
            ),
        ).strip().strip("\"'")
        st.caption(f"[Download DOL disclosure files]({DOL_PERFORMANCE_URL})")

    # --- Inputs ------------------------------------------------------------
    col_left, col_right = st.columns(2)

    # Titles generated during the last run are filled in before the box is drawn.
    if "pending_titles" in st.session_state:
        st.session_state["related_titles"] = st.session_state.pop("pending_titles")

    with col_left:
        target_role_text = st.text_area(
            "Target Role(s)",
            key="target_role",
            height=100,
            placeholder=(
                "One role, or several separated by new lines or commas:\n"
                "Software Developer\nFull Stack Engineer, Backend Engineer"
            ),
            help=(
                "Paste titles straight from the related-titles box. New lines, "
                "commas, semicolons and '|' all separate titles; commas inside "
                "brackets, e.g. 'Backend Engineer (Python, Go)', do not."
            ),
        )
        target_roles = parse_titles(target_role_text)
        target_role = "; ".join(target_roles)
        if len(target_roles) > 1:
            st.caption(f"Read as {len(target_roles)} roles: " + " · ".join(target_roles))
        st.button(
            "Suggest related titles",
            on_click=suggest_titles_callback,
            args=(api_key,),
            help="Gemini lists equivalent titles (e.g. Full Stack Engineer). Edit them freely.",
        )
        titles_error = st.session_state.pop("titles_error", None)
        if titles_error:
            st.warning(titles_error)
        titles_text = st.text_area(
            "Related titles to match (one per line)",
            key="related_titles",
            height=150,
            placeholder=(
                "Leave empty to generate them automatically at launch, or type "
                "your own:\nSoftware Engineer\nFull Stack Engineer\nFull Stack Developer"
            ),
            help=(
                "Job-board roles are matched against these titles, ignoring seniority "
                "words: 'Software Developer' also matches 'Senior Software Developer II'."
            ),
        )
        resume_file = st.file_uploader(
            "Resume (.pdf or .txt)", type=["pdf", "txt"], key="resume_file"
        )
        usa_only = st.checkbox(
            "USA locations only",
            value=True,
            help=(
                "Only count roles based in the United States (US-remote included) "
                "and drop companies whose roles are all outside the US."
            ),
        )

    with col_right:
        raw_notes = st.text_area(
            "Company notes (unstructured)",
            height=320,
            placeholder=(
                "Dump anything here: company names, URLs, recruiter messages, "
                "random notes...\n\n"
                "stripe - payments infra, friend works there\n"
                "databricks.com/company/careers\n"
                "that robotics startup in Pittsburgh, Carnegie Robotics?"
            ),
        )

    launch = st.button(
        "Launch AI Agent Pipeline",
        type="primary",
        width="stretch",
    )

    # --- Pipeline ------------------------------------------------------------
    if launch:
        problems = []
        if not api_key:
            problems.append("Add a Gemini API key (in .env or the sidebar).")
        if not target_role.strip():
            problems.append("Enter a target role.")
        if not raw_notes.strip():
            problems.append("Paste some company notes.")

        resume_text = ""
        if resume_file is not None:
            try:
                resume_text = extract_resume_text(resume_file)
            except Exception as exc:
                problems.append(f"Could not read the resume: {exc}")
            else:
                if not resume_text:
                    st.warning(
                        "No text could be extracted from the resume (it may be a "
                        "scanned PDF). Continuing with the target role only."
                    )
        else:
            st.info("No resume uploaded. Fit will be judged on the target role only.")

        dol_frame = None
        if dol_path and not problems:
            dol_file = Path(dol_path).expanduser()
            if not dol_file.is_file():
                problems.append(f"DOL file not found: {dol_file}")
            else:
                try:
                    with st.spinner("Loading the DOL LCA disclosure file..."):
                        dol_frame = load_dol_lca_file(str(dol_file), dol_file.stat().st_mtime)
                except Exception as exc:
                    problems.append(f"Could not read the DOL file: {exc}")

        if problems:
            for problem in problems:
                st.error(problem)
        else:
            status_box = st.empty()
            progress_bar = st.progress(0.0)

            def on_status(message: str) -> None:
                status_box.info(message)

            def on_progress(done: int, total: int, company_name: str) -> None:
                fraction = done / total if total else 1.0
                progress_bar.progress(fraction, text=f"{done}/{total} done - {company_name}")

            entered_titles = parse_titles(titles_text)
            try:
                rows, titles_used = asyncio.run(
                    process_all_targets(
                        api_key=api_key,
                        raw_notes=raw_notes,
                        target_role=target_role,
                        resume_text=resume_text,
                        usa_only=usa_only,
                        max_concurrency=max_concurrency,
                        use_lookup_sites=use_lookup_sites,
                        use_search=use_search,
                        dol_frame=dol_frame,
                        related_titles=entered_titles,
                        on_status=on_status,
                        on_progress=on_progress,
                    )
                )
            except Exception as exc:
                progress_bar.empty()
                status_box.error(f"Pipeline failed: {exc}")
            else:
                progress_bar.empty()
                if rows:
                    message = f"Finished. Evaluated {len(rows)} companies."
                    if usa_only:
                        kept = [row for row in rows if row["US Roles"] != "No"]
                        dropped = [row["Company"] for row in rows if row["US Roles"] == "No"]
                        if dropped:
                            message += (
                                f" USA filter removed {len(dropped)} with no US roles: "
                                + ", ".join(dropped)
                                + "."
                            )
                        rows = kept
                    status_box.success(message)
                    st.session_state["results_df"] = pd.DataFrame(rows)
                    st.session_state["run_message"] = message
                else:
                    status_box.warning("No companies could be identified in your notes.")
                    st.session_state.pop("results_df", None)
                    st.session_state.pop("run_message", None)
                st.session_state["titles_used"] = titles_used
                if not entered_titles and titles_used:
                    # Show the generated titles in the box so they can be edited.
                    st.session_state["pending_titles"] = "\n".join(titles_used)
                    st.rerun()

    # --- Results (kept in session_state so downloads don't clear them) -------
    results_df = st.session_state.get("results_df")
    if results_df is not None and not results_df.empty:
        st.subheader("Results")
        if st.session_state.get("run_message"):
            st.success(st.session_state["run_message"])
        if st.session_state.get("titles_used"):
            st.caption("Matched against titles: " + ", ".join(st.session_state["titles_used"]))

        match_count = int(results_df["Match"].sum())
        failed_count = int((results_df["Status"] != "OK").sum())
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Companies", len(results_df))
        metric_b.metric("Matches", match_count)
        metric_c.metric("Failed / skipped", failed_count)

        st.dataframe(
            results_df,
            width="stretch",
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("URL"),
                "Match": st.column_config.CheckboxColumn("Match"),
                "Summary": st.column_config.TextColumn("Summary", width="large"),
                "Matching Openings": st.column_config.TextColumn("Matching Openings", width="medium"),
                "H-1B Records": st.column_config.TextColumn("H-1B Records", width="large"),
            },
        )

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_bytes,
            file_name="job_search_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
