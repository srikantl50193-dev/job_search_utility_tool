# Job Search Utility Tool

A local Streamlit app for macOS that turns a messy list of companies into a ranked shortlist. For each company it checks whether there are open roles that fit your target titles and resume, and what public records say about its **H-1B visa sponsorship history**.

You paste your notes, upload your resume and list the roles you want. The app then:

1. Uses Gemini to clean your notes into a list of companies with their careers URLs.
2. Expands your target role into related titles (for example *Software Developer* → *Software Engineer*, *Full Stack Engineer*, *Backend Engineer* …).
3. Crawls each company's careers pages and reads the **complete** list of open roles from its public job board (Greenhouse, Lever or Ashby) when it has one.
4. Collects public H-1B evidence: LCA databases, a live Google Search via Gemini and, optionally, the official Department of Labor disclosure file.
5. Asks Gemini to judge fit and sponsorship, and shows everything in a table you can download as CSV.

> **Not legal or immigration advice.** The results summarise public records and company web pages. A certified LCA shows that an employer intended to sponsor; it is not an approved visa. Confirm sponsorship with the employer before relying on it.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Using the app](#using-the-app)
- [Understanding the results](#understanding-the-results)
- [How it works](#how-it-works)
- [Tuning](#tuning)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- macOS (also works on Linux and Windows)
- Python 3.10 or newer
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))
- About 500 MB of disk space for the headless Chromium browser that crawl4ai installs

| Package | Used for |
|---|---|
| `streamlit` | User interface |
| `crawl4ai` (0.9+) | Asynchronous headless-browser crawling |
| `google-genai` | Gemini SDK (`from google import genai`) |
| `pydantic` | Structured-output schemas |
| `pypdf` | Reading PDF resumes |
| `pandas` | Results table, CSV export, DOL file processing |
| `python-dotenv` | Loading `GEMINI_API_KEY` from `.env` |
| `httpx` | Public job-board APIs (installed with `google-genai`) |
| `openpyxl` | *Optional:* reading the DOL `.xlsx` disclosure file |

---

## Installation

```bash
# 1. Create a project folder containing app.py, then open a terminal in it
cd job_search_utility_tool

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the dependencies
pip install streamlit crawl4ai google-genai pydantic pypdf pandas python-dotenv httpx openpyxl

# 4. One-time crawl4ai setup (downloads the headless browser)
crawl4ai-setup

# 5. Optional: faster reloads while editing
pip install watchdog
```

If `crawl4ai-setup` reports problems, run `crawl4ai-doctor` to diagnose them.

---

## Configuration

Create a file named `.env` in the same folder as `app.py`:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

If the key isn't found, the sidebar shows a warning and a password box where you can paste it for the current session. A key entered there is not written to disk.

Add `.env` to `.gitignore` if the folder is under version control.

The app uses the model `gemini-3.8-flash`, set by `MODEL_NAME` near the top of `app.py`.

---

## Running the app

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit opens the app at <http://localhost:8501>. Stop it with `Ctrl+C` in the terminal.

---

## Using the app

### 1. Target Role(s)

Enter one role, or several separated by **new lines, commas, semicolons, tabs or `|`**. Bullets (`-`, `•`) and numbering (`1.`, `2)`) are removed automatically, and so are duplicates.

```text
Software Developer
Full Stack Engineer, Backend Engineer (Python, Go)
```

- Commas inside brackets don't split a title, so `Backend Engineer (Python, Go)` stays one role. The bracketed note is passed to Gemini as context but ignored when matching job titles.
- When you enter more than one role, a line under the box confirms how it was read: *Read as 3 roles: …*
- A seniority word you type is treated as part of the title. **Senior** Backend Engineer matches only senior roles; Backend Engineer matches every level.

### 2. Related titles to match

Click **Suggest related titles** and Gemini fills this box with equivalent titles, using your resume if one is uploaded. Edit the list freely; one title per line. You can copy titles from here into Target Role(s).

If you leave the box empty, titles are generated automatically when you launch, and filled into the box afterwards so you can adjust them for the next run.

Matching rules:

- A job matches when **every word** of one of your titles appears in the job title. *Software Developer* matches *Senior Software Developer II, Payments*.
- Spelling variants are treated as the same word: *Full-Stack / Full Stack / Fullstack*, *Back-End / Backend*, *Front-End / Frontend*, *Sr. / Senior*, *SWE*, *SDE*.
- Every target role is always matched, even if it's missing from this box.

### 3. Resume

Upload a `.pdf` or `.txt` resume. Text is extracted with `pypdf`. A scanned (image-only) PDF yields no text; the app warns you and judges fit on the target roles alone.

### 4. USA locations only

Ticked by default. Gemini then counts only roles based in the United States (US-remote included), and companies whose roles are all outside the US are removed from the results. Companies whose sites don't state a location are kept and marked *Unclear*.

### 5. Company notes

Paste anything: company names, URLs, recruiter messages, half-remembered notes.

```text
stripe - payments infra, friend works there
databricks.com/company/careers
that robotics startup in Pittsburgh, Carnegie Robotics?
```

Gemini extracts each distinct company, infers its careers URL (or uses one you gave) and keeps your notes about it.

### 6. Sidebar settings

| Setting | Default | What it does |
|---|---|---|
| Concurrent crawls | 3 | Companies crawled at once. Higher is faster but uses more memory and API quota. |
| Sponsor databases | On | Crawls each employer's page on h1bdata.info and myvisajobs.com. |
| Live Google Search (via Gemini) | On | Gemini searches the web for the employer's LCA, USCIS and PERM records. **Billed separately** by Google as grounding with Google Search. |
| DOL LCA disclosure file | Empty | Full path to an official LCA disclosure file (`.xlsx` or `.csv`). See [below](#official-dol-data-optional). |

### 7. Launch

Click **Launch AI Agent Pipeline**. A progress bar tracks the companies as they finish. Results stay on screen until the next run, so downloading the CSV doesn't clear them.

---

## Understanding the results

### Columns

| Column | Meaning |
|---|---|
| Company | Company name extracted from your notes |
| URL | Careers URL that was crawled |
| Match | Gemini's judgement: is there an open role matching your titles that fits your resume? |
| Sponsorship Status | One of the categories below |
| Matching Openings | For example *37 of 512 open roles match (Greenhouse 'stripe')*, or *No public job board found* |
| H-1B Records | Numbers, fiscal years and job titles found in the public records, with sources |
| H-1B Sources | What each source returned (*page found*, *blocked (anti-bot)*, *DOL file: 123 LCAs* …) |
| US Roles | *Yes*, *No* or *Unclear* |
| Locations | Locations stated for the relevant roles |
| Job Links | Links to the best-fitting postings |
| Summary | Two to four sentences explaining the verdict |
| H-1B Lookup Links | The database pages checked, so you can verify by hand |
| Notes | Your original notes about the company |
| Status | *OK*, or what went wrong (for example a failed crawl) |

### Sponsorship categories

| Status | Meaning |
|---|---|
| Proven sponsor (recent public records) | Records show H-1B filings in the last three fiscal years |
| Past sponsor (older records only) | Filings exist, but only older ones |
| Sponsors (stated on site, no records found) | The company's site says it sponsors; no records were found |
| Likely sponsors (indirect signals) | Indirect markers such as immigration support or relocation |
| Records show sponsorship, but site says no sponsorship | Conflicting evidence; worth asking the recruiter |
| Does not sponsor (stated on site) | The site excludes sponsorship (for example "must be authorized to work without sponsorship") |
| No evidence found | Nothing either way |

The three summary metrics above the table show the number of companies, matches, and failed or skipped rows.

---

## How it works

```text
Notes + Target roles + Resume
          │
          ▼
 Stage A0  Expand target roles into related titles          (Gemini, structured output)
 Stage A   Notes → companies with careers URLs               (Gemini, structured output)
          │
          ▼   for each company, concurrently
 ┌──────────────────────────────────────────────────────────────────┐
 │ Crawl careers page + one follow-up jobs page        (crawl4ai)   │
 │ Read full role list from Greenhouse / Lever / Ashby (public APIs)│
 │ H-1B evidence: h1bdata.info, myvisajobs.com,                     │
 │                Google Search via Gemini, DOL file (optional)     │
 └──────────────────────────────────────────────────────────────────┘
          │
          ▼
 Stage B   Fit + sponsorship evaluation                      (Gemini, structured output)
          │
          ▼
 Results table + CSV download
```

### Crawling

- crawl4ai runs with `CrawlerRunConfig(cache_mode=CacheMode.BYPASS)`, so every run fetches fresh pages.
- Each crawled page's markdown is cut to the first 12,000 characters (`markdown_text[:12000]`) before it goes to Gemini.
- After the careers page, the app follows **one** more link. It prefers job-board links, then job-listing pages. It skips translated copies of the same page (`/fr/careers`), and skips login, demo, help, blog and benefits pages.
- If a site's bot protection blocks the crawl, the page is retried once in crawl4ai's stealth mode.

### Job boards

When a careers page links to or embeds a Greenhouse, Lever or Ashby board, the app reads the board's **full** list of open roles through the public APIs. No key is needed:

| Board | Endpoint |
|---|---|
| Greenhouse | `https://boards-api.greenhouse.io/v1/boards/{token}/jobs` |
| Lever | `https://api.lever.co/v0/postings/{token}?mode=json` |
| Ashby | `https://api.ashbyhq.com/posting-api/job-board/{token}` |

When no board is linked, the app tries the company name as the board name (for example `grafanalabs`, `grafana-labs`) and marks the result *(guessed name)*. Up to 40 matching roles per company are passed to Gemini, US roles first when the USA filter is on.

### H-1B evidence

| Source | How it's used |
|---|---|
| [h1bdata.info](https://www.h1bdata.info/) | The employer's LCA listing across all years is crawled. |
| [myvisajobs.com](https://www.myvisajobs.com/) | The employer profile page is crawled. This site often blocks automated access. |
| Google Search via Gemini | Searches h1bgrader.com, h1bscan.com, myvisajobs.com, USCIS and other public pages. The answer is discarded unless it's backed by actual search results. |
| DOL OFLC disclosure file | Official LCA records, matched locally. Optional. |

Employer names are simplified before lookup: legal suffixes such as *Inc*, *LLC* and *Corp* are removed. Any database page that doesn't mention the company is discarded, because these sites sometimes return a default page about a different employer.

### Official DOL data (optional)

1. Download an **LCA Programs (H-1B, H-1B1, E-3)** disclosure file from the [DOL OFLC Performance Data page](https://www.dol.gov/agencies/eta/foreign-labor/performance).
2. Paste its full path into the sidebar, for example `/Users/you/Downloads/LCA_Disclosure_Data_FY2026_Q3.xlsx`.
3. Make sure `openpyxl` is installed if the file is an `.xlsx`.

The first load of an `.xlsx` can take a few minutes. After that the file is cached until it changes. Only the `CASE_STATUS`, `DECISION_DATE`, `VISA_CLASS`, `JOB_TITLE`, `EMPLOYER_NAME` and `WORKSITE_STATE` columns are read. Each company gets a summary of total filings, visa classes, case statuses, decision years, top job titles and worksite states.

---

## Tuning

These constants near the top of `app.py` control the main limits:

| Constant | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `gemini-3.8-flash` | Gemini model for every call |
| `PAGE_TIMEOUT_MS` | 45000 | Page-load timeout per crawl |
| `MAX_RESUME_CHARS` | 8000 | Resume text sent to the evaluation stage |
| `MAX_MATCHED_ROLES` | 40 | Matching job-board roles passed to Gemini per company |
| `MAX_BOARDS_PER_COMPANY` | 2 | Job boards read per company |
| `GEMINI_MAX_ATTEMPTS` | 3 | Retries for each structured Gemini call |
| `H1B_RECORDS_CONCURRENCY` | 2 | Parallel requests to the H-1B lookup sites |
| `BLOCKS_BEFORE_SKIPPING_SITE` | 2 | Anti-bot blocks before a lookup site is skipped for the rest of the run |
| `TITLE_EQUIVALENTS` | — | Spelling variants folded together when matching titles |

The 12,000-character page limit is the slice in `crawl_page()`.

---

## Limitations

- **Job boards:** only Greenhouse, Lever and Ashby are read in full. For companies on Workday, iCIMS or their own job sites, matching relies on the crawled careers pages (12,000 characters each), so a matching role can be missed.
- **Guessed boards:** a board found by guessing the company name could belong to a different company with the same name.
- **Name matching:** lookups use the simplified company name. A company that files under a different legal name (for example *Alphabet* instead of *Google*) can show no records on the crawled databases or in the DOL file, even if it sponsors.
- **Bot protection:** some careers sites and myvisajobs.com block automated browsers. Stealth mode helps on some sites, not all. Blocked sources are reported in `H-1B Sources` and `Status`.
- **USCIS H-1B Employer Data Hub:** it has no per-employer URL that can be crawled, and it was reported offline in March 2026. It's covered only by the Google Search step.
- **Costs:** every company uses several Gemini calls, and the Google Search step is billed separately. Turn it off in the sidebar if needed.
- **Model judgement:** Match, sponsorship status and summaries come from Gemini. Check them against the linked sources before acting.

---

## Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `GEMINI_API_KEY not found in environment or .env` | `.env` is missing or not next to `app.py`. Create it, or paste the key into the sidebar. |
| `Blocked by anti-bot protection: Cloudflare JS challenge` | The site blocks automated browsers. Nothing to fix; other sources still run. |
| `Near-empty content … with HTTP 200` | crawl4ai treats very short pages as blocked. Usually a JavaScript-only page or a bot check. |
| `Site crawl failed … (H-1B records only)` | The careers page couldn't be crawled. The company was still evaluated on its job board and H-1B records. |
| `No public job board found` | The company doesn't use Greenhouse, Lever or Ashby, or the board name couldn't be found. |
| `Could not read the DOL file` | Wrong path, or `openpyxl` isn't installed for `.xlsx` files. |
| A row says `Evaluation failed: Gemini request failed …` | API quota, network or key problem. Check the key and your quota, then run again. |
| Crawls fail immediately after installing | Run `crawl4ai-setup` (and `crawl4ai-doctor` if it persists). |
| `For better performance, install the Watchdog module` | Optional: `pip install watchdog`. |
| Gemini SDK warning about automatic function calling | Fixed in the current `app.py`. Update to the latest file. |

---

## Project structure

```text
job_search_utility_tool/
├── app.py        # The whole application
├── .env          # GEMINI_API_KEY (not committed)
└── README.md
```

`app.py` is organised in numbered sections:

1. Environment setup and constants
2. Pydantic schemas for structured Gemini output
3. Resume processing
4. Helpers: URLs, careers links, title parsing and matching, job-board APIs
5. Public H-1B records: DOL file, Google Search, lookup sites
6. Gemini calls: title expansion, company extraction, evaluation
7. Asynchronous crawling and the per-company pipeline
8. Streamlit UI
