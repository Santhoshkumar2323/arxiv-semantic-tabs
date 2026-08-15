# arXiv Signal

An automated pipeline that pulls new arXiv papers across 8 research areas every
48 hours, ranks them by semantic relevance against a defined interest profile per
area, and surfaces only the top matches through a dashboard. Runs entirely
unattended once deployed — no manual filtering, no checking arXiv by hand.

## The problem this solves

arXiv publishes far more papers per day than anyone can realistically read, spread
across categories that overlap in confusing ways (there's no single "reinforcement
learning" category, for instance — RL papers get scattered across cs.LG and
cs.AI along with everything else). Manually checking multiple category pages
every few days doesn't scale, and simple keyword search doesn't distinguish a
paper that mentions a term in passing from one that's actually about it. This
project handles both problems: it pulls broadly from the relevant categories per
area, then uses semantic similarity — not keyword matching — to decide what
actually belongs.

## How it works

Each 48-hour cycle runs the following five steps, independently, per sector:

**1. Fetch.** Queries arXiv's API for that sector's configured categories,
using a widened time window (48 hours plus a buffer, currently 24h) rather than
exactly 48 hours. This matters because arXiv doesn't index submissions in real
time — it processes and releases papers in batches, so a paper submitted right
near a cycle boundary can have a submission timestamp inside the window while
not yet being searchable via the API at the exact moment the job runs. Widening
the query window is a cheap safety margin: since results are sorted
newest-first and capped at a fixed maximum per sector, the extra window only
ever adds older candidates at the tail — it never displaces genuinely new ones.
Requests are paginated, paced to arXiv's own requested rate limit, and retried
with exponential backoff on failure (respecting the server's Retry-After header
when one is provided, rather than blindly retrying on a fixed schedule).

**2. Exact-duplicate removal.** Papers are deduplicated by arXiv ID before
anything gets embedded. This ordering is deliberate — embedding is the more
computationally expensive step, so there's no reason to spend it on a paper
that's about to be discarded as a repeat anyway. This mainly catches
pagination overlaps rather than being a major filter on its own.

**3. Embedding.** Each surviving paper's title and abstract are concatenated
and encoded into a numerical vector using BAAI/bge-base-en-v1.5, an
open-source embedding model that runs locally — no external API calls, no
per-request cost. Each sector also has a fixed "profile" — a short, keyword-
dense description of what that sector considers relevant — which gets encoded
the same way, but with a different instruction prefix than the papers get
(BGE is trained asymmetrically: queries and passages are meant to be encoded
differently for the best retrieval quality). Titles are included alongside
abstracts because they often carry sharper, more specific signal than an
abstract's opening sentence, which is frequently generic scene-setting rather
than substance.

**4. Near-duplicate removal.** Papers whose vectors are highly similar to
each other (cosine similarity above a configured threshold, currently 0.92)
are treated as duplicates even when their arXiv IDs differ — this catches
near-identical resubmissions or heavily overlapping papers that exact-ID
matching alone would miss. This comparison happens within a sector's own
candidate pool only, not across sectors.

**5. Ranking.** Every remaining paper's vector is compared against its
sector's profile vector using cosine similarity, sorted highest to lowest,
and the top N papers (configurable per sector, typically 8–10) are kept.
Corrupted or degenerate embeddings — which would otherwise silently produce
nonsensical similarity scores — are detected and excluded from ranking rather
than trusted.

If a sector ends up with zero qualifying papers in a cycle, the dashboard
shows it as empty for that cycle rather than quietly falling back to a
previous cycle's results. A sector's displayed state always reflects what
actually happened in the most recent run, not an approximation of it.

## Research areas

AI, ML, DL, RL, Robotics, Radiotherapy, Semiconductors, Embedded Systems

Every sector is defined entirely in `config/sectors.yaml`: which arXiv
categories it pulls from, its relevance profile text, and how many top papers
it keeps. Nothing about a sector is hardcoded in the pipeline or the
dashboard — both simply iterate over whatever sectors exist in that file.
Adding, removing, or retuning a sector (including changing what "relevant"
means for it) is purely a config edit; no code changes are required in either
the pipeline or the dashboard.

One deliberate quirk worth understanding: AI, ML, DL, and RL intentionally
pull from overlapping arXiv categories, because arXiv itself has no separate
category for deep learning or reinforcement learning specifically — those
areas mostly live inside cs.LG and cs.AI alongside everything else. Since
there's no clean way to split them at the API level, separation happens
entirely through each sector's relevance profile during the ranking step
instead. A practical consequence: a paper can legitimately rank highly — and
appear — in more than one of those four sectors if it's genuinely relevant
to both. This is expected behavior, not a defect, and reflects the fact that
research areas like these genuinely aren't disjoint in practice.

## Automation

A GitHub Actions workflow runs the full pipeline on a schedule approximating
every 48 hours (GitHub's scheduling is calendar-based, not interval-based, so
"every 48 hours" is implemented as "every 2 days at a fixed hour," which can
drift slightly at month boundaries — a known, accepted quirk rather than a
bug). Each run commits the refreshed output files back to the repository.
A connected Streamlit deployment redeploys automatically on that commit, so
once set up, the dashboard stays current with no manual intervention. The
workflow can also be triggered manually from the Actions tab at any time,
independent of the schedule — useful for testing changes without waiting for
the next scheduled run.

The embedding model is cached between runs (keyed specifically to the model
name, not the whole config file) so it isn't re-downloaded on every single
cycle — only when the model itself actually changes.

## Reliability behavior

- Each sector runs in isolation. A failure in one sector — an arXiv timeout,
  a malformed response, anything — is caught and logged without stopping the
  other sectors from completing normally in the same run.
- Every run produces a structured status log with per-sector pulled/retained
  counts and a clear ok/failed state, so if something looks wrong on the
  dashboard, the cause is traceable from the log rather than requiring a
  re-run to investigate.
- A sector that legitimately fetched zero relevant papers is distinguished
  from a sector that failed outright — both are visible, but they mean
  different things and are labeled differently.

## Project structure

- `config/sectors.yaml` — all sector definitions and global pipeline settings
- `pipeline/` — the fetch → dedupe → embed → rank → publish logic
- `app/streamlit_app.py` — the dashboard
- `tests/run_local.py` — run the pipeline locally before trusting it unattended
- `.github/workflows/` — the scheduled automation

## Running it locally

pip install -r requirements.txt
python -m tests.run_local --quick # fast smoke test, isolated output
python -m pipeline.run_pipeline # full run, all sectors
streamlit run app/streamlit_app.py # view the dashboard
