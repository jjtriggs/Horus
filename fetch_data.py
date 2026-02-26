#!/usr/bin/env python3
"""
fetch_data.py - Horus CTO Dashboard data fetcher.

Fetches team metrics from GitHub (GraphQL + REST) and the Anthropic API,
then writes the result to data/team-metrics.json in the exact structure
that the browser reads as STATE.team_cache.

Usage:
    python fetch_data.py [--days 90] [--skip-anthropic] [--skip-member-weekly]

Tokens are read from a .env file in the same directory as this script,
or from environment variables:
    GITHUB_TOKEN         - Personal access token with read:org, repo, read:user
    ANTHROPIC_ADMIN_KEY  - Anthropic admin key for usage/cost reports
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests  # pip install requests

# ---------------------------------------------------------------------------
# Team configuration
# ---------------------------------------------------------------------------

TEAM_DATA = [
    {"team": "Frontend", "members": [
        {"name": "Tom Tidswell",    "github": "tomtidswell",       "cc": True,  "lead": True},
        {"name": "Terry Hibbert",   "github": "TerryHibbert",      "cc": True,  "lead": False},
        {"name": "Chris Crouch",    "github": "chrisdc",           "cc": True,  "lead": False},
    ]},
    {"team": "Python", "members": [
        {"name": "Matt Brighty",    "github": "MBrighty",          "cc": True,  "lead": True},
        {"name": "Cameron Bamford", "github": "CameronBamford",    "cc": True,  "lead": False},
        {"name": "Hikmat Hasan",    "github": "hh2110",            "cc": True,  "lead": False},
        {"name": "Luke Moran",      "github": "lukemoran-so",      "cc": True,  "lead": False},
    ]},
    {"team": "Endpoint", "members": [
        {"name": "Andrea Marangoni","github": "AndreaSenseon",     "cc": True,  "lead": True},
        {"name": "Ollie Perry",     "github": "operry-senseon",    "cc": True,  "lead": False},
        {"name": "Fabian Ceccato",  "github": "fabianceccato",     "cc": True,  "lead": False},
    ]},
    {"team": "DevOps", "members": [
        {"name": "Ethan Moore",       "github": "EthanM-0",       "cc": True,  "lead": True},
        {"name": "Tim Curtis",        "github": "timjohncurtis",        "cc": True,  "lead": False},
        {"name": "Mark JL",           "github": "mark-jordanovic-lewis",           "cc": True,  "lead": False},
        {"name": "Anthony Sikosa",    "github": "asikosa-senseon",   "cc": True,  "lead": False},
        {"name": "Thomas McGarrigan", "github": "thomas-mcgarrigan", "cc": True,  "lead": False},
    ]},
]

GITHUB_ORG = "senseon-tech"

# Key repos per team for lines-of-code stats (contributor stats API)
# contributionsCollection doesn't work for private repos unless members opt-in
TEAM_REPOS: dict[str, list[str]] = {
    "Frontend":  ["senseon-ui"],
    "Python":    ["senseon-appliance-api", "senseon-customer-api", "senseon-log-receiver"],
    "Endpoint":  ["senseon-enterprise-endpoint", "senseon-apns", "senseon-endpoint-driver"],
    "DevOps":    ["senseon-cloud-infra", "terraform-appliance-cluster", "senseon-actions",
                  "senseon-access-control", "senseon-build-assistant"],
}

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

# Flat list of all members for convenience
ALL_MEMBERS: list[dict[str, Any]] = [
    m for team in TEAM_DATA for m in team["members"]
]

# All GitHub handles (preserving original case from config, lowered for map keys
# where GitHub searches are case-insensitive)
ALL_HANDLES: list[str] = [m["github"] for m in ALL_MEMBERS]

# Map from lowercase handle -> display name
HANDLE_TO_NAME: dict[str, str] = {
    m["github"].lower(): m["name"] for m in ALL_MEMBERS
}

GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"
ANTHROPIC_API = "https://api.anthropic.com"

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "team-metrics.json"

# ---------------------------------------------------------------------------
# Environment / token loading
# ---------------------------------------------------------------------------


def load_env(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file, ignoring blank lines and # comments."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def get_tokens() -> tuple[str, str | None]:
    """
    Return (github_token, anthropic_admin_key).
    Load from .env first, then environment variables.
    Exits with an error message if GITHUB_TOKEN is missing.
    """
    env = load_env(SCRIPT_DIR / ".env")

    github_token = env.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    anthropic_key = env.get("ANTHROPIC_ADMIN_KEY") or os.environ.get("ANTHROPIC_ADMIN_KEY")

    if not github_token:
        print(
            "[ERR] GITHUB_TOKEN is required but was not found.\n"
            "  Set it in a .env file next to this script or as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    return github_token, anthropic_key


# ---------------------------------------------------------------------------
# Date / week helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Current time in UTC (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def iso_date(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DD."""
    return dt.strftime("%Y-%m-%d")


def iso_datetime_z(dt: datetime) -> str:
    """Format datetime as ISO-8601 with milliseconds and Z suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def get_weeks(days: int, now: datetime | None = None) -> list[dict[str, str]]:
    """
    Return a list of week dicts that mirrors the JS _getWeeks() function exactly.

    Each entry: {start, end, label}
      - num_weeks = max(1, round(days / 7))
      - Loop i from num_weeks-1 down to 0
      - end_dt   = now - timedelta(days=i*7)
      - start_dt = end_dt - timedelta(days=6)
      - label    = "{start_dt.day} {Abbr Month}"
    """
    if now is None:
        now = utc_now()

    num_weeks = max(1, round(days / 7))
    weeks: list[dict[str, str]] = []

    for i in range(num_weeks - 1, -1, -1):
        end_dt = now - timedelta(days=i * 7)
        start_dt = end_dt - timedelta(days=6)
        label = f"{start_dt.day} {start_dt.strftime('%b')}"
        weeks.append({
            "start": iso_date(start_dt),
            "end": iso_date(end_dt),
            "label": label,
        })

    return weeks


def team_slug(handle: str) -> str:
    """Convert a GitHub handle to the senseon team slug format."""
    return "u_" + re.sub(r"[^a-z0-9]", "_", handle.lower())


# ---------------------------------------------------------------------------
# GitHub session helpers
# ---------------------------------------------------------------------------


def make_gh_session(token: str) -> requests.Session:
    """Create a requests Session pre-configured for GitHub APIs."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return session


def gh_graphql(query: str, session: requests.Session) -> dict[str, Any]:
    """
    Execute a GitHub GraphQL query and return the parsed JSON response.
    Raises RuntimeError on HTTP or GraphQL errors.
    """
    resp = session.post(GITHUB_GRAPHQL, json={"query": query}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data.get("data") or {}


# ---------------------------------------------------------------------------
# Fetch: aggregate GitHub stats (PR counts per member)
# ---------------------------------------------------------------------------


def fetch_github_stats(
    days: int, gh_session: requests.Session
) -> tuple[dict[str, dict[str, int]], str | None]:
    """
    Fetch prs_opened, prs_merged, prs_reviewed for every team member in one
    GraphQL batch request (3 aliased search fields per member).

    Returns:
        ({handle_lower: {prs_opened, prs_merged, prs_reviewed, commits}}, error_str|None)
    """
    since = iso_date(utc_now() - timedelta(days=days))

    fields: list[str] = []
    for m in ALL_MEMBERS:
        h = m["github"]
        hl = h.lower()
        # We use alias prefixes po_ / pm_ / pr_ to avoid collisions
        alias_o = f"po_{re.sub(r'[^a-z0-9]', '_', hl)}"
        alias_m = f"pm_{re.sub(r'[^a-z0-9]', '_', hl)}"
        alias_r = f"prr_{re.sub(r'[^a-z0-9]', '_', hl)}"

        q_opened  = f"type:pr author:{h} org:{GITHUB_ORG} created:>={since}"
        q_merged  = f"type:pr is:merged author:{h} org:{GITHUB_ORG} merged:>={since}"
        q_reviewed = (
            f"type:pr reviewed-by:{h} org:{GITHUB_ORG} -author:{h} updated:>={since}"
        )

        fields.append(
            f'{alias_o}: search(query:"{q_opened}", type:ISSUE, first:1)'
            f"{{issueCount}}"
        )
        fields.append(
            f'{alias_m}: search(query:"{q_merged}", type:ISSUE, first:1)'
            f"{{issueCount}}"
        )
        fields.append(
            f'{alias_r}: search(query:"{q_reviewed}", type:ISSUE, first:1)'
            f"{{issueCount}}"
        )

    query = "{ " + "\n".join(fields) + " }"

    try:
        print(f"  Fetching GitHub PR stats for {len(ALL_MEMBERS)} members (GraphQL batch)...")
        data = gh_graphql(query, gh_session)
    except Exception as exc:
        print(f"  [ERR] GitHub stats fetch failed: {exc}")
        # Return zeros for everyone
        result = {
            m["github"].lower(): {"prs_opened": 0, "prs_merged": 0, "prs_reviewed": 0, "commits": 0}
            for m in ALL_MEMBERS
        }
        return result, str(exc)

    result: dict[str, dict[str, int]] = {}
    for m in ALL_MEMBERS:
        h = m["github"]
        hl = h.lower()
        alias_o = f"po_{re.sub(r'[^a-z0-9]', '_', hl)}"
        alias_m = f"pm_{re.sub(r'[^a-z0-9]', '_', hl)}"
        alias_r = f"prr_{re.sub(r'[^a-z0-9]', '_', hl)}"

        result[hl] = {
            "prs_opened":   (data.get(alias_o) or {}).get("issueCount", 0),
            "prs_merged":   (data.get(alias_m) or {}).get("issueCount", 0),
            "prs_reviewed": (data.get(alias_r) or {}).get("issueCount", 0),
            "commits": 0,  # filled in separately
        }

    print(f"  [OK] GitHub PR stats fetched")
    return result, None


# ---------------------------------------------------------------------------
# Fetch: commit counts (REST search, rate-limited)
# ---------------------------------------------------------------------------


def fetch_commit_counts(
    handles: list[str], since: str, gh_session: requests.Session
) -> dict[str, int]:
    """
    Fetch commit counts for each handle via the GitHub REST commit search endpoint.

    The commit search API is limited to 30 requests/minute, so we sleep 2.1 s
    between each request to stay safely under the limit.

    Returns {handle_lower: total_count}
    """
    counts: dict[str, int] = {}
    for i, handle in enumerate(handles):
        url = (
            f"{GITHUB_API}/search/commits"
            f"?q=author:{handle}+org:{GITHUB_ORG}+committer-date:>={since}&per_page=1"
        )
        try:
            resp = gh_session.get(url, timeout=30, headers={"Accept": "application/vnd.github.cloak-preview+json"})
            resp.raise_for_status()
            data = resp.json()
            counts[handle.lower()] = data.get("total_count", 0)
            print(f"    [OK] Commits for {handle}: {counts[handle.lower()]}")
        except Exception as exc:
            print(f"    [ERR] Commit count failed for {handle}: {exc}")
            counts[handle.lower()] = 0

        # Respect the 30 req/min limit on the commit search endpoint
        if i < len(handles) - 1:
            time.sleep(2.1)

    return counts


# ---------------------------------------------------------------------------
# Fetch: org-level weekly PR trend
# ---------------------------------------------------------------------------


def fetch_org_weekly_trend(
    days: int, gh_session: requests.Session
) -> list[dict[str, Any]]:
    """
    Fetch weekly PR-opened and PR-merged counts for the whole org in a single
    GraphQL batch.

    Returns a list of week dicts:
        [{start, end, label, prs_opened, prs_merged}, ...]
    """
    weeks = get_weeks(days)
    fields: list[str] = []

    for i, w in enumerate(weeks):
        q_opened = f"type:pr org:{GITHUB_ORG} created:{w['start']}..{w['end']}"
        q_merged = f"type:pr is:merged org:{GITHUB_ORG} merged:{w['start']}..{w['end']}"
        fields.append(
            f'w{i}o: search(query:"{q_opened}", type:ISSUE, first:1){{issueCount}}'
        )
        fields.append(
            f'w{i}m: search(query:"{q_merged}", type:ISSUE, first:1){{issueCount}}'
        )

    query = "{ " + "\n".join(fields) + " }"

    print(f"  Fetching org weekly trend ({len(weeks)} weeks)...")
    try:
        data = gh_graphql(query, gh_session)
    except Exception as exc:
        print(f"  [ERR] Org weekly trend fetch failed: {exc}")
        return []

    result: list[dict[str, Any]] = []
    for i, w in enumerate(weeks):
        result.append({
            "start":      w["start"],
            "end":        w["end"],
            "label":      w["label"],
            "prs_opened": (data.get(f"w{i}o") or {}).get("issueCount", 0),
            "prs_merged":  (data.get(f"w{i}m") or {}).get("issueCount", 0),
        })

    print(f"  [OK] Org weekly trend fetched ({len(result)} weeks)")
    return result


# ---------------------------------------------------------------------------
# Fetch: per-member weekly data + lines + AI reviews
# ---------------------------------------------------------------------------


def fetch_member_weekly(
    handle: str, days: int, gh_session: requests.Session
) -> dict[str, Any]:
    """
    Fetch per-member weekly PR counts, lines-changed stats, and AI-bot review
    count in a single GraphQL query.

    Returns the `data` dict that is stored at member_weekly[handle].data:
        {weeks, lines_added, lines_removed, lines_net,
         lines_per_pr, pr_sample, ai_prs_reviewed}
    """
    now = utc_now()
    since_dt = now - timedelta(days=days)
    since = iso_date(since_dt)

    weeks = get_weeks(days, now)
    fields: list[str] = []

    # Weekly PR fields
    for i, w in enumerate(weeks):
        q_o = f"type:pr author:{handle} org:{GITHUB_ORG} created:{w['start']}..{w['end']}"
        q_m = f"type:pr is:merged author:{handle} org:{GITHUB_ORG} merged:{w['start']}..{w['end']}"
        q_r = (
            f"type:pr reviewed-by:{handle} org:{GITHUB_ORG} "
            f"-author:{handle} updated:{w['start']}..{w['end']}"
        )
        fields.append(f'w{i}o: search(query:"{q_o}", type:ISSUE, first:1){{issueCount}}')
        fields.append(f'w{i}m: search(query:"{q_m}", type:ISSUE, first:1){{issueCount}}')
        fields.append(f'w{i}r: search(query:"{q_r}", type:ISSUE, first:1){{issueCount}}')

    # AI-authored PRs reviewed by this member
    ai_query = (
        f"type:pr reviewed-by:{handle} org:{GITHUB_ORG} "
        f"author:dependabot OR author:google-labs-jules OR author:copilot-swe-agent "
        f"updated:>={since}"
    )
    fields.append(
        f'airev: search(query:"{ai_query}", type:ISSUE, first:1){{issueCount}}'
    )

    # Lines of code: fetch merged PR additions/deletions directly via GraphQL.
    # This works for private repos and correctly handles squash merges.
    # (contributionsCollection is blocked for private repos; contributor stats
    #  report 0 additions/deletions for squash-merged commits.)
    pr_lines_query = (
        f'type:pr is:merged author:{handle} org:{GITHUB_ORG} merged:>={since}'
    )
    fields.append(
        f'prlines: search(query:"{pr_lines_query}", type:ISSUE, first:100)'
        f'{{ issueCount nodes {{ ... on PullRequest {{ additions deletions }} }} }}'
    )

    query = "{ " + "\n".join(fields) + " }"
    data = gh_graphql(query, gh_session)

    # Build weekly list (use `or {}` to guard against GraphQL null fields)
    week_rows: list[dict[str, Any]] = []
    for i, w in enumerate(weeks):
        week_rows.append({
            "start":        w["start"],
            "end":          w["end"],
            "label":        w["label"],
            "prs_opened":   (data.get(f"w{i}o") or {}).get("issueCount", 0),
            "prs_merged":   (data.get(f"w{i}m") or {}).get("issueCount", 0),
            "prs_reviewed": (data.get(f"w{i}r") or {}).get("issueCount", 0),
        })

    ai_prs_reviewed = (data.get("airev") or {}).get("issueCount", 0)

    # Aggregate lines from merged PR nodes
    pr_nodes = (data.get("prlines") or {}).get("nodes") or []
    lines_added   = sum((n or {}).get("additions", 0) for n in pr_nodes)
    lines_removed = sum((n or {}).get("deletions",  0) for n in pr_nodes)
    pr_sample     = len(pr_nodes)
    lines_net     = lines_added - lines_removed
    lines_per_pr  = round((lines_added + lines_removed) / pr_sample) if pr_sample > 0 else 0

    return {
        "weeks":           week_rows,
        "lines_added":     lines_added,
        "lines_removed":   lines_removed,
        "lines_net":       lines_net,
        "lines_per_pr":    lines_per_pr,
        "pr_sample":       pr_sample,
        "ai_prs_reviewed": ai_prs_reviewed,
    }


def fetch_lines_from_stats(
    handle: str,
    days: int,
    gh_session: requests.Session,
    since_dt: datetime,
) -> tuple[int, int, int]:
    """
    Get lines added/removed for a member by scanning contributor stats for their
    team's repos. Returns (lines_added, lines_removed, merged_pr_count_proxy).
    Uses /repos/{org}/{repo}/stats/contributors which works for private repos.
    """
    # Find which team this member belongs to
    member_team = next(
        (t["team"] for t in TEAM_DATA for m in t["members"]
         if m["github"].lower() == handle.lower()),
        None
    )
    repos = TEAM_REPOS.get(member_team, [])
    if not repos:
        return 0, 0, 0

    since_ts = int(since_dt.timestamp())
    lines_added = lines_removed = pr_proxy = 0

    for repo in repos:
        for attempt in range(8):
            resp = gh_session.get(
                f"{GITHUB_API}/repos/{GITHUB_ORG}/{repo}/stats/contributors",
                timeout=30,
            )
            if resp.status_code == 202:
                time.sleep(5)  # GitHub is computing stats, retry
                continue
            if not resp.ok:
                break
            for contributor in resp.json() or []:
                if (contributor.get("author") or {}).get("login", "").lower() != handle.lower():
                    continue
                for week in contributor.get("weeks", []):
                    if week.get("w", 0) >= since_ts:
                        lines_added   += week.get("a", 0)
                        lines_removed += week.get("d", 0)
                        pr_proxy      += week.get("c", 0)  # commits as proxy
            break

    return lines_added, lines_removed, pr_proxy


# ---------------------------------------------------------------------------
# Fetch: Anthropic usage and cost
# ---------------------------------------------------------------------------


def fetch_anthropic_usage(days: int, ant_key: str) -> dict[str, Any] | None:
    """
    Fetch token usage and cost from the Anthropic API.

    Cost endpoint: single request for the full date range.
    Usage endpoint: must be called day-by-day (the multi-day endpoint returns 500).

    Returns:
        {tokens: int, cost_usd: float, daily: [{label, tokens, cost_usd}, ...]}
    or None on total failure.
    """
    headers = {
        "x-api-key": ant_key,
        "anthropic-version": "2023-06-01",
    }

    now = utc_now()
    since_dt = now - timedelta(days=days)
    since = iso_date(since_dt)
    today = iso_date(now)
    tomorrow = iso_date(now + timedelta(days=1))

    session = requests.Session()
    session.headers.update(headers)

    # ------------------------------------------------------------------
    # 1. Cost report (single request for the full window)
    # ------------------------------------------------------------------
    cost_by_date: dict[str, float] = {}
    try:
        print("  Fetching Anthropic cost report (paginated)...")
        params: dict[str, str] = {
            "starting_at": since,
            "ending_at": tomorrow,
            "bucket_width": "1d",
        }
        while True:
            resp = session.get(
                f"{ANTHROPIC_API}/v1/organizations/cost_report",
                params=params, timeout=30,
            )
            resp.raise_for_status()
            cost_data = resp.json()
            for bucket in cost_data.get("data", []):
                # amount is a string in cents ("27609.13371") — must float() it
                raw_date = (bucket.get("starting_at") or bucket.get("start_time") or "")[:10]
                if not raw_date:
                    continue
                total_cents = sum(
                    float(r.get("amount", 0) or 0) for r in bucket.get("results", [])
                )
                cost_by_date[raw_date] = round(total_cents / 100, 2)
            if not cost_data.get("has_more") or not cost_data.get("next_page"):
                break
            params = {**params, "page": cost_data["next_page"]}
        print(f"  [OK] Cost report fetched ({len(cost_by_date)} days with spend)")
    except Exception as exc:
        print(f"  [ERR] Anthropic cost report failed: {exc}")
        # Continue – still write partial data

    # ------------------------------------------------------------------
    # 2. Usage report (one request per day)
    # ------------------------------------------------------------------
    total_tokens = 0
    daily_list: list[dict[str, Any]] = []

    # Build daily list from cost data (reliable) + best-effort token counts.
    # The usage_report/messages endpoint returns 500 on many orgs — treat as optional.
    tokens_by_date: dict[str, int] = {}
    print(f"  Fetching Anthropic usage ({days} days, one request per day)...")
    current = since_dt
    usage_ok = 0
    while current <= now:
        d      = iso_date(current)
        d_next = iso_date(current + timedelta(days=1))
        try:
            resp = session.get(
                f"{ANTHROPIC_API}/v1/organizations/usage_report/messages",
                params={"starting_at": d, "ending_at": d_next, "bucket_width": "1d"},
                timeout=15,
            )
            if resp.ok:
                usage_data = resp.json()
                day_tokens = 0
                for bucket in usage_data.get("data", []):
                    for r in bucket.get("results", []):
                        day_tokens += (
                            (r.get("input_tokens") or 0)
                            + (r.get("output_tokens") or 0)
                            + (r.get("cache_read_input_tokens") or 0)
                            + (r.get("cache_creation_input_tokens") or 0)
                        )
                if day_tokens > 0:
                    tokens_by_date[d] = day_tokens
                    usage_ok += 1
        except Exception:
            pass  # Usage endpoint is best-effort; cost data is the source of truth
        current += timedelta(days=1)

    if usage_ok:
        print(f"  [OK] Usage data: {usage_ok} days with token data")
    else:
        print("  Usage endpoint unavailable (plan limitation) — cost data only")

    total_tokens = sum(tokens_by_date.values())
    total_cost   = round(sum(cost_by_date.values()), 2)

    # Build daily list from whichever days have cost data (always available)
    for d, cost in sorted(cost_by_date.items()):
        if cost > 0:
            dt = datetime.strptime(d, "%Y-%m-%d")
            daily_list.append({
                "label":    f"{dt.day} {dt.strftime('%b')}",
                "tokens":   tokens_by_date.get(d, 0),
                "cost_usd": cost,
            })

    print(f"  [OK] Anthropic: {total_tokens:,} tokens, ${total_cost:.2f} cost, {len(daily_list)} days")

    return {
        "tokens":   total_tokens,
        "cost_usd": total_cost,
        "daily":    daily_list,
    }


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, data: Any) -> None:
    """Write data as JSON to path atomically (write to .tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch team metrics from GitHub and Anthropic for the Horus CTO dashboard."
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days to look back (default: 90)",
    )
    parser.add_argument(
        "--skip-anthropic", action="store_true",
        help="Skip Anthropic API calls",
    )
    parser.add_argument(
        "--skip-member-weekly", action="store_true",
        help="Skip per-member weekly/lines fetch (faster, for quick refreshes)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = args.days
    now = utc_now()

    print(f"\nHorus data fetch - window: {days} days\n")

    github_token, anthropic_key = get_tokens()
    gh_session = make_gh_session(github_token)
    since = iso_date(now - timedelta(days=days))

    # Partial result that we can write even if something fails mid-run
    result: dict[str, Any] = {
        "window_days":   days,
        "fetched_at":    iso_datetime_z(now),
        "github_stats":  {},
        "github_error":  None,
        "anthropic":     None,
        "ant_usage":     None,
        "weekly_org":    [],
        "member_weekly": {},
    }

    try:
        # ------------------------------------------------------------------
        # 1. GitHub aggregate PR stats
        # ------------------------------------------------------------------
        print("[1/4] GitHub aggregate PR stats")
        gh_stats, gh_error = fetch_github_stats(days, gh_session)
        result["github_stats"] = gh_stats
        result["github_error"] = gh_error

        # ------------------------------------------------------------------
        # 2. Commit counts (rate-limited REST)
        # ------------------------------------------------------------------
        print(f"\n[2/4] GitHub commit counts (REST, rate-limited)")
        commit_counts = fetch_commit_counts(ALL_HANDLES, since, gh_session)
        for handle_lower, count in commit_counts.items():
            if handle_lower in result["github_stats"]:
                result["github_stats"][handle_lower]["commits"] = count

        # ------------------------------------------------------------------
        # 3. Org weekly trend
        # ------------------------------------------------------------------
        print(f"\n[3/4] Org weekly trend")
        result["weekly_org"] = fetch_org_weekly_trend(days, gh_session)

        # ------------------------------------------------------------------
        # 4. Per-member weekly data
        # ------------------------------------------------------------------
        if not args.skip_member_weekly:
            print(f"\n[4/4] Per-member weekly data + lines ({len(ALL_MEMBERS)} members)")
            # Pre-warm contributor stats for all repos so 202s resolve before we need them.
            # GitHub caches these; the first request triggers computation (~10-60s for large repos).
            all_repos = list({r for repos in TEAM_REPOS.values() for r in repos})
            # Pre-warm contributor stats for all repos and wait until they're all ready.
            # GitHub returns 202 while computing; we poll until 200 or give up after 90s.
            print(f"  Pre-warming contributor stats for {len(all_repos)} repos...")
            pending = set(all_repos)
            for attempt in range(18):  # up to 90s (18 * 5s)
                still_pending = set()
                for repo in pending:
                    resp = gh_session.get(
                        f"{GITHUB_API}/repos/{GITHUB_ORG}/{repo}/stats/contributors",
                        timeout=15,
                    )
                    if resp.status_code == 202:
                        still_pending.add(repo)
                pending = still_pending
                if not pending:
                    break
                time.sleep(5)
            if pending:
                print(f"  [WARN] Stats still computing for: {', '.join(pending)} (will retry per-member)")
            else:
                print("  [OK] Stats cache warmed")
            fetched_count = 0
            for m in ALL_MEMBERS:
                handle = m["github"]
                name = m["name"]
                print(f"  Fetching {name} ({handle})...")
                try:
                    member_data = fetch_member_weekly(handle, days, gh_session)
                    result["member_weekly"][handle.lower()] = {
                        "data":       member_data,
                        "fetched_at": iso_datetime_z(utc_now()),
                        "window_days": days,
                    }
                    fetched_count += 1
                    print(f"  [OK] {name}")
                except Exception as exc:
                    print(f"  [ERR] {name}: {exc}")
                    # Write a zeroed-out placeholder so the browser doesn't break
                    result["member_weekly"][handle.lower()] = {
                        "data": {
                            "weeks":           [],
                            "lines_added":     0,
                            "lines_removed":   0,
                            "lines_net":       0,
                            "lines_per_pr":    0,
                            "pr_sample":       0,
                            "ai_prs_reviewed": 0,
                        },
                        "fetched_at":  iso_datetime_z(utc_now()),
                        "window_days": days,
                    }

            print(f"  [OK] Per-member data: {fetched_count}/{len(ALL_MEMBERS)} members fetched")
        else:
            print("\n[4/4] Skipping per-member weekly data (--skip-member-weekly)")

        # ------------------------------------------------------------------
        # 5. Anthropic usage
        # ------------------------------------------------------------------
        if not args.skip_anthropic:
            print(f"\n[5/5] Anthropic usage")
            if anthropic_key:
                ant_usage = fetch_anthropic_usage(days, anthropic_key)
                result["ant_usage"] = ant_usage
            else:
                print("  [ERR] ANTHROPIC_ADMIN_KEY not set – skipping")
        else:
            print("\n[5/5] Skipping Anthropic usage (--skip-anthropic)")

    except Exception as exc:
        print(f"\n[ERR] Fatal error: {exc}", file=sys.stderr)
        print("  Writing partial data...", file=sys.stderr)
        result["github_error"] = result.get("github_error") or str(exc)
    finally:
        # Always write whatever we have
        write_json_atomic(OUTPUT_FILE, result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_tokens = (result.get("ant_usage") or {}).get("tokens", 0)
    total_cost   = (result.get("ant_usage") or {}).get("cost_usd", 0.0)
    members_fetched = len(result.get("member_weekly", {}))

    print(f"""
==========================================
Horus fetch complete
==========================================
  Members fetched:  {members_fetched}/{len(ALL_MEMBERS)}
  Weekly weeks:     {len(result.get('weekly_org', []))}
  Anthropic tokens: {total_tokens:,}
  Anthropic cost:   ${total_cost:.2f}
  Output:           {OUTPUT_FILE}
==========================================
""")


if __name__ == "__main__":
    main()
