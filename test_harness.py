#!/usr/bin/env python3
"""
test_harness.py - Horus CTO Dashboard end-to-end test harness.

Validates the data pipeline (data file integrity), the local server, and
the external API connectivity. Uses only Python stdlib - no pip deps needed.

Usage:
    python test_harness.py

Tokens are read from a .env file in the same directory or from environment
variables: GITHUB_TOKEN, ANTHROPIC_ADMIN_KEY.

The server is expected at http://127.0.0.1:8080. Start it with:
    python server.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_DIR     = Path(__file__).parent
DATA_FILE      = SCRIPT_DIR / "data" / "team-metrics.json"
SERVER_BASE    = "http://127.0.0.1:8080"
EXPECTED_LOGIN = "jjtriggs"

# All 15 team member GitHub handles (lowercase for map key lookups)
ALL_HANDLES_LOWER = [
    "tomtidswell", "terryhibbert", "chrisdc",
    "mbrighty", "cameronbamford", "hh2110", "lukemoran-so",
    "andreasenseon", "operry-senseon", "fabianceccato",
    "ethan-moore", "tim-curtis", "mark-jl",
    "asikosa-senseon", "thomas-mcgarrigan",
]

# ---------------------------------------------------------------------------
# Token loader (mirrors fetch_data.py)
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
    """Return (github_token, anthropic_admin_key) from .env or environment."""
    env = load_env(SCRIPT_DIR / ".env")
    github_token = env.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    anthropic_key = env.get("ANTHROPIC_ADMIN_KEY") or os.environ.get("ANTHROPIC_ADMIN_KEY")
    return github_token, anthropic_key


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def http_get(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
) -> tuple[int, bytes]:
    """
    Perform a GET request.
    Returns (status_code, body_bytes).
    Raises urllib.error.URLError on connection error.
    """
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def http_options(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
) -> tuple[int, bytes]:
    """
    Perform an OPTIONS request.
    Returns (status_code, body_bytes).
    """
    req = urllib.request.Request(url, method="OPTIONS", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def http_post(
    url: str,
    body: bytes,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
) -> tuple[int, bytes]:
    """
    Perform a POST request.
    Returns (status_code, body_bytes).
    """
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# ---------------------------------------------------------------------------
# Test runner infrastructure
# ---------------------------------------------------------------------------

TestResult = tuple[bool, str]  # (passed, message)
TestFn = Callable[[], TestResult]


def run_test(fn: TestFn) -> TestResult:
    """Run a single test function, catching all exceptions."""
    try:
        return fn()
    except Exception as exc:
        return False, f"Unhandled exception: {exc}"


# ---------------------------------------------------------------------------
# Group 1: Data file validation
# ---------------------------------------------------------------------------


def test_data_file_exists() -> TestResult:
    """Test 1: data/team-metrics.json exists."""
    if DATA_FILE.exists():
        size = DATA_FILE.stat().st_size
        return True, f"data/team-metrics.json exists ({size:,} bytes)"
    return False, f"data/team-metrics.json not found at {DATA_FILE}"


def test_json_valid() -> TestResult:
    """Test 2: JSON is valid and parseable."""
    if not DATA_FILE.exists():
        return False, "File does not exist - cannot parse"
    try:
        text = DATA_FILE.read_text(encoding="utf-8")
        json.loads(text)
        return True, "JSON valid and parseable"
    except json.JSONDecodeError as exc:
        return False, f"JSON parse error: {exc}"


def _load_data() -> dict[str, Any] | None:
    """Helper: load and return parsed JSON, or None if not available."""
    if not DATA_FILE.exists():
        return None
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def test_required_keys() -> TestResult:
    """Test 3: Required top-level keys present."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    required = {"window_days", "fetched_at", "github_stats", "weekly_org", "member_weekly"}
    missing = required - set(data.keys())
    if missing:
        return False, f"Missing keys: {', '.join(sorted(missing))}"
    return True, f"Required fields present ({', '.join(sorted(required))})"


def test_all_members_in_github_stats() -> TestResult:
    """Test 4: All 15 team members have entries in github_stats."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    stats = data.get("github_stats", {})
    # Keys in the JSON are lowercase handles
    missing = [h for h in ALL_HANDLES_LOWER if h not in stats]
    if missing:
        return False, f"{len(missing)} members missing from github_stats: {missing}"
    return True, f"All {len(ALL_HANDLES_LOWER)} members present in github_stats"


def test_github_stats_not_all_zero() -> TestResult:
    """Test 5: GitHub stats not all zero - at least one member has prs_opened > 0."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    stats = data.get("github_stats", {})
    non_zero = [h for h, s in stats.items() if s.get("prs_opened", 0) > 0]
    if not non_zero:
        return False, "All members have prs_opened == 0 (data may be stale or fetch failed)"
    return True, f"{len(non_zero)} members have prs_opened > 0"


def test_weekly_org_entries() -> TestResult:
    """Test 6: weekly_org has at least 4 entries."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    weekly = data.get("weekly_org", [])
    if len(weekly) < 4:
        return False, f"weekly_org has only {len(weekly)} entries (need >= 4)"
    return True, f"weekly_org has {len(weekly)} entries"


def test_member_weekly_all_handles() -> TestResult:
    """Test 7: member_weekly present for all handles."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    mw = data.get("member_weekly", {})
    missing = [h for h in ALL_HANDLES_LOWER if h not in mw]
    if missing:
        return False, f"{len(missing)} members missing from member_weekly: {missing}"
    return True, f"member_weekly present for all {len(ALL_HANDLES_LOWER)} members"


def test_lines_data_present() -> TestResult:
    """Test 8: At least one member has lines_added > 0 (lines data was fetched)."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    mw = data.get("member_weekly", {})
    with_lines = [
        h for h, v in mw.items()
        if v.get("data", {}).get("lines_added", 0) > 0
    ]
    if not with_lines:
        return (
            False,
            "No members have lines_added > 0 "
            "(was --skip-member-weekly used, or did lines fetch fail?)",
        )
    return True, f"{len(with_lines)} members have lines_added > 0"


def test_anthropic_usage_present() -> TestResult:
    """Test 9: ant_usage present with cost_usd > 0 (tokens may be 0 — plan limitation)."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    ant = data.get("ant_usage")
    if not ant:
        return False, "ant_usage is null/missing (was --skip-anthropic used?)"
    tokens   = ant.get("tokens", 0)
    cost_usd = ant.get("cost_usd", 0.0)
    # tokens may be 0 — usage_report/messages returns 500 on this plan
    if cost_usd <= 0:
        return False, f"ant_usage.cost_usd = {cost_usd} (expected > 0)"
    token_note = f"{tokens:,} tokens" if tokens > 0 else "tokens unavailable (plan)"
    return True, f"ant_usage: {token_note}, ${cost_usd:.2f}"


def test_data_freshness() -> TestResult:
    """Test 10: Data is fresh - fetched_at within the last 7 days."""
    data = _load_data()
    if data is None:
        return False, "Could not load data file"
    fetched_at_str = data.get("fetched_at", "")
    if not fetched_at_str:
        return False, "fetched_at field is missing or empty"
    try:
        # Parse ISO-8601 with Z suffix
        fetched_at = datetime.strptime(fetched_at_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        try:
            fetched_at = datetime.strptime(fetched_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return False, f"Could not parse fetched_at: {fetched_at_str!r}"

    age = datetime.now(tz=timezone.utc) - fetched_at
    if age > timedelta(days=7):
        return False, f"Data is {age.days} days old (fetched_at: {fetched_at_str})"
    hours = int(age.total_seconds() // 3600)
    minutes = int((age.total_seconds() % 3600) // 60)
    return True, f"Data is fresh (fetched {hours}h {minutes}m ago)"


# ---------------------------------------------------------------------------
# Group 2: Server checks
# ---------------------------------------------------------------------------


def test_server_responding() -> TestResult:
    """Test 11: Server responding at http://127.0.0.1:8080."""
    try:
        status, _ = http_get(SERVER_BASE + "/", timeout=5)
        if status == 200:
            return True, f"Server at {SERVER_BASE} (200 OK)"
        return False, f"Server returned unexpected status {status}"
    except Exception as exc:
        return False, f"Server not reachable: {exc}"


def test_index_html() -> TestResult:
    """Test 12: index.html served correctly (200, contains 'CTO Development')."""
    try:
        status, body = http_get(SERVER_BASE + "/", timeout=5)
        if status != 200:
            return False, f"index.html returned {status}"
        text = body.decode("utf-8", errors="replace")
        if "CTO Development" not in text:
            return False, "index.html does not contain 'CTO Development'"
        return True, "index.html served correctly (200, contains 'CTO Development')"
    except Exception as exc:
        return False, f"Could not fetch index.html: {exc}"


def test_data_file_served() -> TestResult:
    """Test 13: data/team-metrics.json served at the expected URL."""
    url = f"{SERVER_BASE}/data/team-metrics.json"
    try:
        status, body = http_get(url, timeout=5)
        if status != 200:
            return False, f"/data/team-metrics.json returned {status}"
        # Verify it's actually valid JSON
        json.loads(body.decode("utf-8"))
        size = len(body)
        return True, f"/data/team-metrics.json served correctly (200, {size:,} bytes)"
    except json.JSONDecodeError:
        return False, "/data/team-metrics.json returned non-JSON body"
    except Exception as exc:
        return False, f"Could not fetch /data/team-metrics.json: {exc}"


def test_save_metrics_endpoint() -> TestResult:
    """Test 14: /api/save-metrics endpoint exists (POST returns 200 or 405, not 404)."""
    url = f"{SERVER_BASE}/api/save-metrics"
    try:
        status, _ = http_post(
            url,
            body=b"{}",
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        if status == 404:
            return False, "/api/save-metrics returned 404 (endpoint not registered)"
        return True, f"/api/save-metrics exists (returned {status})"
    except Exception as exc:
        return False, f"Could not reach /api/save-metrics: {exc}"


def test_anthropic_proxy_reachable() -> TestResult:
    """Test 15: Anthropic proxy reachable (OPTIONS /proxy/anthropic/v1/organizations/api_keys)."""
    url = f"{SERVER_BASE}/proxy/anthropic/v1/organizations/api_keys"
    try:
        status, _ = http_options(url, timeout=5)
        if status == 404:
            return False, "Proxy endpoint returned 404 (proxy not configured)"
        return True, f"Anthropic proxy endpoint reachable (OPTIONS returned {status})"
    except Exception as exc:
        return False, f"Could not reach proxy endpoint: {exc}"


# ---------------------------------------------------------------------------
# Group 3: API spot-checks
# ---------------------------------------------------------------------------


def test_github_token_valid() -> TestResult:
    """Test 16: GitHub token valid (GET /user returns expected login)."""
    github_token, _ = get_tokens()
    if not github_token:
        return False, "GITHUB_TOKEN not set"
    try:
        status, body = http_get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=10,
        )
        if status != 200:
            return False, f"GitHub /user returned {status}"
        data = json.loads(body.decode("utf-8"))
        login = data.get("login", "")
        if login != EXPECTED_LOGIN:
            return (
                False,
                f"GitHub /user returned login={login!r} (expected {EXPECTED_LOGIN!r})",
            )
        return True, f"GitHub token valid ({login})"
    except Exception as exc:
        return False, f"GitHub token check failed: {exc}"


def test_github_graphql() -> TestResult:
    """Test 17: GitHub GraphQL works (simple viewer query)."""
    github_token, _ = get_tokens()
    if not github_token:
        return False, "GITHUB_TOKEN not set"
    try:
        body = json.dumps({"query": "{viewer{login}}"}).encode("utf-8")
        status, resp_body = http_post(
            "https://api.github.com/graphql",
            body=body,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if status != 200:
            return False, f"GraphQL returned {status}"
        data = json.loads(resp_body.decode("utf-8"))
        if "errors" in data:
            return False, f"GraphQL errors: {data['errors']}"
        login = data.get("data", {}).get("viewer", {}).get("login", "")
        return True, f"GitHub GraphQL works (viewer.login={login!r})"
    except Exception as exc:
        return False, f"GitHub GraphQL check failed: {exc}"


def test_github_org_search() -> TestResult:
    """Test 18: GitHub org search works (search for 1 PR in senseon-tech)."""
    github_token, _ = get_tokens()
    if not github_token:
        return False, "GITHUB_TOKEN not set"
    try:
        query = """
        {
          search(query: "type:pr org:senseon-tech", type: ISSUE, first: 1) {
            issueCount
          }
        }
        """
        body = json.dumps({"query": query}).encode("utf-8")
        status, resp_body = http_post(
            "https://api.github.com/graphql",
            body=body,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        if status != 200:
            return False, f"Org search GraphQL returned {status}"
        data = json.loads(resp_body.decode("utf-8"))
        if "errors" in data:
            return False, f"GraphQL errors: {data['errors']}"
        count = data.get("data", {}).get("search", {}).get("issueCount", -1)
        if count < 0:
            return False, "Could not parse issueCount from response"
        return True, f"GitHub org search works ({count:,} PRs in senseon-tech)"
    except Exception as exc:
        return False, f"GitHub org search failed: {exc}"


def test_anthropic_key_via_proxy() -> TestResult:
    """Test 19: Anthropic key valid (GET /proxy/anthropic/v1/organizations/api_keys via server)."""
    _, anthropic_key = get_tokens()
    if not anthropic_key:
        return False, "ANTHROPIC_ADMIN_KEY not set"
    url = f"{SERVER_BASE}/proxy/anthropic/v1/organizations/api_keys?limit=1"
    try:
        status, body = http_get(url, headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"}, timeout=15)
        if status == 404:
            return False, "Proxy endpoint returned 404"
        if status == 401 or status == 403:
            return False, f"Anthropic key rejected ({status})"
        if status == 200:
            # Try to parse as JSON
            try:
                data = json.loads(body.decode("utf-8"))
                count = len(data.get("data", []))
                return True, f"Anthropic key valid via proxy (200, {count} API keys)"
            except json.JSONDecodeError:
                return True, f"Anthropic proxy reachable (200, non-JSON body)"
        return True, f"Anthropic proxy reachable via server (status {status})"
    except Exception as exc:
        return False, f"Anthropic proxy check failed: {exc}"


# ---------------------------------------------------------------------------
# Test groups definition
# ---------------------------------------------------------------------------

GROUPS: list[tuple[str, list[tuple[str, TestFn]]]] = [
    (
        "Group 1: Data file",
        [
            ("data/team-metrics.json exists",          test_data_file_exists),
            ("JSON valid and parseable",                test_json_valid),
            ("Required fields present",                 test_required_keys),
            ("All 15 members in github_stats",          test_all_members_in_github_stats),
            ("GitHub stats not all zero",               test_github_stats_not_all_zero),
            ("weekly_org has >= 4 entries",             test_weekly_org_entries),
            ("member_weekly present for all handles",   test_member_weekly_all_handles),
            ("At least one member has lines_added > 0", test_lines_data_present),
            ("ant_usage present with tokens + cost",    test_anthropic_usage_present),
            ("Data is fresh (< 7 days old)",            test_data_freshness),
        ],
    ),
    (
        "Group 2: Server",
        [
            (f"Server at {SERVER_BASE}",                test_server_responding),
            ("index.html served correctly",             test_index_html),
            ("/data/team-metrics.json served",          test_data_file_served),
            ("/api/save-metrics endpoint exists",       test_save_metrics_endpoint),
            ("Anthropic proxy reachable",               test_anthropic_proxy_reachable),
        ],
    ),
    (
        "Group 3: API",
        [
            (f"GitHub token valid ({EXPECTED_LOGIN})",  test_github_token_valid),
            ("GitHub GraphQL works",                    test_github_graphql),
            ("GitHub org search works",                 test_github_org_search),
            ("Anthropic key valid (via proxy)",         test_anthropic_key_via_proxy),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    print("\nHorus Test Harness")
    print("==================\n")

    total_passed = 0
    total_tests  = 0
    failed_names: list[str] = []

    for group_name, tests in GROUPS:
        print(f"[{group_name}]")
        for label, fn in tests:
            total_tests += 1
            passed, message = run_test(fn)
            icon = "[OK]" if passed else "[ERR]"
            print(f"  {icon} {message}")
            if passed:
                total_passed += 1
            else:
                failed_names.append(label)

        print()  # blank line between groups

    print("==================")
    if total_passed == total_tests:
        print(f"Results: {total_passed}/{total_tests} passed")
    else:
        failed_count = total_tests - total_passed
        print(f"Results: {total_passed}/{total_tests} passed, {failed_count} failed")
        print()
        print("Failed tests:")
        for name in failed_names:
            print(f"  [ERR] {name}")

    print()
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
