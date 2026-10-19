"""Regression tests for ``ghstats repos -n N`` star ranking (issue #84).

``fetch_user_repos`` used to stop collecting as soon as ``limit`` repos were
pooled, so the star sort ran over an already-truncated pool in GitHub's
default API order. "Top Repos" then showed the API-order head of the listing
(all zero stars) while the account's most-starred repos — further down in API
order — were truncated away before the sort ever saw them, with exit code 0
and no warning. The pool must span the whole pageable account list; only the
final sort-then-cap narrows the result to ``limit``.

The mock below bisects the endpoint query string exactly like the real API
would: a fixed account served in a fixed (non-star) order, paged by
``per_page``/``page``.
"""
import json
import subprocess

import pytest

from ghstats.fetcher import StatsFetcher


def _rest_repo(name, stars, fork=False):
    return {
        "name": name,
        "full_name": f"acme/{name}",
        "fork": fork,
        "stargazers_count": stars,
        "forks_count": 0,
        "language": "Python",
        "description": f"{name} ({stars} stars)",
        "updated_at": "2026-01-01T00:00:00Z",
    }


# The issue's 18-repo account: the most-starred repos sit LAST in API order,
# preceded by fifteen zero-star repos. No forks, so filtering is inert and the
# test isolates pure ordering (the #81 duplicate-rows defect is not in play).
API_ORDER_ACCOUNT = [
    _rest_repo(f"{name}-zero", 0)
    for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                 "golf", "hotel", "india", "juliet", "kilo", "lima",
                 "mike", "november", "oscar")
] + [
    _rest_repo("starred-top", 500),
    _rest_repo("starred-second", 400),
    _rest_repo("starred-third", 300),
]


class ApiOrderLister:
    """A ONE-page account: every repo is served on page 1 in API order."""

    def __init__(self, account):
        self.account = account
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        endpoint = cmd[2]
        per_page = int(endpoint.split("per_page=")[1].split("&")[0])
        # `page=` is the last parameter, but a bare split on "page=" also
        # matches inside "per_page=", so anchor on the last occurrence.
        page = int(endpoint.rsplit("page=", 1)[1])
        chunk = self.account[(page - 1) * per_page:page * per_page]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(chunk), stderr="")


@pytest.fixture
def lister(monkeypatch):
    run = ApiOrderLister(API_ORDER_ACCOUNT)
    monkeypatch.setattr(subprocess, "run", run)
    return run


class TestTopReposAreSelectedAfterSorting:
    def test_limit_3_returns_the_three_most_starred(self, lister):
        """-n 3 must surface starred-top/second/third, not the API-order head."""
        repos = StatsFetcher("acme").fetch_user_repos(limit=3)
        assert [r["name"] for r in repos] == [
            "starred-top", "starred-second", "starred-third",
        ]
        assert [r["stars"] for r in repos] == [500, 400, 300]

    def test_zero_star_repos_do_not_displace_starred_ones(self, lister):
        repos = StatsFetcher("acme").fetch_user_repos(limit=3)

        assert all(r["stars"] > 0 for r in repos), repos

    def test_the_pool_spans_the_whole_account_before_the_cap(self, lister):
        """One 18-repo account needs one whole page — one call, full page size.

        The old code asked for ``per_page=min(MAX, limit)`` and stopped the walk
        after ``limit`` repos, capping the pool in API order before the sort.
        With the account served on a single page, only an uncapped pool can
        rank the account correctly.
        """
        StatsFetcher("acme").fetch_user_repos(limit=3)

        assert len(lister.calls) == 1, lister.calls
        assert "per_page=100" in lister.calls[0][2], lister.calls[0][2]

    def test_starred_repos_hidden_behind_a_full_first_page_are_found(self, monkeypatch):
        """A full first page must not fence off page 2's high-star repos.

        Page 1 is a whole ``_MAX_PER_PAGE``-sized page of zero-star repos plus
        a fork; page 2 holds the account's most-starred repos. The walk must
        continue (stable per_page, monotonic offsets — both #81 invariants)
        and the final sort must pull the page-2 repos to the top.
        """
        page_one = [_rest_repo("fork1", 0, fork=True)] + [
            _rest_repo(f"filler-{i:03d}", 0) for i in range(99)
        ]
        page_two = [
            _rest_repo("starred-top", 500),
            _rest_repo("starred-second", 400),
            _rest_repo("starred-third", 300),
        ]
        account = page_one + page_two

        run = ApiOrderLister(account)
        monkeypatch.setattr(subprocess, "run", run)

        repos = StatsFetcher("acme").fetch_user_repos(limit=3)

        assert [r["name"] for r in repos] == [
            "starred-top", "starred-second", "starred-third",
        ]
        assert len(run.calls) == 2, run.calls  # two API pages, not dozens
        assert "per_page=100&page=1" in run.calls[0][2], run.calls[0][2]
        assert "per_page=100&page=2" in run.calls[1][2], run.calls[1][2]
