"""Git metadata helpers must remain optional in stripped runtime images."""

from __future__ import annotations

from nequip.utils import git


def test_get_commit_returns_none_when_git_is_unavailable(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(git.subprocess, "run", missing_git)
    assert git.get_commit("nequip") is None
