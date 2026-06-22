from __future__ import annotations

from neuralfabric import __version__


def test_version_exists() -> None:
    assert isinstance(__version__, str)
    assert len(__version__) > 0
