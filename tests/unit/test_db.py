"""Tests for the alert disposition store (dashboard/db.py) against a temp database."""

import pytest

from dashboard import db


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / "alert_state_test.db")
    db.init_db(db_path=path)
    return path


def test_init_db_is_idempotent(db_path):
    db.init_db(db_path=db_path)  # second call must not raise


def test_upsert_inserts_then_overwrites(db_path):
    db.upsert_disposition("user01", "2010-01-05", "open", "first look", db_path=db_path)
    db.upsert_disposition("user01", "2010-01-05", "closed", "benign", db_path=db_path)

    rows = db.get_all_dispositions(db_path=db_path)
    assert len(rows) == 1  # upsert keyed on (user, day) — no duplicate row
    assert rows[0]["status"] == "closed"
    assert rows[0]["note"] == "benign"


def test_get_disposition_roundtrip(db_path):
    db.upsert_disposition("user02", "2010-01-06", "escalated", db_path=db_path)
    row = db.get_disposition("user02", "2010-01-06", db_path=db_path)
    assert row is not None
    assert row["status"] == "escalated"
    assert row["note"] == ""


def test_get_disposition_missing_returns_none(db_path):
    assert db.get_disposition("ghost", "2010-01-01", db_path=db_path) is None


def test_get_all_orders_newest_first(db_path):
    db.upsert_disposition("user01", "2010-01-01", "open", db_path=db_path)
    db.upsert_disposition("user02", "2010-01-02", "open", db_path=db_path)
    rows = db.get_all_dispositions(db_path=db_path)
    assert [r["user"] for r in rows] == ["user02", "user01"]
