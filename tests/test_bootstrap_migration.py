import json

import pytest

from core.bootstrap import ServiceGraph


def test_canonical_startup_requires_verified_hindsight_tags(tmp_path):
    (tmp_path / "business.sqlite3").touch()
    migration_dir = tmp_path / "migrations" / "run-1"
    migration_dir.mkdir(parents=True)
    (migration_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "canonical_cutover",
                "mappings": [{"canonical_key": "agent:main:chat:qq:default:direct:1"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Hindsight"):
        ServiceGraph._assert_canonical_cutover_ready(tmp_path)


def test_canonical_startup_accepts_matching_verified_hindsight_tags(tmp_path):
    (tmp_path / "business.sqlite3").touch()
    migration_dir = tmp_path / "migrations" / "run-1"
    migration_dir.mkdir(parents=True)
    (migration_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "canonical_cutover",
                "mappings": [{"canonical_key": "agent:main:chat:qq:default:direct:1"}],
            }
        ),
        encoding="utf-8",
    )
    (migration_dir / "hindsight-tag-plan.json").write_text(
        json.dumps(
            {
                "source_run_id": "run-1",
                "bank_id": "qq_bot",
                "state": "verified",
            }
        ),
        encoding="utf-8",
    )

    ServiceGraph._assert_canonical_cutover_ready(tmp_path)


def test_canonical_startup_rejects_newer_unverified_migration(tmp_path):
    (tmp_path / "business.sqlite3").touch()
    old_dir = tmp_path / "migrations" / "run-1"
    old_dir.mkdir(parents=True)
    (old_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-1", "state": "verified", "created_at": 1}),
        encoding="utf-8",
    )
    (old_dir / "hindsight-tag-plan.json").write_text(
        json.dumps(
            {"source_run_id": "run-1", "bank_id": "qq_bot", "state": "verified"}
        ),
        encoding="utf-8",
    )
    new_dir = tmp_path / "migrations" / "run-2"
    new_dir.mkdir(parents=True)
    (new_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-2", "state": "applying", "created_at": 2}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="latest cold migration"):
        ServiceGraph._assert_canonical_cutover_ready(tmp_path)
