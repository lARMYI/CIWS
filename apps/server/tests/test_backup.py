"""M5: a workspace that outlives the machine it started on.

CIWS_HOME was portable by copy from the first commit. Portable is not durable,
and these tests pin the difference: a snapshot taken while the hub is running is
whole, a backup is opened before it is trusted, an archive can be encrypted, and
a restore does not cost you the credential vault that is already on the machine.

One thing deliberately not claimed: the live database is not encrypted at rest.
That needs SQLCipher, which is a build of SQLite rather than a Python package.
``encryption_status`` says so, and a test holds it to saying so -- quietly
implying protection that is not there would be worse than not offering it.
"""

from __future__ import annotations

import sqlite3
import tarfile
from pathlib import Path

import pytest

from ciws import backup
from ciws.core import paths, secrets
from ciws.core.errors import ValidationFailed
from ciws.memory import store
from ciws.ontology import graph


async def _populate() -> None:
    """Enough of a workspace that a lost table would be visible."""
    await store.remember("Deploys run Fridays at 16:00 UTC.", kind="procedure", importance=0.8)
    await store.remember("The staging cluster lives in eu-west-2.", kind="fact")
    person = await graph.upsert_entity("person", "Sarah Chen")
    project = await graph.upsert_entity("project", "Helios Migration")
    await graph.link(person.id, project.id, "owns")


# ---------------------------------------------------------------------------
# Create and verify
# ---------------------------------------------------------------------------


async def test_a_backup_round_trips_and_verifies(tmp_path: Path):
    await _populate()

    archive = tmp_path / "workspace.ciws"
    report = backup.create(archive)

    assert archive.exists()
    assert report["bytes"] > 0
    assert report["encrypted"] is False
    assert report["counts"]["memories"] == 2

    checked = backup.verify(archive)
    assert checked["ok"] is True
    assert checked["integrity"] == "ok"
    assert checked["counts"]["memories"] == 2
    assert checked["counts"]["entities"] == 2


async def test_the_snapshot_carries_committed_writes(tmp_path: Path):
    """WAL keeps recent commits in a sidecar; a naive file copy can miss them.

    This is the failure the sqlite backup API and the checkpoint exist to
    prevent, and it is invisible until the day you restore.
    """
    await _populate()
    archive = tmp_path / "first.ciws"
    backup.create(archive)

    await store.remember("A fact written after the first backup.", kind="fact")
    second = tmp_path / "second.ciws"
    report = backup.create(second)

    assert report["counts"]["memories"] == 3, "a committed write was missing from the snapshot"


async def test_assets_and_corpus_travel_with_the_database(tmp_path: Path):
    (paths.corpus_dir() / "handbook.md").write_text("## Retention\n90 days.\n", "utf-8")
    (paths.assets_dir() / "picture.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    archive = tmp_path / "full.ciws"
    backup.create(archive)

    # Look inside rather than trusting the manifest.
    staging = tmp_path / "peek"
    staging.mkdir()
    payload = staging / "payload.tar.gz"
    payload.write_bytes(archive.read_bytes())
    with tarfile.open(payload, "r:gz") as tar:
        names = set(tar.getnames())

    assert any(n.endswith("handbook.md") for n in names), "the corpus was left behind"
    assert any(n.endswith("picture.png") for n in names), "assets were left behind"


async def test_logs_and_cache_are_not_carried(tmp_path: Path):
    """Both regenerate, and both are most of the bytes."""
    (paths.logs_dir() / "ciws.log").write_text("noise" * 1000, "utf-8")
    (paths.cache_dir() / "scratch.bin").write_bytes(b"0" * 10_000)

    archive = tmp_path / "lean.ciws"
    backup.create(archive)

    staging = tmp_path / "peek2"
    staging.mkdir()
    payload = staging / "payload.tar.gz"
    payload.write_bytes(archive.read_bytes())
    with tarfile.open(payload, "r:gz") as tar:
        names = " ".join(tar.getnames())

    assert "ciws.log" not in names
    assert "scratch.bin" not in names


async def test_verify_rejects_a_damaged_archive(tmp_path: Path):
    await _populate()
    archive = tmp_path / "damaged.ciws"
    backup.create(archive)

    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive.write_bytes(bytes(raw))

    with pytest.raises(Exception):
        backup.verify(archive)


async def test_verify_rejects_something_that_is_not_a_backup(tmp_path: Path):
    impostor = tmp_path / "notabackup.ciws"
    with tarfile.open(impostor, "w:gz") as tar:
        payload = tmp_path / "hello.txt"
        payload.write_text("not a workspace", "utf-8")
        tar.add(payload, arcname="hello.txt")

    with pytest.raises(ValidationFailed):
        backup.verify(impostor)


async def test_verify_on_a_missing_file_says_so(tmp_path: Path):
    with pytest.raises(ValidationFailed):
        backup.verify(tmp_path / "never-created.ciws")


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


async def test_an_encrypted_archive_is_not_readable_without_the_passphrase(tmp_path: Path):
    await store.remember("A distinctive marker phrase for grepping.", kind="fact")

    archive = tmp_path / "secret.ciws"
    report = backup.create(archive, passphrase="correct horse battery staple")
    assert report["encrypted"] is True

    raw = archive.read_bytes()
    assert b"distinctive marker phrase" not in raw, "the archive stored plaintext"
    assert raw.startswith(b"CIWSENC1")

    checked = backup.verify(archive, passphrase="correct horse battery staple")
    assert checked["ok"] is True


async def test_a_wrong_passphrase_is_refused_clearly(tmp_path: Path):
    await _populate()
    archive = tmp_path / "secret2.ciws"
    backup.create(archive, passphrase="the right one")

    with pytest.raises(ValidationFailed) as caught:
        backup.verify(archive, passphrase="the wrong one")
    assert "passphrase" in str(caught.value).lower()


async def test_an_encrypted_archive_without_a_passphrase_says_what_is_wrong(tmp_path: Path):
    await _populate()
    archive = tmp_path / "secret3.ciws"
    backup.create(archive, passphrase="hunter2")

    with pytest.raises(ValidationFailed) as caught:
        backup.verify(archive)
    assert "encrypted" in str(caught.value).lower()


def test_encryption_status_does_not_overstate_itself():
    status = backup.encryption_status()
    assert status["vault"]["encrypted"] is True
    assert status["database"]["encrypted"] is False, (
        "claiming the live database is encrypted would let someone store things "
        "in it that they otherwise would not"
    )
    assert "SQLCipher" in status["database"]["detail"]


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


async def test_restore_reproduces_the_workspace_elsewhere(tmp_path: Path):
    await _populate()
    (paths.corpus_dir() / "notes.md").write_text("Something ingested.", "utf-8")

    archive = tmp_path / "move.ciws"
    backup.create(archive)

    destination = tmp_path / "other-machine"
    result = backup.restore(archive, into=destination)

    assert result["ok"] is True
    moved_db = destination / "data" / "ciws.db"
    assert moved_db.exists()
    assert (destination / "corpus" / "notes.md").read_text("utf-8") == "Something ingested."

    connection = sqlite3.connect(str(moved_db))
    try:
        memories = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        edges = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    finally:
        connection.close()
    assert memories == 2
    assert edges == 1, "the graph did not survive the move"


async def test_restore_keeps_the_vault_already_on_this_machine(tmp_path: Path):
    """Moving a workspace must not cost you every stored key."""
    await _populate()
    archive = tmp_path / "novault.ciws"
    backup.create(archive)

    destination = tmp_path / "machine-with-keys"
    (destination).mkdir()
    (destination / "vault.enc").write_bytes(b"an existing encrypted vault")

    backup.restore(archive, into=destination)
    assert (destination / "vault.enc").read_bytes() == b"an existing encrypted vault"


async def test_a_backup_never_carries_credentials(tmp_path: Path):
    """The vault key beside the vault is not an encrypted vault."""
    secrets.put("openai", "sk-should-never-be-in-a-backup")
    try:
        archive = tmp_path / "nokeys.ciws"
        backup.create(archive)
        raw = archive.read_bytes()
        assert b"sk-should-never-be-in-a-backup" not in raw

        staging = tmp_path / "peek3"
        staging.mkdir()
        payload = staging / "payload.tar.gz"
        payload.write_bytes(raw)
        with tarfile.open(payload, "r:gz") as tar:
            names = " ".join(tar.getnames())
        assert "vault.key" not in names
        assert "vault.enc" not in names
    finally:
        secrets.delete("openai")


async def test_restore_moves_the_old_database_aside_rather_than_deleting_it(tmp_path: Path):
    await _populate()
    archive = tmp_path / "replace.ciws"
    backup.create(archive)

    destination = tmp_path / "occupied"
    (destination / "data").mkdir(parents=True)
    (destination / "data" / "ciws.db").write_bytes(b"the previous database")

    backup.restore(archive, into=destination)

    kept = list((destination / "data").glob("ciws.db.replaced-*"))
    assert kept, "the previous database was deleted rather than kept aside"
    assert kept[0].read_bytes() == b"the previous database"


async def test_a_traversal_entry_in_an_archive_is_refused(tmp_path: Path):
    """A tar member is untrusted input."""
    evil = tmp_path / "evil.ciws"
    victim = tmp_path / "escaped.txt"
    payload = tmp_path / "payload.txt"
    payload.write_text("pwned", "utf-8")

    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")

    with pytest.raises(ValidationFailed) as caught:
        backup.restore(evil, into=tmp_path / "target")
    assert "outside" in str(caught.value).lower()
    assert not victim.exists()


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------


async def test_backup_and_verify_through_the_api(api, tmp_path: Path):
    await _populate()
    destination = tmp_path / "api.ciws"

    created = await api.post("/api/system/backup", json={"path": str(destination)})
    assert created.status_code == 200
    body = created.json()
    assert body["verified"]["ok"] is True, "the route returned an unverified backup"
    assert destination.exists()

    checked = await api.post("/api/system/backup/verify", json={"path": str(destination)})
    assert checked.status_code == 200
    assert checked.json()["ok"] is True


async def test_backup_status_reports_what_is_encrypted(api):
    response = await api.get("/api/system/backup")
    assert response.status_code == 200
    body = response.json()
    assert body["encryption"]["database"]["encrypted"] is False
    assert "backups" in body


async def test_restore_without_a_path_is_a_typed_error(api):
    response = await api.post("/api/system/restore", json={})
    assert response.status_code == 422  # ValidationFailed
    body = response.json()
    assert body["error"] == "validation_failed"
    assert "path" in body["message"].lower(), "the error did not say what was missing"
