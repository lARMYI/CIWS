"""Backup, restore, and moving a workspace between machines.

``CIWS_HOME`` is already portable by copy -- that was true from the first
commit. Portable is not the same as durable, and the gap this closes is:

* **A copy taken while the hub is running can be torn.** SQLite in WAL mode
  keeps recent commits in a sidecar file; copying only ``ciws.db`` can capture a
  database missing its most recent writes. This checkpoints first and uses
  SQLite's own backup API, which is safe against a live writer.
* **A backup is only as good as its restore.** ``verify`` opens the archive and
  reads the row counts back, so "the backup worked" is a checked statement
  rather than a hopeful one.
* **A workspace on a USB stick is plaintext.** The credential vault is
  encrypted; everything else -- memories, documents, conversations -- is not.
  An archive can now be encrypted with a passphrase.

On encrypting the *live* database: that needs SQLCipher, which is a build-time
dependency rather than a Python one, so this module does not pretend to offer
it. ``encryption_status`` says so plainly instead of implying protection that
is not there.
"""

from __future__ import annotations

import base64
import json
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .core import paths
from .core.errors import ValidationFailed
from .core.logging import get_logger
from .core.util import now

log = get_logger("backup")

MANIFEST = "ciws-backup.json"
FORMAT_VERSION = 1

#: Directories carried in an archive, relative to CIWS_HOME. `cache` and `logs`
#: are deliberately absent: both regenerate, and both are the bulk of the bytes.
INCLUDED_DIRS = ("assets", "corpus", "workspace")

#: Files carried alongside the database.
INCLUDED_FILES = ("config.json",)

#: PBKDF2 rounds for a passphrase-derived key. Slow on purpose.
KDF_ROUNDS = 480_000


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


def _key_from(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=KDF_ROUNDS)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def encryption_status() -> dict[str, Any]:
    """What is and is not encrypted at rest, stated plainly.

    Overstating this would be worse than not offering it: someone who believes
    their whole workspace is encrypted will put things in it that they would
    not otherwise.
    """
    return {
        "vault": {
            "encrypted": True,
            "detail": "Credentials are encrypted with Fernet; the key sits beside the vault at 0600.",
        },
        "database": {
            "encrypted": False,
            "detail": (
                "The SQLite file is plaintext on disk. Encrypting it requires SQLCipher, "
                "which is a build of SQLite rather than a Python package, so CIWS does not "
                "claim it. Use full-disk encryption for the live workspace."
            ),
        },
        "backups": {
            "encrypted": "optional",
            "detail": "Pass a passphrase to create() and the archive is encrypted with Fernet.",
        },
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def _snapshot_database(destination: Path) -> dict[str, int]:
    """Copy the live database safely and return its table counts.

    ``sqlite3.Connection.backup`` is the supported way to copy a database that
    something else may be writing to; a filesystem copy of a WAL database can
    miss the sidecar and land a torn file.
    """
    source_path = paths.db_file()
    if not source_path.exists():
        raise ValidationFailed(f"No database at {source_path}")

    source = sqlite3.connect(str(source_path))
    try:
        # Fold the WAL back into the main file first so the snapshot is whole
        # even if the sidecar is not carried with it.
        source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target = sqlite3.connect(str(destination))
        try:
            source.backup(target)
            counts = _counts(target)
        finally:
            target.close()
    finally:
        source.close()
    return counts


def _counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (name,) in rows:
        # FTS shadow tables are derived; counting them says nothing useful.
        if name.endswith(("_data", "_idx", "_docsize", "_config", "_content")):
            continue
        try:
            counts[name] = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.DatabaseError:
            continue
    return counts


def create(destination: Path | str, *, passphrase: str = "") -> dict[str, Any]:
    """Write a portable archive of the whole workspace."""
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ciws-backup-") as staging_name:
        staging = Path(staging_name)
        db_copy = staging / "ciws.db"
        counts = _snapshot_database(db_copy)

        manifest = {
            "format": FORMAT_VERSION,
            "created_at": now().isoformat(),
            "home": str(paths.home()),
            "counts": counts,
            "encrypted": bool(passphrase),
            # Recorded but never carried: a backup that contains the vault key
            # alongside the vault is not an encrypted vault.
            "includes_credentials": False,
        }
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2), "utf-8")

        for name in INCLUDED_FILES:
            source = paths.home() / name
            if source.exists():
                shutil.copy2(source, staging / name)

        for name in INCLUDED_DIRS:
            source = paths.home() / name
            if source.is_dir():
                shutil.copytree(source, staging / name, dirs_exist_ok=True)

        archive = staging / "payload.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for entry in sorted(staging.iterdir()):
                if entry.name == "payload.tar.gz":
                    continue
                tar.add(entry, arcname=entry.name)

        payload = archive.read_bytes()

    if passphrase:
        salt = base64.urlsafe_b64decode(Fernet.generate_key())[:16]
        token = Fernet(_key_from(passphrase, salt)).encrypt(payload)
        # Salt travels in the clear ahead of the ciphertext; that is what a salt
        # is for, and it means restore needs only the passphrase.
        destination.write_bytes(b"CIWSENC1" + salt + token)
    else:
        destination.write_bytes(payload)

    size = destination.stat().st_size
    log.info("Backup written to %s (%d bytes)", destination, size)
    return {
        "path": str(destination),
        "bytes": size,
        "encrypted": bool(passphrase),
        "counts": counts,
        "created_at": now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Read back
# ---------------------------------------------------------------------------


def _payload_of(archive: Path, passphrase: str) -> bytes:
    raw = archive.read_bytes()
    if raw.startswith(b"CIWSENC1"):
        if not passphrase:
            raise ValidationFailed("This backup is encrypted. Supply its passphrase.")
        salt, token = raw[8:24], raw[24:]
        try:
            return Fernet(_key_from(passphrase, salt)).decrypt(token)
        except InvalidToken as exc:
            raise ValidationFailed("Wrong passphrase, or the archive is damaged.") from exc
    if passphrase:
        raise ValidationFailed("This backup is not encrypted; no passphrase is needed.")
    return raw


def _extract(archive: Path, passphrase: str, into: Path) -> dict[str, Any]:
    payload = _payload_of(archive, passphrase)
    bundle = into / "payload.tar.gz"
    bundle.write_bytes(payload)

    with tarfile.open(bundle, "r:gz") as tar:
        for member in tar.getmembers():
            # A tar member is untrusted input: a path with .. or a leading slash
            # writes outside the destination.
            target = (into / member.name).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise ValidationFailed(f"Refusing an archive entry outside the target: {member.name}")
        tar.extractall(into)  # noqa: S202 - every member checked above

    manifest_file = into / MANIFEST
    if not manifest_file.exists():
        raise ValidationFailed("Not a CIWS backup: no manifest inside the archive.")
    return json.loads(manifest_file.read_text("utf-8"))


def verify(archive: Path | str, *, passphrase: str = "") -> dict[str, Any]:
    """Open an archive and read its contents back.

    A backup nobody has opened is a hope, not a backup. This is deliberately
    cheap enough to run right after ``create``.
    """
    archive = Path(archive).expanduser()
    if not archive.exists():
        raise ValidationFailed(f"No backup at {archive}")

    with tempfile.TemporaryDirectory(prefix="ciws-verify-") as staging_name:
        staging = Path(staging_name)
        manifest = _extract(archive, passphrase, staging)

        database = staging / "ciws.db"
        if not database.exists():
            raise ValidationFailed("The archive carries no database.")

        connection = sqlite3.connect(str(database))
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            actual = _counts(connection)
        finally:
            connection.close()

        if integrity != "ok":
            raise ValidationFailed(f"The archived database failed its integrity check: {integrity}")

        recorded = manifest.get("counts") or {}
        drift = {
            table: {"recorded": count, "actual": actual.get(table)}
            for table, count in recorded.items()
            if actual.get(table) != count
        }
        if drift:
            raise ValidationFailed(f"The archive does not match its manifest: {drift}")

        return {
            "ok": True,
            "format": manifest.get("format"),
            "created_at": manifest.get("created_at"),
            "encrypted": manifest.get("encrypted", False),
            "counts": actual,
            "integrity": integrity,
        }


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def restore(
    archive: Path | str, *, passphrase: str = "", into: Path | str | None = None
) -> dict[str, Any]:
    """Unpack a workspace, keeping the credential vault that is already there.

    The vault is never carried in an archive, so restoring must not delete the
    one on the destination machine -- otherwise moving a workspace silently
    costs you every stored key.
    """
    archive = Path(archive).expanduser()
    target = Path(into).expanduser() if into else paths.home()
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ciws-restore-") as staging_name:
        staging = Path(staging_name)
        manifest = _extract(archive, passphrase, staging)

        database = staging / "ciws.db"
        if not database.exists():
            raise ValidationFailed("The archive carries no database.")

        data = target / "data"
        data.mkdir(parents=True, exist_ok=True)

        # Move the old database aside rather than deleting it. A restore is the
        # moment someone is most likely to have picked the wrong archive.
        live = data / "ciws.db"
        if live.exists():
            stamp = now().strftime("%Y%m%d-%H%M%S")
            live.rename(data / f"ciws.db.replaced-{stamp}")
        for sidecar in ("ciws.db-wal", "ciws.db-shm"):
            (data / sidecar).unlink(missing_ok=True)

        shutil.copy2(database, live)

        for name in INCLUDED_FILES:
            source = staging / name
            if source.exists():
                shutil.copy2(source, target / name)

        for name in INCLUDED_DIRS:
            source = staging / name
            if source.is_dir():
                shutil.copytree(source, target / name, dirs_exist_ok=True)

    log.info("Restored workspace into %s", target)
    return {
        "ok": True,
        "home": str(target),
        "created_at": manifest.get("created_at"),
        "counts": manifest.get("counts") or {},
        "vault_preserved": (target / "vault.enc").exists(),
    }
