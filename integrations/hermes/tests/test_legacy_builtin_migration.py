from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from hermes_memory_provider import MnemosyneMemoryProvider as RootMnemosyneMemoryProvider

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider


class RecordingBeam:
    author_id = None

    def __init__(self, *args, db_path=None, **kwargs):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS working_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                scope TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def remember(self, **kwargs):
        self.conn.execute(
            "INSERT INTO working_memory(content, source, scope) VALUES (?, ?, ?)",
            (kwargs["content"], kwargs["source"], kwargs["scope"]),
        )
        self.conn.commit()
        return str(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])


class RacingBeam(RecordingBeam):
    gate = threading.Event()
    started = 0
    started_condition = threading.Condition()
    insert_lock = threading.Lock()

    def remember(self, **kwargs):
        with self.started_condition:
            type(self).started += 1
            self.started_condition.notify_all()
        self.gate.wait(timeout=2)
        with self.insert_lock:
            return super().remember(**kwargs)


class WriteOnlyBeam:
    def __init__(self):
        self.writes = []

    def remember(self, **kwargs):
        self.writes.append(kwargs)
        return "memory-id"


def test_initialize_imports_existing_builtin_memory_once_as_global(monkeypatch, tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text(
        "Windows environment fact\n§\nHermes workflow convention",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "User prefers What, Why, How, Recommendations",
        encoding="utf-8",
    )
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: RecordingBeam)

    provider = MnemosyneMemoryProvider()
    provider.initialize("first", hermes_home=str(tmp_path))
    provider.initialize("second", hermes_home=str(tmp_path))

    rows = provider._beam.conn.execute(
        "SELECT content, source, scope FROM working_memory ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("Windows environment fact", "builtin_memory_memory", "global"),
        ("Hermes workflow convention", "builtin_memory_memory", "global"),
        (
            "User prefers What, Why, How, Recommendations",
            "builtin_memory_user",
            "global",
        ),
    ]


def test_builtin_memory_mirror_is_global_for_both_targets():
    provider = MnemosyneMemoryProvider()
    provider._beam = WriteOnlyBeam()

    provider.on_memory_write("add", "memory", "environment fact")
    provider.on_memory_write("add", "user", "user preference")

    assert [write["scope"] for write in provider._beam.writes] == ["global", "global"]


def test_invalid_utf8_legacy_file_does_not_disable_provider(monkeypatch, tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_bytes(b"\xff\xfe")
    (memories / "USER.md").write_text("valid user preference", encoding="utf-8")
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: RecordingBeam)

    provider = MnemosyneMemoryProvider()
    provider.initialize("invalid-utf8", hermes_home=str(tmp_path))

    assert provider._beam is not None
    rows = provider._beam.conn.execute(
        "SELECT content, source, scope FROM working_memory ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("valid user preference", "builtin_memory_user", "global")
    ]


def test_root_provider_backfills_legacy_memory_idempotently(tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("root environment fact", encoding="utf-8")
    (memories / "USER.md").write_text("root user preference", encoding="utf-8")

    provider = RootMnemosyneMemoryProvider()
    provider._hermes_home = str(tmp_path)
    provider._beam = RecordingBeam(db_path=tmp_path / "root.db")

    provider._migrate_legacy_builtin_memories()
    provider._migrate_legacy_builtin_memories()

    rows = provider._beam.conn.execute(
        "SELECT content, source, scope FROM working_memory ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("root environment fact", "builtin_memory_memory", "global"),
        ("root user preference", "builtin_memory_user", "global"),
    ]


def test_root_provider_mirrors_both_builtin_targets_globally():
    provider = RootMnemosyneMemoryProvider()
    provider._beam = WriteOnlyBeam()

    provider.on_memory_write("add", "memory", "root environment fact")
    provider.on_memory_write("add", "user", "root user preference")

    assert [write["scope"] for write in provider._beam.writes] == ["global", "global"]


def test_completed_migration_does_not_resurrect_deleted_memory(tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("delete me after migration", encoding="utf-8")

    provider = MnemosyneMemoryProvider()
    provider._hermes_home = str(tmp_path)
    provider._beam = RecordingBeam(db_path=tmp_path / "no-resurrection.db")

    provider._migrate_legacy_builtin_memories()
    provider._beam.conn.execute("DELETE FROM working_memory")
    provider._beam.conn.commit()
    provider._migrate_legacy_builtin_memories()

    assert provider._beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0


def test_concurrent_migration_imports_each_entry_once(tmp_path):
    provider_a = MnemosyneMemoryProvider()
    provider_b = MnemosyneMemoryProvider()
    db_path = tmp_path / "concurrent.db"
    provider_a._hermes_home = str(tmp_path)
    provider_b._hermes_home = str(tmp_path)
    provider_a._beam = RacingBeam(db_path=db_path)
    provider_b._beam = RacingBeam(db_path=db_path)

    # Let the fixed implementation initialize its migration ledger before the
    # concurrent import. The pre-fix implementation treats this as a no-op.
    provider_a._migrate_legacy_builtin_memories()
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_text("concurrent environment fact", encoding="utf-8")

    RacingBeam.gate.clear()
    RacingBeam.started = 0
    threads = [
        threading.Thread(target=provider._migrate_legacy_builtin_memories)
        for provider in (provider_a, provider_b)
    ]
    for thread in threads:
        thread.start()
    with RacingBeam.started_condition:
        RacingBeam.started_condition.wait_for(lambda: RacingBeam.started >= 2, timeout=0.5)
    RacingBeam.gate.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    count = provider_a._beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?",
        ("concurrent environment fact",),
    ).fetchone()[0]
    assert count == 1


def test_crlf_legacy_delimiters_import_as_separate_entries(tmp_path):
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "MEMORY.md").write_bytes(
        "first fact\r\n§\r\nsecond fact".encode()
    )

    provider = MnemosyneMemoryProvider()
    provider._hermes_home = str(tmp_path)
    provider._beam = RecordingBeam(db_path=tmp_path / "crlf.db")
    provider._migrate_legacy_builtin_memories()

    rows = provider._beam.conn.execute(
        "SELECT content FROM working_memory ORDER BY id"
    ).fetchall()
    assert [row[0] for row in rows] == ["first fact", "second fact"]
