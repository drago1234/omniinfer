# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import time
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from unittest.mock import patch

from omni_npu.connector.mm_feature_transfer.mm_feature_connector.disk_connector import SQLiteMetadataStore


# ---------- Test SQLiteMetadataStore ----------
class TestSQLiteMetadataStore:
    @pytest.fixture
    def store(self):
        db = SQLiteMetadataStore(":memory:")
        yield db
        db.conn.close()

    @pytest.mark.parametrize(("sql", "fetch", "expected"), [
        ("UPDATE features SET refcount = 0", None, []),
        ("SELECT 1 UNION ALL SELECT 2", None, []),
        ("SELECT 1 UNION ALL SELECT 2", "one", [(1,)]),
        ("SELECT 1 UNION ALL SELECT 2", "all", [(1,), (2,)]),
        ("SELECT 1 WHERE 0", "one", []),
        ("SELECT 1 WHERE 0", "all", []),
    ])
    def test_execute_returns_rows_as_list(self, store, sql, fetch, expected):
        rows = store._run_with_retry(
            lambda: store._execute(sql, fetch=fetch), write=fetch is None
        )
        assert isinstance(rows, list)
        assert rows == expected

    def test_add_feature(self, store):
        store.add_feature("hash1", 100)
        cur = store.conn.execute("SELECT size, refcount, last_used FROM features WHERE mm_hash='hash1'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 100
        assert row[1] == 0
        assert row[2] > 0  

    def test_inc_refcount(self, store):
        store.add_feature("h1", 50)
        store.inc_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 1
        store.inc_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 2

    def test_dec_refcount(self, store):
        store.add_feature("h1", 10)
        store.inc_refcount("h1")
        store.inc_refcount("h1")
        store.dec_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 1
        store.dec_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 0
        store.dec_refcount("h1")  # refcount 为0时不减
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 0

    def test_get_refcount(self, store):
        assert store.get_refcount("missing") is None
        store.add_feature("h1", 5)
        assert store.get_refcount("h1") == 0
        store.inc_refcount("h1")
        assert store.get_refcount("h1") == 1

    def test_get_total_used(self, store):
        assert store.get_total_used() == 0
        store.add_feature("h1", 10)
        store.add_feature("h2", 20)
        assert store.get_total_used() == 30

    def test_get_eviction_candidates(self, store):
        store.add_feature("a", 100)
        time.sleep(0.01)
        store.add_feature("b", 200)
        time.sleep(0.01)
        store.add_feature("c", 300)
        store.inc_refcount("b")

        candidates = store.get_eviction_candidates(150)
        assert candidates == [("a", 100), ("c", 300)]  # Total size reaches 400，> 150

        candidates = store.get_eviction_candidates(50)
        assert candidates == [("a", 100)]

        candidates = store.get_eviction_candidates(1000)
        assert candidates == [("a", 100), ("c", 300)]

    def test_delete_feature(self, store):
        store.add_feature("h1", 10)
        store.delete_feature("h1")
        assert store.get_refcount("h1") is None

    def test_delete_old_features(self, store):
        store.add_feature("old", 10)
        time.sleep(1)
        now = int(time.time())
        store.add_feature("new", 20)
        deleted = store.delete_old_features(now)
        assert deleted == ["old"]
        assert store.get_refcount("old") is None
        assert store.get_refcount("new") == 0

        deleted = store.delete_old_features(now)
        assert deleted == []

    def test_delete_old_features_skips_refcount_positive(self, store):
        store.add_feature("h1", 10)
        store.inc_refcount("h1")
        old_time = int(time.time()) - 100
        store.conn.execute("UPDATE features SET last_used = ? WHERE mm_hash='h1'", (old_time,))
        store.conn.commit()
        deleted = store.delete_old_features(old_time + 10)
        assert deleted == []  
        assert store.get_refcount("h1") == 1

    def test_duplicate_insert_releases_writer_transaction(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            with SQLiteMetadataStore(store.db_path, cleanup_existing=False) as other:
                store.add_feature("h1", 10)
                with pytest.raises(sqlite3.IntegrityError):
                    store.add_feature("h1", 20)

                assert not store.conn.in_transaction
                other.inc_refcount("h1")
                assert store.get_refcount("h1") == 1

    def test_busy_snapshot_rolls_back_and_retries_complete_operation(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            with SQLiteMetadataStore(store.db_path, cleanup_existing=False) as other:
                store.add_feature("h1", 10)
                attempts = []
                errors = []

                def operation():
                    # Deliberately create a stale WAL snapshot, as in the incident.
                    store._execute("BEGIN")
                    before = store._execute(
                        "SELECT refcount FROM features WHERE mm_hash = ?",
                        ("h1",), fetch="one",
                    )[0][0]
                    attempts.append(before)
                    if len(attempts) == 1:
                        other.inc_refcount("h1")
                    try:
                        store._execute(
                            "UPDATE features SET refcount = refcount + 1 WHERE mm_hash = ?",
                            ("h1",),
                        )
                    except sqlite3.OperationalError as exc:
                        errors.append(exc.sqlite_errorcode)
                        raise
                    store.conn.commit()

                store._run_with_retry(operation)
                assert errors == [sqlite3.SQLITE_BUSY_SNAPSHOT]
                assert attempts == [0, 1]
                assert store.get_refcount("h1") == 2
                assert not store.conn.in_transaction
                # The old snapshot must no longer prevent WAL checkpointing.
                assert other.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0

    def test_commit_failure_retries_without_double_increment(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            store.add_feature("h1", 10)
            connection = store.conn

            class FailFirstCommit:
                commits = 0
                rollbacks = 0

                def __getattr__(self, name):
                    return getattr(connection, name)

                def commit(self):
                    self.commits += 1
                    if self.commits == 1:
                        raise sqlite3.OperationalError("database is locked")
                    connection.commit()

                def rollback(self):
                    self.rollbacks += 1
                    connection.rollback()

            store.conn = proxy = FailFirstCommit()
            store.inc_refcount("h1")
            assert proxy.commits == 2
            assert proxy.rollbacks == 1
            assert store.get_refcount("h1") == 1
            assert not connection.in_transaction

    def test_retry_exhaustion_rolls_back_partial_work(self, store):
        store.add_feature("h1", 10)
        store._max_retry_attempts = 2

        def failing_operation():
            store._execute("UPDATE features SET refcount = refcount + 1")
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError):
            store._run_with_retry(failing_operation, write=True)
        assert not store.conn.in_transaction
        assert store.get_refcount("h1") == 0

    def test_shared_connection_and_multiple_writers(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            with SQLiteMetadataStore(store.db_path, cleanup_existing=False) as other:
                store.add_feature("h1", 10)

                def update(target):
                    for _ in range(30):
                        target.inc_refcount("h1")
                        target.get_refcount("h1")
                        target.get_total_used()
                        target.get_eviction_candidates(10)

                # Two threads per connection, with two independent WAL writers.
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(update, [store, store, other, other]))
                assert store.get_refcount("h1") == 120
                assert not store.conn.in_transaction
                assert not other.conn.in_transaction

    def test_public_read_does_not_leave_snapshot_outside_lock(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            with SQLiteMetadataStore(store.db_path, cleanup_existing=False) as other:
                store.add_feature("h1", 10)
                read_stage = threading.Event()
                resume_read = threading.Event()
                context = threading.local()
                run_operation = store._run_with_retry

                def pause_exposed_cursor(func, *args, **kwargs):
                    result = run_operation(func, *args, **kwargs)
                    if getattr(context, "reader", False) and isinstance(result, sqlite3.Cursor):
                        # Hold the old execute/fetch gap deterministically. No SQL
                        # or transaction is injected into the public operation.
                        read_stage.set()
                        if not resume_read.wait(5):
                            result.close()
                            raise TimeoutError("Reader did not resume")
                    return result

                def read():
                    context.reader = True
                    try:
                        return store.get_refcount("h1")
                    finally:
                        context.reader = False
                        read_stage.set()

                with patch.object(store, "_run_with_retry", side_effect=pause_exposed_cursor):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        reader = pool.submit(read)
                        try:
                            assert read_stage.wait(5)
                            other.inc_refcount("h1")
                            # Must succeed after a concurrent connection commits,
                            # even while the reader task has not been joined.
                            store.inc_refcount("h1")
                        finally:
                            resume_read.set()
                        assert reader.result(timeout=5) == 0
                assert store.get_refcount("h1") == 2
                assert not store.conn.in_transaction

    def test_ttl_retry_rechecks_references_before_deleting(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            with SQLiteMetadataStore(store.db_path, cleanup_existing=False) as other:
                store.add_feature("keep", 10)
                store.add_feature("delete", 20)
                store.conn.execute("UPDATE features SET last_used = 1")
                store.conn.commit()
                connection = store.conn

                class FailFirstTTLCommit:
                    commits = 0

                    def __getattr__(self, name):
                        return getattr(connection, name)

                    def commit(self):
                        self.commits += 1
                        if self.commits == 1:
                            raise sqlite3.OperationalError("database is locked")
                        connection.commit()

                    def rollback(self):
                        connection.rollback()
                        # A request takes a reference before cleanup retries.
                        # Retrying only DELETE, or reusing old candidates, would
                        # incorrectly remove this referenced item's metadata.
                        other.inc_refcount("keep")

                store.conn = proxy = FailFirstTTLCommit()
                assert store.delete_old_features(2) == ["delete"]
                assert proxy.commits == 2
                assert store.get_refcount("keep") == 1
                assert store.get_refcount("delete") is None
                assert not connection.in_transaction

    def test_busy_writer_can_succeed_after_other_writer_commits(self, tmp_path):
        with SQLiteMetadataStore(str(tmp_path / "metadata.db")) as store:
            with SQLiteMetadataStore(store.db_path, cleanup_existing=False) as other:
                store.add_feature("h1", 10)
                store.conn.execute("PRAGMA busy_timeout=0")
                other.conn.execute("BEGIN IMMEDIATE")
                other.conn.execute("UPDATE features SET refcount = refcount + 1 WHERE mm_hash='h1'")
                releases = []

                def release_writer(_delay):
                    releases.append(True)
                    # A failed BEGIN on store must not affect the transaction
                    # owned by the independent connection.
                    assert other.conn.in_transaction
                    other.conn.commit()

                with patch.object(time, "sleep", side_effect=release_writer):
                    store.inc_refcount("h1")
                assert releases == [True]
                assert store.get_refcount("h1") == 2
                assert not store.conn.in_transaction

    def test_sql_error_rolls_back_without_retrying_locked_table_name(self, store):
        store.add_feature("h1", 10)
        attempts = []

        def invalid_operation():
            attempts.append(True)
            store._execute("UPDATE features SET refcount = refcount + 1")
            # The message contains "locked", but the real code is SQLITE_ERROR,
            # not SQLITE_BUSY/LOCKED. It must not be retried based on text alone.
            store._execute("SELECT * FROM locked")

        with pytest.raises(sqlite3.OperationalError) as error:
            store._run_with_retry(invalid_operation, write=True)
        assert error.value.sqlite_errorcode == sqlite3.SQLITE_ERROR
        assert attempts == [True]
        assert store.get_refcount("h1") == 0
        assert not store.conn.in_transaction
