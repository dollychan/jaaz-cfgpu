from . import Migration
import sqlite3


class V4AddAssetFiles(Migration):
    version = 4
    description = "Add asset_files table for material library"

    def up(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_files (
                fid         INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                asset_id    TEXT NOT NULL UNIQUE,
                group_id    TEXT DEFAULT '',
                status      TEXT NOT NULL DEFAULT 'Processing',
                asset_type  TEXT NOT NULL DEFAULT 'Image',
                project_name TEXT NOT NULL DEFAULT 'default',
                url         TEXT DEFAULT '',
                created_at  TEXT DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%SZ', 'now')),
                updated_at  TEXT DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_files_asset_id
            ON asset_files(asset_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_files_status
            ON asset_files(status)
        """)

    def down(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE IF EXISTS asset_files")
