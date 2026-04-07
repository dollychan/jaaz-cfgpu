import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
import aiosqlite
from .config_service import USER_DATA_DIR
from .migrations.manager import MigrationManager, CURRENT_VERSION

DB_PATH = os.path.join(USER_DATA_DIR, "localmanus.db")

class DatabaseService:
    def __init__(self):
        self.db_path = DB_PATH
        self._ensure_db_directory()
        self._migration_manager = MigrationManager()
        self._init_db()

    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _init_db(self):
        """Initialize the database with the current schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Create version table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS db_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # Get current version
            cursor = conn.execute("SELECT version FROM db_version")
            current_version = cursor.fetchone()
            print('local db version', current_version, 'latest version', CURRENT_VERSION)
            
            if current_version is None:
                # First time setup - start from version 0
                conn.execute("INSERT INTO db_version (version) VALUES (0)")
                self._migration_manager.migrate(conn, 0, CURRENT_VERSION)
            elif current_version[0] < CURRENT_VERSION:
                print('Migrating database from version', current_version[0], 'to', CURRENT_VERSION)
                # Need to migrate
                self._migration_manager.migrate(conn, current_version[0], CURRENT_VERSION)

    async def create_canvas(self, id: str, name: str):
        """Create a new canvas"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO canvases (id, name)
                VALUES (?, ?)
            """, (id, name))
            await db.commit()

    async def list_canvases(self) -> List[Dict[str, Any]]:
        """Get all canvases"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute("""
                SELECT id, name, description, thumbnail, created_at, updated_at
                FROM canvases
                ORDER BY updated_at DESC
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def create_chat_session(self, id: str, model: str, provider: str, canvas_id: str, title: Optional[str] = None):
        """Save a new chat session"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO chat_sessions (id, model, provider, canvas_id, title)
                VALUES (?, ?, ?, ?, ?)
            """, (id, model, provider, canvas_id, title))
            await db.commit()

    async def create_message(self, session_id: str, role: str, message: str):
        """Save a chat message"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO chat_messages (session_id, role, message)
                VALUES (?, ?, ?)
            """, (session_id, role, message))
            await db.commit()

    async def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute("""
                SELECT role, message, id
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))
            rows = await cursor.fetchall()
            
            messages = []
            for row in rows:
                row_dict = dict(row)
                if row_dict['message']:
                    try:
                        msg = json.loads(row_dict['message'])
                        messages.append(msg)
                    except:
                        pass
                
            return messages

    async def list_sessions(self, canvas_id: str) -> List[Dict[str, Any]]:
        """List all chat sessions"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            if canvas_id:
                cursor = await db.execute("""
                    SELECT id, title, model, provider, created_at, updated_at
                    FROM chat_sessions
                    WHERE canvas_id = ?
                    ORDER BY updated_at DESC
                """, (canvas_id,))
            else:
                cursor = await db.execute("""
                    SELECT id, title, model, provider, created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def save_canvas_data(self, id: str, data: str, thumbnail: str = None):
        """Save canvas data"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE canvases 
                SET data = ?, thumbnail = ?, updated_at = STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
            """, (data, thumbnail, id))
            await db.commit()

    async def get_canvas_data(self, id: str) -> Optional[Dict[str, Any]]:
        """Get canvas data"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute("""
                SELECT data, name
                FROM canvases
                WHERE id = ?
            """, (id,))
            row = await cursor.fetchone()

            sessions = await self.list_sessions(id)
            
            if row:
                return {
                    'data': json.loads(row['data']) if row['data'] else {},
                    'name': row['name'],
                    'sessions': sessions
                }
            return None

    async def delete_session(self, session_id: str):
        """Delete a chat session and all its messages"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            await db.commit()

    async def delete_canvas(self, id: str):
        """Delete canvas and all related sessions and messages"""
        async with aiosqlite.connect(self.db_path) as db:
            # 先收集该 canvas 下所有 session id
            cursor = await db.execute(
                "SELECT id FROM chat_sessions WHERE canvas_id = ?", (id,)
            )
            rows = await cursor.fetchall()
            session_ids = [row[0] for row in rows]

            # 删除这些 session 的全部消息
            if session_ids:
                placeholders = ','.join('?' * len(session_ids))
                await db.execute(
                    f"DELETE FROM chat_messages WHERE session_id IN ({placeholders})",
                    session_ids,
                )

            # 删除 sessions
            await db.execute("DELETE FROM chat_sessions WHERE canvas_id = ?", (id,))

            # 删除 canvas 本体
            await db.execute("DELETE FROM canvases WHERE id = ?", (id,))
            await db.commit()

    async def rename_canvas(self, id: str, name: str):
        """Rename canvas"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE canvases SET name = ? WHERE id = ?", (name, id))
            await db.commit()

    async def create_comfy_workflow(self, name: str, api_json: str, description: str, inputs: str, outputs: str = None):
        """Create a new comfy workflow"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO comfy_workflows (name, api_json, description, inputs, outputs)
                VALUES (?, ?, ?, ?, ?)
            """, (name, api_json, description, inputs, outputs))
            await db.commit()

    async def list_comfy_workflows(self) -> List[Dict[str, Any]]:
        """List all comfy workflows"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute("SELECT id, name, description, api_json, inputs, outputs FROM comfy_workflows ORDER BY id DESC")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_comfy_workflow(self, id: int):
        """Delete a comfy workflow"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM comfy_workflows WHERE id = ?", (id,))
            await db.commit()

    async def get_comfy_workflow(self, id: int):
        """Get comfy workflow dict"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                "SELECT api_json FROM comfy_workflows WHERE id = ?", (id,)
            )
            row = await cursor.fetchone()
        try:
            workflow_json = (
                row["api_json"]
                if isinstance(row["api_json"], dict)
                else json.loads(row["api_json"])
            )
            return workflow_json
        except json.JSONDecodeError as exc:
            raise ValueError(f"Stored workflow api_json is not valid JSON: {exc}")

    # ── Asset Files (Material Library) ──────────────────────────────────────

    async def insert_asset_file(
        self,
        name: str,
        asset_id: str,
        asset_type: str = "Image",
        group_id: str = "",
        project_name: str = "default",
        status: str = "Processing",
    ) -> int:
        """Insert a new asset file record. Returns the auto-generated fid."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO asset_files (name, asset_id, asset_type, group_id, project_name, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, asset_id, asset_type, group_id, project_name, status),
            )
            await db.commit()
            return cursor.lastrowid

    async def update_asset_file(
        self,
        asset_id: str,
        status: Optional[str] = None,
        url: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Partially update an asset file record by asset_id."""
        fields: list[str] = ["updated_at = STRFTIME('%Y-%m-%dT%H:%M:%SZ', 'now')"]
        params: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if url is not None:
            fields.append("url = ?")
            params.append(url)
        if group_id is not None:
            fields.append("group_id = ?")
            params.append(group_id)
        params.append(asset_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE asset_files SET {', '.join(fields)} WHERE asset_id = ?",
                params,
            )
            await db.commit()

    async def rename_asset_file(self, asset_id: str, new_name: str) -> None:
        """Rename the display name of an asset file."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE asset_files SET name = ?, updated_at = STRFTIME('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE asset_id = ?",
                (new_name, asset_id),
            )
            await db.commit()

    async def get_asset_files_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Return all asset file records with the given status."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                "SELECT * FROM asset_files WHERE status = ? ORDER BY fid DESC",
                (status,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_asset_files(self) -> List[Dict[str, Any]]:
        """Return all asset file records ordered by creation time (newest first)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                "SELECT * FROM asset_files ORDER BY fid DESC"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_asset_file_by_asset_id(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Return a single asset file record by asset_id, or None if not found."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = await db.execute(
                "SELECT * FROM asset_files WHERE asset_id = ?",
                (asset_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete_asset_file(self, asset_id: str) -> None:
        """Delete an asset file record."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM asset_files WHERE asset_id = ?",
                (asset_id,),
            )
            await db.commit()

# Create a singleton instance
db_service = DatabaseService()
