Feature: User Multi-Chat (Backend Implementation Steps)
=====================================================

Purpose
- Provide a concise, actionable implementation checklist for the backend work required to support per-user multi-chat persistence, chat lifecycle operations (create/delete/pin/rename), and message storage/retrieval.

Assumptions
- Using Supabase (Postgres + REST) for persistence; dev fallback exists in `SupabaseService` for local runs.
- Authentication is Supabase-based and available via existing `auth_router` dependency `get_current_user`.

High-level Plan
- Database schema (SQL migration)
- Server-side persistence helpers (`SupabaseService`)
- HTTP API endpoints (FastAPI router)
- Integration tests and manual smoke checks
- Frontend contract description (for frontend implementers)

Detailed Implementation Steps (Backend)
- 1) Database migration
	- Create SQL migration to add `chats` and `chat_messages` tables.
	- Columns:
		- `chats`: `id uuid PK`, `user_id uuid`, `title text`, `pinned boolean default false`, `last_message_at timestamptz`, `created_at timestamptz default now()`, `deleted boolean default false`.
		- `chat_messages`: `id uuid PK`, `chat_id uuid FK -> chats(id)`, `sender text`, `content text`, `metadata jsonb`, `created_at timestamptz default now()`.
	- Add indexes on `chats(user_id)`, `chats(last_message_at)` and `chat_messages(chat_id)`.
	- File: `supabase_migrations/001_create_chats_messages.sql` (example provided).

- 2) Supabase persistence helpers
	- Add methods to `src/db/supabase_service.py`:
		- `create_chat(user_id, title) -> chat` (POST `/rest/v1/chats`)
		- `list_user_chats(user_id, limit) -> list[chat]` (GET `/rest/v1/chats`)
		- `add_message(chat_id, sender, content, metadata) -> message` (POST `/rest/v1/chat_messages`)
		- `get_chat_messages(chat_id, limit, offset) -> list[message]` (GET `/rest/v1/chat_messages`)
		- `soft_delete_chat(chat_id) -> bool` (PATCH `/rest/v1/chats?id=eq.{chat_id}` set `deleted=true`)
		- `pin_chat(chat_id, pinned) -> bool` (PATCH `/rest/v1/chats`)
		- `rename_chat(chat_id, title) -> bool` (PATCH `/rest/v1/chats`)
	- Ensure each method supports the existing dev-fallback in-memory behavior when Supabase is unconfigured.

- 3) FastAPI router & endpoints
	- Add `src/api/chat_router.py` and register it in `main.py`.
	- Endpoints & Contracts (prefix `/chats`)
		- GET `/chats/` — list user's non-deleted chats (auth required). Response: list of chats ordered by `pinned desc, last_message_at desc, created_at desc`.
		- POST `/chats/` — create chat with optional `title`. Returns created chat.
		- GET `/chats/{chat_id}/messages` — fetch messages for chat (pagination via `limit`/`offset`).
		- POST `/chats/{chat_id}/messages` — add message; request body: `sender`, `content`, `metadata`.
		- DELETE `/chats/{chat_id}` — soft-delete a chat (set `deleted=true`).
		- PATCH `/chats/{chat_id}/pin?pinned=true|false` — pin/unpin a chat.
		- PATCH `/chats/{chat_id}/rename` — rename chat via `title` param/body.
	- Use existing `get_current_user` dependency to enforce per-user access.

- 4) App wiring
	- Include the router in `backend/main.py` so it initializes under the app lifespan.
	- Ensure CORS allowed origins include frontend dev host(s).

- 5) Testing & validation
	- Manual smoke commands (use `curl` or `httpie` with Bearer token):
		- Create chat: `POST /chats` with `{ "title": "Notes" }`.
		- List chats: `GET /chats/`.
		- Add message: `POST /chats/{chat_id}/messages` with `{ "sender": "user", "content": "Hello" }`.
		- Fetch messages: `GET /chats/{chat_id}/messages`.
		- Pin chat: `PATCH /chats/{chat_id}/pin?pinned=true`.
		- Delete chat: `DELETE /chats/{chat_id}`.
	- Add lightweight unit/integration tests to exercise SupabaseService dev-fallback and REST flows. Keep tests in `tests/` referencing a test config that disables Supabase to use in-memory fallback.

- 6) Observability
	- Add logging for key operations and error paths in `SupabaseService` and `chat_router`.
	- Consider tracing critical flows for later performance tuning (e.g., add metrics around message insert rate).

Frontend Contract (for frontend implementers)
- API base: `POST /chats` — create chat => returns chat object including `id`.
- `GET /chats` — returns array of chats: each chat includes `id`, `user_id`, `title`, `pinned`, `last_message_at`, `created_at`.
- `GET /chats/{chat_id}/messages` — returns array of messages ordered ascending by `created_at`.
- `POST /chats/{chat_id}/messages` — body: `{ sender, content, metadata? }` returns created message with `id` and `created_at`.
- Auth: Bearer token required (same as `/auth/me`).

Operational Notes
- Migration: Apply `supabase_migrations/001_create_chats_messages.sql` through Supabase SQL Editor or your migration pipeline.
- Supabase Limits: For high-volume message writes, consider batching or applying write-rate controls; consider moving heavy chat logs to a separate table partitioned by time if necessary.
- Data retention: `soft_delete_chat` marks `deleted=true` — consider a scheduled job to purge old deleted chats.

Next Steps
- Implement frontend pieces (API client `lib/api/chats.ts`, `ChatList` component, adapt `ChatThread` to accept `chatId`) and wire optimistic local cache.
- Add integration tests and CI step to run backend tests and smoke the chat endpoints.

