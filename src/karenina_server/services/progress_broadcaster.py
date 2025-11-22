"""Shared progress broadcaster for WebSocket updates across services."""

import asyncio
import contextlib
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ProgressBroadcaster:
    """Manages WebSocket connections for broadcasting progress updates."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for thread-safe broadcasting."""
        self._event_loop = loop

    async def subscribe(self, job_id: str, websocket: WebSocket) -> None:
        """Add a WebSocket subscriber for a job."""
        async with self._lock:
            self.subscribers[job_id].append(websocket)

    async def unsubscribe(self, job_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket subscriber."""
        async with self._lock:
            if job_id in self.subscribers:
                with contextlib.suppress(ValueError):
                    self.subscribers[job_id].remove(websocket)
                # Clean up empty lists
                if not self.subscribers[job_id]:
                    del self.subscribers[job_id]

    async def broadcast(self, job_id: str, event_data: dict[str, Any]) -> None:
        """Broadcast a progress event to all subscribers of a job."""
        async with self._lock:
            if job_id not in self.subscribers:
                return

            # Send to all subscribers, removing dead connections
            dead_sockets = []
            for websocket in self.subscribers[job_id]:
                try:
                    await websocket.send_json(event_data)
                except Exception:
                    dead_sockets.append(websocket)

            # Clean up dead connections
            for dead_socket in dead_sockets:
                with contextlib.suppress(ValueError):
                    self.subscribers[job_id].remove(dead_socket)

            # Clean up empty lists
            if not self.subscribers[job_id]:
                del self.subscribers[job_id]

    def broadcast_from_thread(self, job_id: str, event_data: dict[str, Any]) -> None:
        """Thread-safe method to broadcast from worker threads."""
        if self._event_loop and not self._event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast(job_id, event_data), self._event_loop)

    async def cleanup_job(self, job_id: str) -> None:
        """Close all connections for a job and clean up."""
        async with self._lock:
            if job_id in self.subscribers:
                for websocket in self.subscribers[job_id]:
                    with contextlib.suppress(Exception):
                        await websocket.close()
                del self.subscribers[job_id]
