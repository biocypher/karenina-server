"""Shared progress broadcaster for WebSocket updates across services."""

import asyncio
import contextlib
import logging
from collections import defaultdict
from concurrent.futures import Future
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ProgressBroadcaster:
    """Manages WebSocket connections for broadcasting progress updates.

    Lock Hierarchy (conc-005):
        1. _lock (asyncio.Lock) - Protects subscribers dict modifications

    This class uses asyncio.Lock (not threading.Lock) because all subscriber
    management occurs in async context. The broadcast_from_thread() method
    safely schedules broadcasts onto the event loop without holding locks
    across thread boundaries.
    """

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
        """Thread-safe method to broadcast from worker threads.

        Schedules a broadcast coroutine on the main event loop from a worker thread.
        Logs warnings if broadcasting fails (event loop unavailable or closed) to
        ensure progress update issues are visible for debugging.
        """
        if not self._event_loop:
            logger.warning(
                "Cannot broadcast progress for job %s: no event loop available. Event type: %s",
                job_id,
                event_data.get("event", "unknown"),
            )
            return

        if self._event_loop.is_closed():
            logger.warning(
                "Cannot broadcast progress for job %s: event loop is closed. Event type: %s",
                job_id,
                event_data.get("event", "unknown"),
            )
            return

        try:
            future = asyncio.run_coroutine_threadsafe(self.broadcast(job_id, event_data), self._event_loop)
            # Add callback to catch any exceptions during broadcast
            future.add_done_callback(lambda f: self._handle_broadcast_result(f, job_id, event_data))
        except Exception:
            logger.exception(
                "Failed to schedule broadcast for job %s. Event type: %s",
                job_id,
                event_data.get("event", "unknown"),
            )

    def _handle_broadcast_result(
        self,
        future: Future[None],
        job_id: str,
        event_data: dict[str, Any],
    ) -> None:
        """Handle the result of a broadcast future, logging any exceptions.

        This callback is invoked when the broadcast coroutine completes (or fails).
        It ensures that exceptions raised during broadcast are logged rather than
        silently swallowed.
        """
        try:
            # Retrieve result to surface any exceptions (no timeout - callback fires after completion)
            future.result()
        except asyncio.CancelledError:
            logger.warning(
                "Broadcast was cancelled for job %s. Event type: %s",
                job_id,
                event_data.get("event", "unknown"),
            )
        except Exception:
            logger.exception(
                "Broadcast failed for job %s. Event type: %s",
                job_id,
                event_data.get("event", "unknown"),
            )

    async def cleanup_job(self, job_id: str) -> None:
        """Close all connections for a job and clean up."""
        async with self._lock:
            if job_id in self.subscribers:
                for websocket in self.subscribers[job_id]:
                    with contextlib.suppress(Exception):
                        await websocket.close()
                del self.subscribers[job_id]
