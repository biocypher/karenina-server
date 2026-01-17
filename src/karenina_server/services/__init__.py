"""Async job services for long-running operations.

Lock Hierarchy (conc-005)
=========================

This module documents the lock ordering to prevent deadlocks across services.
Always acquire locks in the order listed below (lower numbers first).

Global Lock Order:
    1. CsrfTokenStore._lock (class-level singleton) - auth_handlers.py
    2. CsrfTokenStore._store_lock (instance operations) - auth_handlers.py
    3. VerificationService._shutdown_lock - verification_service.py
    4. VerificationService._master_lock - verification_service.py
    5. VerificationService._job_locks[job_id] - verification_service.py
    6. GenerationService._shutdown_lock - generation_service.py
    7. ProgressBroadcaster._lock (asyncio.Lock) - progress_broadcaster.py

Service-Specific Lock Details:
------------------------------

CsrfTokenStore (auth_handlers.py):
    - _lock: Class-level Lock for singleton pattern (double-checked locking)
    - _store_lock: Instance Lock for token CRUD operations
    - No external dependencies; safe to acquire from any context

VerificationService (verification_service.py):
    - _shutdown_lock: Protects shutdown state and coordination
    - _master_lock: Protects _job_locks dict modification
    - _job_locks[job_id]: Per-job locks for thread-safe status access
    - Always acquire: shutdown_lock -> master_lock -> job_lock

GenerationService (generation_service.py):
    - _shutdown_lock: Protects shutdown state and coordination
    - No per-job locks (simpler job model)

ProgressBroadcaster (progress_broadcaster.py):
    - _lock: asyncio.Lock for subscriber dict modifications
    - Called from worker threads via broadcast_from_thread()
    - Uses asyncio.run_coroutine_threadsafe for thread-safe event loop access

Cross-Service Considerations:
-----------------------------
- Services do not call each other while holding locks
- Each service's broadcaster is independent
- CsrfTokenStore is isolated from job services
- Deadlock risk is minimal due to service isolation

Lock Acquisition Timeouts:
--------------------------
Standard threading.Lock does not support timeouts, but the lock hierarchy
ensures deadlock-free operation. If a service needs timeout-based acquisition
for defensive programming, it should:
1. Log a warning if acquisition takes longer than expected
2. Use threading.RLock with timeout in Python 3.12+
"""

from .generation_service import GenerationService
from .verification_service import VerificationService

__all__ = ["GenerationService", "VerificationService"]
