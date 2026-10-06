# Python 3.10.0 only. Its asyncio.Condition(lock) rejects a lock that is not yet bound to the running
# loop, which makes Ray's dashboard agent crash at start-up ("ValueError: loop argument must agree with
# lock") and takes the raylet down with it. Later 3.10.x releases dropped that check; this restores
# their behaviour. It is a no-op on every other Python version.
#
# Put this directory on PYTHONPATH (run_webshop_seed_repro.sh does so when the env runs 3.10.0).
import sys

if sys.version_info[:3] == (3, 10, 0):
    import asyncio.locks as _locks
    import collections as _collections

    def _condition_init(self, lock=None, *, loop=_locks.mixins._marker):
        _locks.mixins._LoopBoundMixin.__init__(self, loop=loop)
        if lock is None:
            lock = _locks.Lock()
        self._lock = lock
        self.locked = lock.locked
        self.acquire = lock.acquire
        self.release = lock.release
        self._waiters = _collections.deque()

    _locks.Condition.__init__ = _condition_init
