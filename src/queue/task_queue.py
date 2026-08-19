import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional, Union

logger = logging.getLogger("medical_rag.queue.task_queue")


class TaskQueue:
    """Asyncio-based background task queue for processing document ingestion tasks sequentially."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        """Returns True if the worker loop is active."""
        return self._running

    @property
    def qsize(self) -> int:
        """Returns the current number of queued tasks."""
        return self._queue.qsize()

    def _ensure_worker_started(self) -> None:
        """Helper to safely start background worker task if an asyncio event loop is running."""
        if not self._running or self._worker_task is None or self._worker_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._running = True
                self._worker_task = loop.create_task(self._worker_loop())
                logger.info("Started TaskQueue background worker loop")
            except RuntimeError:
                # Event loop not running in current thread yet; startup will defer until enqueue/async context
                pass

    def start(self) -> None:
        """Starts the background worker loop task."""
        self._ensure_worker_started()

    async def stop(self) -> None:
        """Stops the background worker loop task gracefully."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("Stopped TaskQueue background worker loop")

    async def enqueue(self, task: Union[Coroutine, Callable[[], Any]]) -> None:
        """Enqueues a coroutine or callable task for background execution."""
        self._ensure_worker_started()
        await self._queue.put(task)
        logger.info(f"Enqueued task into TaskQueue (queue size: {self._queue.qsize()})")

    def enqueue_nowait(self, task: Union[Coroutine, Callable[[], Any]]) -> None:
        """Enqueues a task synchronously into the queue without awaiting."""
        self._ensure_worker_started()
        self._queue.put_nowait(task)
        logger.info(f"Enqueued task into TaskQueue via put_nowait (queue size: {self._queue.qsize()})")

    async def join(self) -> None:
        """Blocks until all items in the queue have been processed."""
        await self._queue.join()

    async def _worker_loop(self) -> None:
        """Worker loop that pops tasks from queue and executes them sequentially."""
        while self._running:
            try:
                task = await self._queue.get()
                try:
                    logger.info("Processing task from TaskQueue")
                    if asyncio.iscoroutine(task):
                        await task
                    elif callable(task):
                        res = task()
                        if asyncio.iscoroutine(res):
                            await res
                    else:
                        logger.error(f"Unsupported task object in TaskQueue: {type(task)}")
                except Exception as err:
                    logger.error(f"Error processing background task: {err}", exc_info=True)
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as loop_err:
                logger.error(f"Unexpected error in TaskQueue worker loop: {loop_err}", exc_info=True)
