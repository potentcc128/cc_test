from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from tts_client import TTSClient, TTSError


@dataclass
class TTSTask:
    index: int
    text: str
    voice_type: str
    context_texts: list[str] | None = None
    section_id: str | None = None


@dataclass
class TTSResult:
    index: int
    text: str
    audio: bytes | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.audio is not None


class BatchProcessor:
    def __init__(self, client: TTSClient, max_workers: int = 5):
        self.client = client
        self.max_workers = max_workers

    def process(
        self,
        tasks: list[TTSTask],
        on_result: Callable[[TTSResult], None] | None = None,
    ) -> list[TTSResult]:
        """
        并发执行所有任务。每完成一条就调用 on_result 回调（就绪即回调）。
        返回按 index 排序的结果列表。
        """
        results: dict[int, TTSResult] = {}
        lock = threading.Lock()

        def run_task(task: TTSTask) -> TTSResult:
            try:
                audio = self.client.synthesize(
                    text=task.text,
                    voice_type=task.voice_type,
                    context_texts=task.context_texts,
                    section_id=task.section_id,
                )
                return TTSResult(index=task.index, text=task.text, audio=audio)
            except TTSError as e:
                return TTSResult(index=task.index, text=task.text, error=str(e))
            except Exception as e:
                return TTSResult(index=task.index, text=task.text, error=f"未知错误: {e}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                with lock:
                    results[result.index] = result
                if on_result:
                    on_result(result)

        return [results[i] for i in sorted(results)]
