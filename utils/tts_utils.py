"""Shared non-blocking text-to-speech worker.

pyttsx3's runAndWait() blocks the calling thread until speech finishes.
Running it on a background thread keeps a webcam/inference loop responsive
regardless of which recognition script (words or letters) is speaking.
"""
import queue
import threading


class SpeechWorker:
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import pyttsx3
        engine = pyttsx3.init()
        while True:
            text = self._queue.get()
            if text is None:
                break
            engine.stop()
            engine.say(text)
            engine.runAndWait()

    def say(self, text: str):
        self._queue.put(text)

    def stop(self):
        self._queue.put(None)
