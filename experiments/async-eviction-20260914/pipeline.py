"""Bounded source-preservation pipeline; persistence remains the public ACK."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue


class StagedEvictions:
    def __init__(self, slots, persistence_executor):
        self.free = Queue(maxsize=slots)
        for i in range(slots):
            self.free.put(i)
        self.persistence_executor = persistence_executor
        self.stager = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kv-stage')

    def submit(self, stage, persist):
        preserved, completed = Future(), Future()

        def copy():
            slot = self.free.get()  # bounded backpressure, including queued disk work
            handed_off = False
            try:
                payload = stage(slot)  # must synchronously finish GPU source reads

                def write():
                    try:
                        completed.set_result(persist(slot, payload))
                    except BaseException as exc:
                        completed.set_exception(exc)
                    finally:
                        self.free.put(slot)

                self.persistence_executor.submit(write)
                handed_off = True
                preserved.set_result(None)
            except BaseException as exc:
                preserved.set_exception(exc)
                completed.set_exception(exc)
            finally:
                if not handed_off:
                    self.free.put(slot)

        try:
            self.stager.submit(copy)
        except BaseException as exc:
            preserved.set_exception(exc)
            completed.set_exception(exc)
        return preserved, completed

    def shutdown(self):
        self.stager.shutdown(wait=True)
