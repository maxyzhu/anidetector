import os
from celery import Celery
from celery.signals import worker_init, worker_process_init


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("anidetector")
# Read config from Django settings, only keys prefixed CELERY_.
app.config_from_object("django.conf:settings", namespace="CELERY")
# Auto-find tasks.py in every installed app.
app.autodiscover_tasks()


# Probed in the parent, before any fork: asking torch about the GPU after a
# fork is the very thing this guard exists to prevent.
_gpu_present = False
_parent_pid = None


@worker_init.connect
def _probe_gpu_before_fork(**_):
    global _gpu_present, _parent_pid
    _parent_pid = os.getpid()
    import torch

    mps = getattr(torch.backends, "mps", None)
    _gpu_present = torch.cuda.is_available() or (
        mps is not None and mps.is_available()
    )


@worker_process_init.connect
def _refuse_forked_gpu(**_):
    """Neither CUDA nor MPS survives a fork; the crash is a bare SIGABRT the
    first time a task touches the GPU, with nothing in the log to explain it.

    The test is the pid, not the pool name: solo and threads run in the process
    that emitted worker_init, prefork runs in a child of it.
    """
    if os.getpid() == _parent_pid or not _gpu_present:
        return
    raise RuntimeError(
        "A GPU is present and this worker was forked. Neither CUDA nor MPS is "
        "fork-safe. Start the worker with --pool=solo or -P threads."
    )