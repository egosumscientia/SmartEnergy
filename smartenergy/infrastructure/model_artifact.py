"""Publicacion atomica y bloqueo compartido del bundle joblib en Linux/Docker.

El archivo .lock es estable y no se reemplaza ni elimina. Los monitores con montaje
de solo lectura requieren que ese archivo se aprovisione junto al artefacto.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
import time

import joblib
from filelock import FileLock, Timeout


@contextmanager
def model_read_lock(model_path, timeout=60):
    lock_path = str(model_path) + ".lock"
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise Timeout(lock_path)
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def load_model_bundle(model_path, timeout=60):
    with model_read_lock(model_path, timeout):
        return joblib.load(model_path)


def save_model_bundle(bundle, model_path, timeout=60):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=timeout):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                joblib.dump(bundle, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
