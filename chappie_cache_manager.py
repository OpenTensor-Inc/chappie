"""
Chappie Cache Manager
======================
Handles cache cleanup to prevent disk space exhaustion.

Strategy:
    - Conservative cleanup: never touch files modified in the last N minutes
      (avoids deleting files still in use by interleave_datasets or streaming).
    - Periodic removal of stale temp files (every N steps).
    - Size-based eviction: oldest unused files first when cache exceeds limit.
    - Full cleanup at end of training.

Design notes:
    - We use HF streaming + disable_caching(), so almost no Arrow files
      are produced. The main cache is tokenizer files and dataset metadata.
    - Deleting a .lock file mid-download can corrupt the cache, so we skip
      any file modified recently.
"""

import os
import time
import shutil
import glob
import logging

logger = logging.getLogger(__name__)

# Files modified within this many seconds are never deleted
# (protects active downloads and in-use streaming metadata).
RECENT_FILE_GRACE_SECONDS = 300  # 5 minutes


def _iter_cache_files(cache_dirs):
    """Yield (path, size, mtime) for every file under the given dirs."""
    for d in cache_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                yield fp, st.st_size, st.st_mtime


def _is_safe_to_delete(fp: str, mtime: float) -> bool:
    """Never delete lock files or recently-modified files."""
    if fp.endswith(".lock"):
        return False
    if time.time() - mtime < RECENT_FILE_GRACE_SECONDS:
        return False
    return True


def cleanup_cache(step: int, every_n: int = 500, max_size_gb: float = 50.0):
    """
    Remove old cache files to free disk space.

    Args:
        step: Current training step.
        every_n: Cleanup interval in steps.
        max_size_gb: Maximum allowed cache size in GB.
    """
    cache_dirs = [
        os.environ.get("HF_DATASETS_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface/datasets"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]

    # 1. Periodic cleanup: remove only *stale* temp files
    if step > 0 and step % every_n == 0:
        now = time.time()
        for d in cache_dirs:
            if not d or not os.path.isdir(d):
                continue
            for pattern in ["*.tmp", "*.partial", "*.incomplete"]:
                for fp in glob.glob(os.path.join(d, "**", pattern), recursive=True):
                    try:
                        mtime = os.stat(fp).st_mtime
                        if now - mtime > RECENT_FILE_GRACE_SECONDS:
                            os.remove(fp)
                    except OSError:
                        pass

    # 2. Size-based eviction
    total_size = 0
    candidates = []  # (mtime, path, size)
    for fp, size, mtime in _iter_cache_files(cache_dirs):
        total_size += size
        if _is_safe_to_delete(fp, mtime):
            candidates.append((mtime, fp, size))

    total_gb = total_size / (1024 ** 3)
    if total_gb <= max_size_gb:
        return

    logger.info(
        f"[cache] Size {total_gb:.2f}GB > {max_size_gb}GB, "
        f"evicting oldest files..."
    )
    candidates.sort(key=lambda x: x[0])  # oldest first

    target_bytes = int(max_size_gb * 0.8 * (1024 ** 3))
    for _, fp, size in candidates:
        try:
            os.remove(fp)
            total_size -= size
        except OSError:
            continue
        if total_size < target_bytes:
            break

    logger.info(f"[cache] New size: {total_size / (1024 ** 3):.2f}GB")


def full_cleanup(cache_root: str = None):
    """Remove the entire cache directory. Call at end of training."""
    if cache_root is None:
        cache_root = os.path.expanduser("~/tmp_chappie_cache")
    if os.path.isdir(cache_root):
        try:
            shutil.rmtree(cache_root)
            logger.info(f"[cache] Removed: {cache_root}")
        except OSError as e:
            logger.warning(f"[cache] Could not remove {cache_root}: {e}")


def report_cache_size():
    """Log the current cache size (useful for debugging)."""
    cache_dirs = [
        os.environ.get("HF_DATASETS_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface/datasets"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    total = sum(size for _, size, _ in _iter_cache_files(cache_dirs))
    logger.info(f"[cache] Current size: {total / (1024 ** 3):.3f} GB")