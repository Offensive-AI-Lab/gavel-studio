# utils/env_file.py
# Atomic single-var read/write against backend/.env, shared by every module
# that stores a credential or setting there (utils/openai_key.py,
# utils/local_llm.py). Extracted so both write the same way instead of two
# copies of the same crash-safety logic drifting apart.
import os
import stat as stat_module
import uuid
from pathlib import Path


def read_lines(path: Path, fallback_path: Path = None) -> list:
    """Existing lines of the .env, seeded from `fallback_path` (typically
    .env.example) on first write."""
    for candidate in (path, fallback_path):
        if candidate is None:
            continue
        try:
            return candidate.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
    return []


def write_var(path: Path, key: str, value: str, fallback_path: Path = None) -> None:
    """Replace (or append) `key=value`, leaving every other line untouched.

    Same file convention as run.sh's `set_env` (run.sh:126): first line whose
    stripped form starts with `KEY=` wins, commented-out lines are not touched,
    LF endings, trailing newline.

    Writes atomically (temp file + os.replace) so a crash mid-write can never
    truncate the operator's config. A file we create is owner-only — it may
    hold a paid credential; a file that already exists keeps the permissions
    and owner it came with.
    """
    lines = read_lines(path, fallback_path)

    out, found = [], False
    for line in lines:
        if not found and line.lstrip().startswith(key + "="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")

    text = "\n".join(out) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)

    # os.replace swaps in a brand-new inode, so the replacement starts with the
    # mode and ownership of whatever we just created — not the file the operator
    # had. That matters: the documented Docker run bind-mounts ./backend into a
    # container with no USER, so a fresh root-owned 0600 file would leave the
    # host user unable to read their own backend/.env. Carry the original
    # identity across the swap; only a file we create from nothing gets 0600.
    try:
        original = path.stat()
    except OSError:
        original = None

    # Unique per call, not per process: FastAPI runs sync routes in a
    # threadpool, so two saves can be in this function at the same time.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if original is not None:
            _restore_identity(tmp, original)
        os.replace(str(tmp), str(path))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _restore_identity(tmp: Path, original: os.stat_result) -> None:
    """Give the replacement file the mode and owner of the file it replaces.

    Applied to the temp file before os.replace, so the .env is never briefly
    readable by someone it wasn't readable by before. Both calls are best
    effort: an unprivileged process can't chown to another user, and Windows
    has neither call in a meaningful form.
    """
    try:
        os.chmod(str(tmp), stat_module.S_IMODE(original.st_mode))
    except OSError:
        pass
    try:
        os.chown(str(tmp), original.st_uid, original.st_gid)
    except (OSError, AttributeError, NotImplementedError):
        pass
