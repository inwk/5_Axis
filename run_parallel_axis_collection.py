"""Parallel runner for large-scale axis dataset collection on shared storage."""

import argparse
import ctypes
import ctypes.wintypes as wintypes
import glob
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time

# Use the current interpreter so jobs run in the active environment.
PYTHON_EXE = sys.executable
WORKER_SCRIPT = "collect_axis_dataset.py"
PC_NAME = "615"
WORKER_JOB_CLEANUP = os.getenv("WORKER_JOB_CLEANUP", "1") != "0"

# Shared network paths for distributed collection machines.
SHARED_BASE_DIR = r"Y:\04_개별폴더\22. 통합과정 오인욱"
SHARED_INPUT_DIR = os.path.join(SHARED_BASE_DIR, "prt_dataset")
SHARED_OUTPUT_DIR = os.path.join(SHARED_BASE_DIR, "sdf_dataset_out")
CURRENT_SEED = 0


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def _is_completed_run_dir(run_dir: str) -> bool:
    """Returns True only for fully completed episode output directories."""
    if not os.path.isdir(run_dir):
        return False
    episode_record_path = os.path.join(run_dir, "episode_record.json")
    if not os.path.isfile(episode_record_path):
        return False
    local_parquets = glob.glob(os.path.join(run_dir, "*_process_skeleton_dataset.parquet"))
    if not local_parquets:
        return False
    try:
        with open(episode_record_path, "r", encoding="utf-8") as f:
            episode_record = json.load(f)
    except Exception:
        return False
    termination = episode_record.get("termination")
    if not isinstance(termination, dict):
        return False
    return bool(termination.get("reason"))


def _global_parquet_run_name(filename: str) -> str:
    """Extracts the run directory stem from a global parquet filename."""
    if filename.endswith("_chosen_only.parquet"):
        return filename[: -len("_chosen_only.parquet")]
    if filename.endswith(".parquet"):
        return filename[: -len(".parquet")]
    return filename


def _global_parquet_run_name_from_folder(part_name: str, run_name: str) -> str:
    """Returns the PC-independent parquet stem for a completed run folder."""
    prefix = f"{part_name}_seed{CURRENT_SEED}_"
    if not run_name.startswith(prefix):
        return run_name
    suffix = run_name[len(prefix):]
    parts = suffix.split("_")
    # create_run_output_dir uses YYYYMMDD_HHMMSS[_pc].
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return prefix + "_".join(parts[:2])
    return run_name


def _cleanup_incomplete_global_parquets(part_name: str, output_dir: str) -> dict[str, int]:
    """Removes only stale global parquet files for incomplete runs.

    Incomplete local run directories are intentionally preserved so interrupted
    debug sessions keep their logs and partial artifacts.
    """
    pattern = os.path.join(output_dir, f"{part_name}_seed{CURRENT_SEED}_*")
    completed_run_names: set[str] = set()
    removed_global_parquets = 0

    for folder in glob.glob(pattern):
        if not os.path.isdir(folder):
            continue
        run_name = os.path.basename(folder)
        if _is_completed_run_dir(folder):
            completed_run_names.add(run_name)
            completed_run_names.add(_global_parquet_run_name_from_folder(part_name, run_name))

    global_dir = os.path.join(output_dir, "_ALL_PARQUET_FILES")
    if os.path.isdir(global_dir):
        for parquet_path in glob.glob(os.path.join(global_dir, f"{part_name}_seed{CURRENT_SEED}_*.parquet")):
            run_name = _global_parquet_run_name(os.path.basename(parquet_path))
            run_dir = os.path.join(output_dir, run_name)
            if run_name in completed_run_names or _is_completed_run_dir(run_dir):
                continue
            try:
                os.remove(parquet_path)
                removed_global_parquets += 1
            except OSError as exc:
                print(f"[Warn] Failed to remove incomplete parquet {parquet_path}: {exc}")

    return {
        "removed_global_parquets": removed_global_parquets,
    }


def _already_collected(part_name: str, output_dir: str) -> bool:
    """Checks whether a part already has a fully completed dataset run."""
    pattern = os.path.join(output_dir, f"{part_name}_seed{CURRENT_SEED}_*")
    for folder in glob.glob(pattern):
        if _is_completed_run_dir(folder):
            return True
    return False


def _safe_name_component(text: str) -> str:
    """Returns a filesystem-safe name fragment."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("._-")


def _format_last_error() -> str:
    err = ctypes.get_last_error()
    if not err:
        return "unknown error"
    return f"WinError {err}"


def _create_worker_cleanup_job():
    """Creates a Windows job that kills leftover child processes on close."""
    if os.name != "nt" or not WORKER_JOB_CLEANUP:
        return None
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(_format_last_error())
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = _kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        _kernel32.CloseHandle(job)
        raise OSError(_format_last_error())
    return job


def _assign_process_to_cleanup_job(job, proc: subprocess.Popen) -> bool:
    if os.name != "nt" or not job:
        return False
    process_handle = wintypes.HANDLE(int(proc._handle))  # noqa: SLF001 - Windows Popen handle
    ok = _kernel32.AssignProcessToJobObject(job, process_handle)
    if not ok:
        raise OSError(_format_last_error())
    return True


def _close_worker_cleanup_job(job) -> None:
    if os.name == "nt" and job:
        _kernel32.CloseHandle(job)


def _run_worker_subprocess(cmd: list[str], log_f, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Runs one dataset worker and cleans up any child process tree leftovers."""
    if os.name != "nt" or not WORKER_JOB_CLEANUP:
        return subprocess.run(cmd, check=False, stdout=log_f, stderr=log_f, env=env)

    job = None
    proc = None
    assigned_to_job = False
    try:
        try:
            job = _create_worker_cleanup_job()
        except Exception as exc:
            print(f"[Warn] Worker cleanup job disabled: {exc}", file=log_f, flush=True)

        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)
        if job:
            try:
                assigned_to_job = _assign_process_to_cleanup_job(job, proc)
            except Exception as exc:
                print(f"[Warn] Failed to assign worker to cleanup job: {exc}", file=log_f, flush=True)
        returncode = proc.wait()
        return subprocess.CompletedProcess(cmd, returncode)
    finally:
        # Closing a kill-on-job-close job releases any NX helper processes that
        # survived after collect_axis_dataset.py itself exited.
        if assigned_to_job:
            print("[Info] Closing worker cleanup job", file=log_f, flush=True)
        _close_worker_cleanup_job(job)
        if proc is not None and proc.poll() is None:
            proc.kill()


def process_file_safe(file_info: tuple[str, str, str]) -> None:
    """Processes one part with lock-file protection to avoid duplicate work."""
    prt_path, output_dir, pc_name = file_info
    part_name = os.path.splitext(os.path.basename(prt_path))[0]
    pc_slug = _safe_name_component(pc_name)

    cleanup_stats = _cleanup_incomplete_global_parquets(part_name, output_dir)
    if cleanup_stats["removed_global_parquets"]:
        print(
            f"[Cleanup] {part_name}: removed {cleanup_stats['removed_global_parquets']} stale global parquet files"
        )

    if _already_collected(part_name, output_dir):
        return

    lock_file = os.path.join(output_dir, f"{part_name}.processing")
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL)
        os.close(fd)
    except (FileExistsError, OSError):
        return

    pc_text = f" pc={pc_slug}" if pc_slug else ""
    print(f"[Start] {part_name} (PID: {os.getpid()}{pc_text})")
    start_time = time.time()

    try:
        log_stem = f"{part_name}_seed{CURRENT_SEED}"
        if pc_slug:
            log_stem = f"{log_stem}_{pc_slug}"
        log_path = os.path.join(output_dir, f"{log_stem}.log")
        cmd = [
            PYTHON_EXE,
            WORKER_SCRIPT,
            "--input",
            prt_path,
            "--output",
            output_dir,
            "--seed",
            str(CURRENT_SEED),
        ]
        if pc_slug:
            cmd.extend(["--pc-name", pc_slug])
        env = os.environ.copy()
        if pc_slug:
            env["PC_NAME"] = pc_slug
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = _run_worker_subprocess(cmd, log_f=log_f, env=env)
        if result.returncode != 0:
            print(f"[Error] {part_name} failed (rc={result.returncode}), see {log_path}")
        else:
            print(f"[Done] {part_name} finished in {time.time() - start_time:.1f}s")
    except subprocess.CalledProcessError:
        print(f"[Error] Script failed for {part_name}")
    except Exception as exc:
        print(f"[Error] Unexpected error for {part_name}: {exc}")
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass


def main() -> None:
    """Parses arguments and executes multiprocessing workers."""
    parser = argparse.ArgumentParser(description="Parallel axis dataset collector runner")
    parser.add_argument("--cores", type=int, default=1, help="Number of local CPU cores to use")
    parser.add_argument("--pc-name", type=str, default=PC_NAME, help="PC label to include in log and output folder names")
    args = parser.parse_args()

    print(f"Using Python: {PYTHON_EXE}")
    print(f"Worker Script: {os.path.abspath(WORKER_SCRIPT)}")

    if not os.path.exists(SHARED_INPUT_DIR):
        print(f"[Error] Shared directory not found: {SHARED_INPUT_DIR}")
        return

    all_files = glob.glob(os.path.join(SHARED_INPUT_DIR, "*.prt"))
    random.shuffle(all_files)
    tasks = [(path, SHARED_OUTPUT_DIR, args.pc_name) for path in all_files]

    print(f"Found {len(all_files)} files. Starting with {args.cores} cores.")
    with multiprocessing.Pool(processes=args.cores, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(process_file_safe, tasks, chunksize=1):
            pass
    print("All tasks processed (or skipped).")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
