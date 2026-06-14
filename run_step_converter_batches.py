# run_step_converter_batches.py
#
# NX 외부에서 실행하는 launcher.
# run_journal.exe를 반복 호출해서
# NX session을 50개마다 새로 열고 닫는 구조.
#
# 실행:
#   python run_step_converter_batches.py

import os
import csv
import subprocess
import time


# ============================================================
# 사용자 설정 영역
# ============================================================

# run_journal.exe 경로를 본인 NX 설치 경로에 맞게 수정.
#
# 예시 후보:
#   C:\Program Files\Siemens\NX2312\NXBIN\run_journal.exe
#   C:\Program Files\Siemens\NX2206\NXBIN\run_journal.exe
#   C:\Program Files\Siemens\NX2007\NXBIN\run_journal.exe
#
run_journal_exe = r"C:\Program Files\Siemens\DesigncenterNX2512\NXBIN\run_journal.exe"

# 위 1번 파일 경로
journal_path = r"C:\Users\inwoo\Desktop\5_Axis\step_converter_batch50.py"

# Journal 코드의 output_dir와 동일해야 함
output_dir = r"Y:\04_개별폴더\22. 통합과정 오인욱\zc_prt_dataset"

state_filename = "zc_prt_conversion_state.txt"

# 무한 루프 방지용 최대 NX session 실행 횟수
max_sessions = 100000

# 각 session 사이 대기 시간, 초
sleep_seconds_between_sessions = 3

# 한 STEP 파일 변환이 이 시간(초)을 넘기면 해당 파일은 skip_timeout으로 기록하고 넘어감
per_file_timeout_seconds = 300

# ============================================================


def get_state_path():
    return os.path.join(output_dir, state_filename)


def get_manifest_path():
    return os.path.join(output_dir, "zc_prt_conversion_manifest.csv")


def read_state_details():
    state_path = get_state_path()

    if not os.path.exists(state_path):
        return "UNKNOWN", "", {}

    state = "UNKNOWN"
    message = ""
    fields = {}

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line.startswith("state="):
                    state = line[len("state="):]

                if line.startswith("message="):
                    message = line[len("message="):]

                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key] = value

    except Exception as ex:
        return "UNKNOWN", str(ex), {}

    return state, message, fields


def read_state():
    state, message, _fields = read_state_details()
    return state, message


def write_state(state, message):
    try:
        with open(get_state_path(), "w", encoding="utf-8") as f:
            f.write("state={}\n".format(state))
            f.write("message={}\n".format(message))
    except Exception:
        pass


def ensure_manifest_header():
    manifest_path = get_manifest_path()

    if os.path.exists(manifest_path):
        return

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sequence_index",
            "split",
            "range_folder",
            "id_folder",
            "step_path",
            "output_path",
            "status",
            "error"
        ])


def manifest_has_completed_sequence(sequence_index):
    manifest_path = get_manifest_path()
    completed_statuses = set([
        "success",
        "skip_existing_prt",
        "skip_timeout"
    ])

    if not os.path.exists(manifest_path):
        return False

    try:
        with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if (
                    row.get("sequence_index", "") == str(sequence_index) and
                    row.get("status", "") in completed_statuses
                ):
                    return True
    except Exception:
        return False

    return False


def append_timeout_manifest(fields):
    sequence_index = fields.get("sequence_index", "")

    if sequence_index and manifest_has_completed_sequence(sequence_index):
        return

    ensure_manifest_header()

    with open(get_manifest_path(), "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            sequence_index,
            fields.get("split", ""),
            fields.get("range_folder", ""),
            fields.get("id_folder", ""),
            fields.get("step_path", ""),
            fields.get("output_path", ""),
            "skip_timeout",
            "conversion exceeded {} seconds".format(per_file_timeout_seconds)
        ])


def terminate_process_tree(process):
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False
        )
        return

    process.terminate()


def validate_paths():
    if not os.path.isfile(run_journal_exe):
        raise RuntimeError("run_journal.exe를 찾을 수 없습니다: {}".format(run_journal_exe))

    if not os.path.isfile(journal_path):
        raise RuntimeError("journal 파일을 찾을 수 없습니다: {}".format(journal_path))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def run_one_nx_session(session_index):
    print("============================================================")
    print("NX session {} 시작".format(session_index))
    print("Journal: {}".format(journal_path))
    print("per_file_timeout_seconds={}".format(per_file_timeout_seconds))

    cmd = [
        run_journal_exe,
        journal_path
    ]

    process = subprocess.Popen(
        cmd,
        shell=False
    )

    timed_out = False
    timeout_fields = None

    while True:
        returncode = process.poll()

        if returncode is not None:
            break

        state, _message, fields = read_state_details()

        if state == "RUNNING_FILE":
            try:
                started_at = float(fields.get("started_at", "0"))
            except Exception:
                started_at = 0

            if started_at > 0 and time.time() - started_at > per_file_timeout_seconds:
                timed_out = True
                timeout_fields = fields
                print("STEP 변환 timeout. NX session을 종료하고 해당 파일을 skip합니다.")
                print("sequence_index={}".format(fields.get("sequence_index", "")))
                print("STEP={}".format(fields.get("step_path", "")))
                terminate_process_tree(process)
                break

        time.sleep(2)

    if timed_out:
        try:
            process.wait(timeout=10)
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()

        if timeout_fields is not None:
            append_timeout_manifest(timeout_fields)
            write_state(
                "BATCH_DONE",
                "sequence_index={} skipped by timeout".format(
                    timeout_fields.get("sequence_index", "")
                )
            )

        returncode = process.returncode

    print("NX session {} 종료, returncode={}".format(
        session_index,
        returncode
    ))

    return returncode, timed_out


def main():
    validate_paths()

    for session_index in range(1, max_sessions + 1):
        returncode, timed_out = run_one_nx_session(session_index)

        state, message = read_state()

        print("state={}".format(state))
        print("message={}".format(message))

        if timed_out:
            print("{}초 후 다음 NX session을 실행합니다.".format(
                sleep_seconds_between_sessions
            ))
            time.sleep(sleep_seconds_between_sessions)
            continue

        if state == "ALL_DONE":
            print("전체 변환 완료.")
            break

        if state == "FATAL":
            print("FATAL 오류 발생. 자동 반복을 중단합니다.")
            print(message)
            break

        if returncode != 0:
            print("run_journal.exe returncode가 0이 아닙니다. 자동 반복을 중단합니다.")
            break

        if state in ["BATCH_DONE", "RUNNING", "UNKNOWN"]:
            print("{}초 후 다음 NX session을 실행합니다.".format(
                sleep_seconds_between_sessions
            ))
            time.sleep(sleep_seconds_between_sessions)
            continue

    else:
        print("max_sessions={}에 도달하여 중단합니다.".format(max_sessions))


if __name__ == "__main__":
    main()
