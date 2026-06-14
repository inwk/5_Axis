# step_converter_batch50.py
#
# Siemens NX Journal
# STEP/STP -> PRT converter
#
# 목적:
#   - 한 NX session에서 batch_size_per_session 개수만 처리
#   - 처리 후 정상 종료
#   - manifest 기반 자동 재시작
#   - 외부 launcher가 이 journal을 반복 실행하면
#     50개마다 NX session이 새로 열리는 효과가 됨
#
# 입력 구조:
#
# F:\Zero_to_CAD
#  ├─ test
#  │   └─ 000000000-000009999
#  │       ├─ 000000000
#  │       │   └─ model.step
#  │       └─ ...
#  └─ train
#      ├─ 000000000-000009999
#      │   ├─ 000000000
#      │   └─ ...
#      ├─ 000010000-000019999
#      └─ ...
#
# 출력:
#   3dDatasetZC0000.prt
#   3dDatasetZC0001.prt
#   ...

import os
import csv
import re
import shutil
import time
import traceback
import NXOpen


# ============================================================
# 사용자 설정 영역
# ============================================================

input_root = r"F:\Zero_to_CAD"

output_dir = r"Y:\04_개별폴더\22. 통합과정 오인욱\zc_prt_dataset"

# NX STEP translator 작업용 로컬 임시 폴더
local_work_root = r"C:\nx_step_import_tmp"

output_prefix = "3dDatasetZC"
output_digits = 4

dataset_splits = ["test", "train"]

target_step_names = [
    "model.step",
    "model.stp",
    "model.STEP",
    "model.STP"
]

overwrite_existing = False

auto_resume = True

manual_start_sequence_index = 0

# 핵심:
# 한 NX session에서 50개만 처리하고 종료.
# 외부 launcher가 다시 NX를 열어서 다음 50개를 처리함.
batch_size_per_session = 50

progress_log_interval = 10
scan_log_interval = 1000

manifest_filename = "zc_prt_conversion_manifest.csv"
fail_log_filename = "zc_prt_conversion_fail_log.txt"
state_filename = "zc_prt_conversion_state.txt"

stop_on_fatal_error = True

# ============================================================


def write_listing(the_session, message):
    text = str(message)

    try:
        lw = the_session.ListingWindow
        if not lw.IsOpen:
            lw.Open()
        lw.WriteLine(text)
    except Exception:
        pass


def normalize_path(path_text):
    return os.path.normpath(path_text.strip().strip('"'))


def safe_listdir(path):
    try:
        return os.listdir(path)
    except Exception:
        return []


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def sorted_subdirs(path):
    result = []

    for name in safe_listdir(path):
        full_path = os.path.join(path, name)

        if os.path.isdir(full_path):
            result.append(name)

    result.sort()
    return result


def find_target_step_in_id_folder(id_folder):
    for step_name in target_step_names:
        candidate = os.path.join(id_folder, step_name)

        if os.path.isfile(candidate):
            return candidate

    return None


def make_output_name(sequence_index):
    return "{}{:0{}d}.prt".format(
        output_prefix,
        sequence_index,
        output_digits
    )


def make_output_path(output_directory, sequence_index):
    return os.path.join(output_directory, make_output_name(sequence_index))


def get_manifest_path(output_directory):
    return os.path.join(output_directory, manifest_filename)


def get_fail_log_path(output_directory):
    return os.path.join(output_directory, fail_log_filename)


def get_state_path(output_directory):
    return os.path.join(output_directory, state_filename)


def write_state(output_directory, state, message, extra_fields=None):
    state_path = get_state_path(output_directory)

    try:
        with open(state_path, "w", encoding="utf-8") as f:
            f.write("state={}\n".format(state))
            f.write("message={}\n".format(message))

            if extra_fields:
                for key, value in extra_fields.items():
                    safe_value = str(value).replace("\r", " ").replace("\n", " ")
                    f.write("{}={}\n".format(key, safe_value))
    except Exception:
        pass


def ensure_manifest_header(output_directory):
    manifest_path = get_manifest_path(output_directory)

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


def append_manifest_row(
    output_directory,
    sequence_index,
    split_name,
    range_folder_name,
    id_folder_name,
    step_path,
    output_path,
    status,
    error_text
):
    manifest_path = get_manifest_path(output_directory)

    with open(manifest_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            sequence_index,
            split_name,
            range_folder_name,
            id_folder_name,
            step_path,
            output_path,
            status,
            error_text
        ])


def write_running_file_state(
    output_directory,
    sequence_index,
    split_name,
    range_folder_name,
    id_folder_name,
    step_path,
    output_path
):
    write_state(
        output_directory,
        "RUNNING_FILE",
        "STEP conversion running",
        {
            "started_at": "{:.6f}".format(time.time()),
            "sequence_index": sequence_index,
            "split": split_name,
            "range_folder": range_folder_name,
            "id_folder": id_folder_name,
            "step_path": step_path,
            "output_path": output_path
        }
    )


def append_fail_log(output_directory, sequence_index, step_path, error, error_trace):
    fail_log_path = get_fail_log_path(output_directory)

    try:
        with open(fail_log_path, "a", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write("sequence_index: {}\n".format(sequence_index))
            f.write("step_path: {}\n".format(step_path))
            f.write("error: {}\n".format(error))
            f.write("traceback:\n")
            f.write(str(error_trace))
            f.write("\n")
    except Exception:
        pass


def read_resume_info_from_manifest(output_directory):
    manifest_path = get_manifest_path(output_directory)

    if not os.path.exists(manifest_path):
        return None, None

    valid_statuses = set([
        "success",
        "skip_existing_prt",
        "skip_timeout"
    ])

    max_sequence_index = None
    last_valid_row = None

    try:
        with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                status = row.get("status", "")

                if status not in valid_statuses:
                    continue

                try:
                    sequence_index = int(row.get("sequence_index", ""))
                except Exception:
                    continue

                if max_sequence_index is None or sequence_index > max_sequence_index:
                    max_sequence_index = sequence_index
                    last_valid_row = row

    except Exception:
        return None, None

    if max_sequence_index is None:
        return None, None

    return max_sequence_index + 1, last_valid_row


def read_resume_index_from_existing_prt_files(output_directory):
    if not os.path.isdir(output_directory):
        return None

    pattern = re.compile(
        r"^" + re.escape(output_prefix) + r"(\d+)\.prt$",
        re.IGNORECASE
    )

    max_index = None

    for name in safe_listdir(output_directory):
        match = pattern.match(name)

        if not match:
            continue

        try:
            index = int(match.group(1))
        except Exception:
            continue

        if max_index is None or index > max_index:
            max_index = index

    if max_index is None:
        return None

    return max_index + 1


def determine_resume_info(the_session, output_directory):
    if not auto_resume:
        write_listing(
            the_session,
            "자동 재시작 비활성화. manual_start_sequence_index={}".format(
                manual_start_sequence_index
            )
        )
        return manual_start_sequence_index, None

    manifest_resume_index, last_valid_row = read_resume_info_from_manifest(output_directory)

    if manifest_resume_index is not None:
        write_listing(the_session, "자동 재시작: manifest 기준으로 재개합니다.")
        write_listing(the_session, "다음 시작 sequence_index: {}".format(manifest_resume_index))

        if last_valid_row is not None:
            write_listing(
                the_session,
                "마지막 정상 기록: sequence_index={}, split={}, range={}, id={}, status={}".format(
                    last_valid_row.get("sequence_index", ""),
                    last_valid_row.get("split", ""),
                    last_valid_row.get("range_folder", ""),
                    last_valid_row.get("id_folder", ""),
                    last_valid_row.get("status", "")
                )
            )

        return manifest_resume_index, last_valid_row

    existing_resume_index = read_resume_index_from_existing_prt_files(output_directory)

    if existing_resume_index is not None:
        write_listing(the_session, "manifest 정상 기록은 없지만 기존 PRT 파일 기준으로 재개합니다.")
        write_listing(the_session, "기존 PRT 기준 다음 시작 sequence_index: {}".format(existing_resume_index))
        return existing_resume_index, None

    write_listing(the_session, "기존 정상 기록 없음. 0번부터 시작합니다.")
    return 0, None


def split_index_map():
    result = {}

    for i, split_name in enumerate(dataset_splits):
        result[split_name] = i

    return result


def should_skip_split_by_resume(split_name, resume_row):
    if resume_row is None:
        return False

    resume_split = resume_row.get("split", "")

    mapping = split_index_map()

    if split_name not in mapping or resume_split not in mapping:
        return False

    return mapping[split_name] < mapping[resume_split]


def should_skip_range_by_resume(split_name, range_folder_name, resume_row):
    if resume_row is None:
        return False

    resume_split = resume_row.get("split", "")
    resume_range = resume_row.get("range_folder", "")

    if split_name != resume_split:
        return False

    return range_folder_name < resume_range


def should_skip_id_by_resume(split_name, range_folder_name, id_folder_name, resume_row):
    if resume_row is None:
        return False

    resume_split = resume_row.get("split", "")
    resume_range = resume_row.get("range_folder", "")
    resume_id = resume_row.get("id_folder", "")

    if split_name != resume_split:
        return False

    if range_folder_name != resume_range:
        return False

    return id_folder_name <= resume_id


def close_part_safely(the_session, part):
    if part is None:
        return

    try:
        part.Close(
            NXOpen.BasePart.CloseWholeTree.TrueValue,
            NXOpen.BasePart.CloseModified.CloseModified,
            None
        )
        return
    except Exception:
        pass

    try:
        close_responses = the_session.Parts.NewPartCloseResponses()

        part.Close(
            NXOpen.BasePart.CloseWholeTree.TrueValue,
            NXOpen.BasePart.CloseModified.UseResponses,
            close_responses
        )

        close_responses.Dispose()
        return
    except Exception:
        pass

    try:
        part.Close(
            NXOpen.BasePart.CloseWholeTree.FalseValue,
            NXOpen.BasePart.CloseModified.CloseModified,
            None
        )
    except Exception:
        pass


def close_all_open_parts_safely(the_session):
    try:
        work_part = the_session.Parts.Work

        if work_part is not None:
            close_part_safely(the_session, work_part)
    except Exception:
        pass


def set_builder_value(builder, property_name, value):
    try:
        setattr(builder, property_name, value)
        return True
    except Exception:
        pass

    setter_name = "Set" + property_name

    try:
        setter = getattr(builder, setter_name)
        setter(value)
        return True
    except Exception:
        pass

    return False


def set_process_hold_flag(builder, value):
    try:
        builder.ProcessHoldFlag = value
        return True
    except Exception:
        pass

    try:
        builder.SetProcessHoldFlag(value)
        return True
    except Exception:
        pass

    return False


def set_import_to_new_part(step_importer):
    candidates = []

    try:
        enum_type = NXOpen.Step214Importer.ImportToOption

        for name in [
            "ImportToOptionNewPart",
            "NewPart"
        ]:
            try:
                candidates.append(getattr(enum_type, name))
            except Exception:
                pass
    except Exception:
        pass

    for value in candidates:
        if set_builder_value(step_importer, "ImportTo", value):
            return True

    return False


def configure_step_importer(the_session, step_importer, step_path, local_output_path):
    if not set_builder_value(step_importer, "InputFile", step_path):
        raise RuntimeError("Step214Importer InputFile 설정 실패")

    if not set_builder_value(step_importer, "OutputFile", local_output_path):
        raise RuntimeError("Step214Importer OutputFile 설정 실패")

    if not set_builder_value(step_importer, "FileOpenFlag", False):
        raise RuntimeError("Step214Importer FileOpenFlag=False 설정 실패")

    if not set_import_to_new_part(step_importer):
        raise RuntimeError("Step214Importer ImportTo=NewPart 설정 실패")

    set_process_hold_flag(step_importer, True)

    set_builder_value(step_importer, "ImportToTeamcenter", False)
    set_builder_value(step_importer, "FlattenAssembly", True)
    set_builder_value(step_importer, "Optimize", True)
    set_builder_value(step_importer, "SewSurfaces", True)
    set_builder_value(step_importer, "SimplifyGeometry", True)
    set_builder_value(step_importer, "SmoothBSurfaces", True)

    try:
        step_importer.ObjectTypes.Solids = True
        step_importer.ObjectTypes.Surfaces = True
        step_importer.ObjectTypes.Curves = True
        step_importer.ObjectTypes.Csys = True
        step_importer.ObjectTypes.ProductData = True
        step_importer.ObjectTypes.PmiData = True
    except Exception:
        pass

    try:
        step214_dir = the_session.GetEnvironmentVariableValue("STEP214UG_DIR")
        settings_file = os.path.join(step214_dir, "step214ug.def")

        if os.path.isfile(settings_file):
            set_builder_value(step_importer, "SettingsFile", settings_file)
    except Exception:
        pass


def create_clean_work_dir(local_work_root_checked, sequence_index):
    ensure_dir(local_work_root_checked)

    work_dir_name = "zc_{:09d}".format(sequence_index)
    work_dir = os.path.join(local_work_root_checked, work_dir_name)

    if os.path.isdir(work_dir):
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    ensure_dir(work_dir)

    return work_dir


def cleanup_work_dir(work_dir):
    try:
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
    except Exception:
        pass


def find_created_prt_in_work_dir(work_dir, preferred_path):
    if os.path.isfile(preferred_path):
        return preferred_path

    candidates = []

    for name in safe_listdir(work_dir):
        if name.lower().endswith(".prt"):
            full_path = os.path.join(work_dir, name)

            try:
                mtime = os.path.getmtime(full_path)
            except Exception:
                mtime = 0

            candidates.append((mtime, full_path))

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][1]


def atomic_copy_to_final(local_prt_path, final_output_path):
    final_dir = os.path.dirname(final_output_path)
    ensure_dir(final_dir)

    temp_final_path = final_output_path + ".copying"

    if os.path.exists(temp_final_path):
        try:
            os.remove(temp_final_path)
        except Exception:
            pass

    shutil.copy2(local_prt_path, temp_final_path)

    if os.path.exists(final_output_path):
        os.remove(final_output_path)

    os.replace(temp_final_path, final_output_path)


def convert_one_step_to_prt(
    the_session,
    step_path,
    final_output_path,
    sequence_index,
    local_work_root_checked
):
    step_importer = None
    work_dir = None
    previous_cwd = None

    try:
        close_all_open_parts_safely(the_session)

        work_dir = create_clean_work_dir(
            local_work_root_checked,
            sequence_index
        )

        local_output_path = os.path.join(
            work_dir,
            make_output_name(sequence_index)
        )

        previous_cwd = os.getcwd()

        try:
            os.chdir(work_dir)
        except Exception:
            previous_cwd = None

        step_importer = the_session.DexManager.CreateStep214Importer()

        configure_step_importer(
            the_session,
            step_importer,
            step_path,
            local_output_path
        )

        step_importer.Commit()

        try:
            step_importer.Destroy()
            step_importer = None
        except Exception:
            pass

        close_all_open_parts_safely(the_session)

        created_prt = find_created_prt_in_work_dir(
            work_dir,
            local_output_path
        )

        if created_prt is None:
            raise RuntimeError(
                "Importer Commit 후 로컬 작업 폴더에 PRT가 생성되지 않았습니다: {}".format(
                    work_dir
                )
            )

        atomic_copy_to_final(created_prt, final_output_path)

        if previous_cwd is not None:
            try:
                os.chdir(previous_cwd)
            except Exception:
                pass

        cleanup_work_dir(work_dir)

        if not os.path.isfile(final_output_path):
            raise RuntimeError(
                "최종 PRT 복사 후 파일이 존재하지 않습니다: {}".format(
                    final_output_path
                )
            )

        return True, None, None

    except Exception as ex:
        error_trace = traceback.format_exc()

        if step_importer is not None:
            try:
                step_importer.Destroy()
            except Exception:
                pass

        close_all_open_parts_safely(the_session)

        if previous_cwd is not None:
            try:
                os.chdir(previous_cwd)
            except Exception:
                pass

        if work_dir is not None:
            cleanup_work_dir(work_dir)

        return False, ex, error_trace


def is_fatal_nx_error(error_text):
    lower_text = str(error_text).lower()

    fatal_keywords = [
        "memory access violation",
        "internal error",
        "file already exists"
    ]

    for keyword in fatal_keywords:
        if keyword in lower_text:
            return True

    return False


def process_step_file(
    the_session,
    step_path,
    output_path,
    output_directory,
    sequence_index,
    local_work_root_checked
):
    if os.path.exists(output_path):
        if overwrite_existing:
            try:
                os.remove(output_path)
                write_listing(
                    the_session,
                    "기존 파일 삭제 후 덮어쓰기: {}".format(output_path)
                )
            except Exception as ex:
                error_trace = traceback.format_exc()

                append_fail_log(
                    output_directory,
                    sequence_index,
                    step_path,
                    ex,
                    error_trace
                )

                return "fail_delete_existing_prt", str(ex)
        else:
            return "skip_existing_prt", ""

    ok, error, error_trace = convert_one_step_to_prt(
        the_session,
        step_path,
        output_path,
        sequence_index,
        local_work_root_checked
    )

    if ok:
        return "success", ""

    error_text = str(error)

    append_fail_log(
        output_directory,
        sequence_index,
        step_path,
        error,
        error_trace
    )

    if is_fatal_nx_error(error_text):
        return "fatal_import_error", error_text

    return "fail_convert", error_text


def should_stop_by_batch_limit(convert_attempt_count):
    return convert_attempt_count >= batch_size_per_session


def main():
    the_session = NXOpen.Session.GetSession()

    input_root_checked = normalize_path(input_root)
    output_dir_checked = normalize_path(output_dir)
    local_work_root_checked = normalize_path(local_work_root)

    write_listing(the_session, "===== Journal 시작 =====")
    write_listing(the_session, "50개 단위 NX session batch 변환 모드")
    write_listing(the_session, "입력 루트: {}".format(input_root_checked))
    write_listing(the_session, "출력 폴더: {}".format(output_dir_checked))
    write_listing(the_session, "로컬 작업 폴더: {}".format(local_work_root_checked))
    write_listing(the_session, "batch_size_per_session: {}".format(batch_size_per_session))
    write_listing(the_session, "auto_resume: {}".format(auto_resume))
    write_listing(the_session, "overwrite_existing: {}".format(overwrite_existing))

    if not os.path.isdir(input_root_checked):
        ensure_dir(output_dir_checked)
        write_state(output_dir_checked, "FATAL", "입력 폴더가 존재하지 않습니다.")
        raise RuntimeError("입력 폴더가 존재하지 않습니다: {}".format(input_root_checked))

    ensure_dir(output_dir_checked)
    ensure_dir(local_work_root_checked)

    write_state(output_dir_checked, "RUNNING", "NX journal 실행 중")

    ensure_manifest_header(output_dir_checked)

    resume_start_index, resume_row = determine_resume_info(
        the_session,
        output_dir_checked
    )

    try:
        the_session.ApplicationSwitchImmediate("UG_APP_MANUFACTURING")
        write_listing(the_session, "NX Manufacturing/CAM 환경 전환 완료")
    except Exception as ex:
        write_listing(the_session, "NX Manufacturing/CAM 환경 전환 실패 또는 불필요: {}".format(ex))

    close_all_open_parts_safely(the_session)

    if resume_row is not None:
        sequence_index = resume_start_index
        count_based_resume = False
    else:
        sequence_index = 0
        count_based_resume = True

    checked_id_folder_count = 0
    found_step_count = 0
    skipped_before_resume_count = 0
    convert_attempt_count = 0

    success_count = 0
    skip_count = 0
    fail_count = 0
    missing_step_count = 0

    stop_requested = False
    all_done = True

    for split_name in dataset_splits:
        if stop_requested:
            all_done = False
            break

        if should_skip_split_by_resume(split_name, resume_row):
            write_listing(the_session, "resume 위치 이전 split 건너뜀: {}".format(split_name))
            continue

        split_path = os.path.join(input_root_checked, split_name)

        if not os.path.isdir(split_path):
            write_listing(
                the_session,
                "경고: split 폴더 없음, 건너뜀: {}".format(split_path)
            )
            continue

        write_listing(the_session, "===== split 처리 시작: {} =====".format(split_name))

        range_folder_names = sorted_subdirs(split_path)

        write_listing(
            the_session,
            "{} 하위 range 폴더 수: {}".format(
                split_name,
                len(range_folder_names)
            )
        )

        for range_folder_name in range_folder_names:
            if stop_requested:
                all_done = False
                break

            if should_skip_range_by_resume(split_name, range_folder_name, resume_row):
                continue

            range_folder_path = os.path.join(split_path, range_folder_name)

            write_listing(
                the_session,
                "range 폴더 처리 시작: {}".format(range_folder_path)
            )

            id_folder_names = sorted_subdirs(range_folder_path)

            write_listing(
                the_session,
                "개별 ID 폴더 수: {}".format(len(id_folder_names))
            )

            for id_folder_name in id_folder_names:
                checked_id_folder_count += 1

                if checked_id_folder_count % scan_log_interval == 0:
                    write_listing(
                        the_session,
                        "스캔 진행: ID 폴더 {}개 확인, STEP {}개 발견, 현재 sequence_index={}, 재시작 기준={}".format(
                            checked_id_folder_count,
                            found_step_count,
                            sequence_index,
                            resume_start_index
                        )
                    )

                if should_skip_id_by_resume(split_name, range_folder_name, id_folder_name, resume_row):
                    continue

                id_folder_path = os.path.join(range_folder_path, id_folder_name)

                step_path = find_target_step_in_id_folder(id_folder_path)

                if step_path is None:
                    missing_step_count += 1
                    continue

                found_step_count += 1

                if count_based_resume:
                    current_sequence_index = sequence_index
                    sequence_index += 1

                    if current_sequence_index < resume_start_index:
                        skipped_before_resume_count += 1
                        continue
                else:
                    current_sequence_index = sequence_index
                    sequence_index += 1

                if should_stop_by_batch_limit(convert_attempt_count):
                    stop_requested = True
                    all_done = False
                    break

                convert_attempt_count += 1

                output_path = make_output_path(
                    output_dir_checked,
                    current_sequence_index
                )

                if (
                    convert_attempt_count == 1 or
                    convert_attempt_count % progress_log_interval == 0
                ):
                    write_listing(
                        the_session,
                        "[변환 진행] 이번 세션 시도 {}, sequence_index={}, STEP={}".format(
                            convert_attempt_count,
                            current_sequence_index,
                            step_path
                        )
                    )
                    write_listing(
                        the_session,
                        "저장 대상: {}".format(output_path)
                    )

                write_running_file_state(
                    output_dir_checked,
                    current_sequence_index,
                    split_name,
                    range_folder_name,
                    id_folder_name,
                    step_path,
                    output_path
                )

                status, error_text = process_step_file(
                    the_session,
                    step_path,
                    output_path,
                    output_dir_checked,
                    current_sequence_index,
                    local_work_root_checked
                )

                append_manifest_row(
                    output_dir_checked,
                    current_sequence_index,
                    split_name,
                    range_folder_name,
                    id_folder_name,
                    step_path,
                    output_path,
                    status,
                    error_text
                )

                if status == "success":
                    success_count += 1
                elif status.startswith("skip"):
                    skip_count += 1
                elif status.startswith("fatal"):
                    fail_count += 1
                    all_done = False

                    write_listing(the_session, "치명적 NX import 오류 발생. 즉시 중단합니다.")
                    write_listing(
                        the_session,
                        "status={}, sequence_index={}, STEP={}".format(
                            status,
                            current_sequence_index,
                            step_path
                        )
                    )
                    write_listing(the_session, "error={}".format(error_text))

                    write_state(
                        output_dir_checked,
                        "FATAL",
                        "sequence_index={}, error={}".format(
                            current_sequence_index,
                            error_text
                        )
                    )

                    if stop_on_fatal_error:
                        stop_requested = True
                        break
                else:
                    fail_count += 1

                if (
                    convert_attempt_count == 1 or
                    convert_attempt_count % progress_log_interval == 0
                ):
                    write_listing(
                        the_session,
                        "현재 결과: 성공 {}, 스킵 {}, 실패 {}, 이번 세션 시도 {}".format(
                            success_count,
                            skip_count,
                            fail_count,
                            convert_attempt_count
                        )
                    )

            write_listing(
                the_session,
                "range 폴더 처리 완료: {}".format(range_folder_path)
            )
            write_listing(
                the_session,
                "누적: 다음 sequence_index={}, 성공 {}, 스킵 {}, 실패 {}, STEP 없음 {}".format(
                    sequence_index,
                    success_count,
                    skip_count,
                    fail_count,
                    missing_step_count
                )
            )

        write_listing(the_session, "===== split 처리 완료: {} =====".format(split_name))

    write_listing(the_session, "===== 변환 종료 =====")
    write_listing(the_session, "자동 재시작 기준 sequence_index: {}".format(resume_start_index))
    write_listing(the_session, "이번 NX session 처리 시도 수: {}".format(convert_attempt_count))
    write_listing(the_session, "성공: {}개".format(success_count))
    write_listing(the_session, "스킵: {}개".format(skip_count))
    write_listing(the_session, "실패: {}개".format(fail_count))
    write_listing(the_session, "다음 sequence_index 후보: {}".format(sequence_index))
    write_listing(the_session, "manifest 파일: {}".format(get_manifest_path(output_dir_checked)))
    write_listing(the_session, "실패 로그 파일: {}".format(get_fail_log_path(output_dir_checked)))

    if fail_count > 0 and stop_requested:
        write_listing(the_session, "FATAL 상태로 종료합니다.")
    elif all_done:
        write_state(output_dir_checked, "ALL_DONE", "모든 데이터 처리 완료")
        write_listing(the_session, "모든 데이터 처리가 완료되었습니다.")
    else:
        write_state(
            output_dir_checked,
            "BATCH_DONE",
            "이번 NX session에서 {}개 처리 완료. 다음 session에서 이어집니다.".format(
                convert_attempt_count
            )
        )
        write_listing(
            the_session,
            "batch_size_per_session={}에 도달하여 정상 종료합니다. 다음 NX session에서 이어집니다.".format(
                batch_size_per_session
            )
        )

    write_listing(the_session, "===== Journal 종료 =====")


def GetUnloadOption(dummy):
    return NXOpen.Session.LibraryUnloadOption.Immediately


if __name__ == "__main__":
    main()
