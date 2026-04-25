import csv
import json
from pathlib import Path


BASE_DIR = Path("pads-parkinsons-disease-smartwatch-dataset")
OUTPUT_DIR = Path("pads_cleaned_csv")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def read_json(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def bool_to_flag(value):
    if value is True:
        return 1
    if value is False:
        return 0
    return ""


def write_csv(file_path, fieldnames, rows):
    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_patients():
    rows = []

    for file_path in sorted((BASE_DIR / "patients").glob("*.json")):
        data = read_json(file_path)
        rows.append(
            {
                "patient_id": data.get("id", ""),
                "study_id": data.get("study_id", ""),
                "condition": data.get("condition", ""),
                "disease_comment": data.get("disease_comment", ""),
                "age_at_diagnosis": data.get("age_at_diagnosis", ""),
                "current_age": data.get("age", ""),
                "height_cm": data.get("height", ""),
                "weight_kg": data.get("weight", ""),
                "gender": data.get("gender", ""),
                "handedness": data.get("handedness", ""),
                "appearance_in_kinship": bool_to_flag(
                    data.get("appearance_in_kinship")
                ),
                "appearance_in_first_grade_kinship": bool_to_flag(
                    data.get("appearance_in_first_grade_kinship")
                ),
                "effect_of_alcohol_on_tremor": data.get(
                    "effect_of_alcohol_on_tremor", ""
                ),
            }
        )

    fieldnames = [
        "patient_id",
        "study_id",
        "condition",
        "disease_comment",
        "age_at_diagnosis",
        "current_age",
        "height_cm",
        "weight_kg",
        "gender",
        "handedness",
        "appearance_in_kinship",
        "appearance_in_first_grade_kinship",
        "effect_of_alcohol_on_tremor",
    ]
    write_csv(OUTPUT_DIR / "patients.csv", fieldnames, rows)
    return rows


def clean_questionnaires():
    answer_rows = []
    summary_rows = []

    for file_path in sorted((BASE_DIR / "questionnaire").glob("*.json")):
        data = read_json(file_path)
        subject_id = data.get("subject_id", "")
        questionnaire_id = data.get("id", "")
        questionnaire_name = data.get("questionnaire_name", "")
        study_id = data.get("study_id", "")
        yes_count = 0
        total_questions = 0

        for item in data.get("item", []):
            answer_flag = bool_to_flag(item.get("answer"))
            if answer_flag == 1:
                yes_count += 1
            total_questions += 1

            answer_rows.append(
                {
                    "patient_id": subject_id,
                    "study_id": study_id,
                    "questionnaire_id": questionnaire_id,
                    "questionnaire_name": questionnaire_name,
                    "question_id": item.get("link_id", ""),
                    "question_text": item.get("text", ""),
                    "answer_flag": answer_flag,
                }
            )

        summary_rows.append(
            {
                "patient_id": subject_id,
                "study_id": study_id,
                "questionnaire_id": questionnaire_id,
                "questionnaire_name": questionnaire_name,
                "total_questions": total_questions,
                "yes_answers": yes_count,
                "yes_answer_rate": round(yes_count / total_questions, 4)
                if total_questions
                else "",
            }
        )

    answer_fieldnames = [
        "patient_id",
        "study_id",
        "questionnaire_id",
        "questionnaire_name",
        "question_id",
        "question_text",
        "answer_flag",
    ]
    summary_fieldnames = [
        "patient_id",
        "study_id",
        "questionnaire_id",
        "questionnaire_name",
        "total_questions",
        "yes_answers",
        "yes_answer_rate",
    ]

    write_csv(OUTPUT_DIR / "questionnaire_answers.csv", answer_fieldnames, answer_rows)
    write_csv(OUTPUT_DIR / "questionnaire_summary.csv", summary_fieldnames, summary_rows)
    return answer_rows, summary_rows


def clean_movement():
    session_rows = []
    record_rows = []

    for file_path in sorted((BASE_DIR / "movement").glob("*.json")):
        data = read_json(file_path)

        for session in data.get("session", []):
            records = session.get("records", [])
            session_rows.append(
                {
                    "patient_id": data.get("subject_id", ""),
                    "study_id": data.get("study_id", ""),
                    "assessment_id": data.get("id", ""),
                    "device_id": data.get("device_id", ""),
                    "sampling_rate_hz": data.get("sampling_rate", ""),
                    "endianness": data.get("endianness", ""),
                    "data_type": data.get("data_type", ""),
                    "bits": data.get("bits", ""),
                    "task_name": session.get("record_name", ""),
                    "expected_rows_per_record": session.get("rows", ""),
                    "record_count": len(records),
                }
            )

            for record in records:
                record_rows.append(
                    {
                        "patient_id": data.get("subject_id", ""),
                        "study_id": data.get("study_id", ""),
                        "assessment_id": data.get("id", ""),
                        "task_name": session.get("record_name", ""),
                        "expected_rows_per_record": session.get("rows", ""),
                        "device_location": record.get("device_location", ""),
                        "file_name": record.get("file_name", ""),
                        "channels": "|".join(record.get("channels", [])),
                        "units": "|".join(record.get("units", [])),
                    }
                )

    session_fieldnames = [
        "patient_id",
        "study_id",
        "assessment_id",
        "device_id",
        "sampling_rate_hz",
        "endianness",
        "data_type",
        "bits",
        "task_name",
        "expected_rows_per_record",
        "record_count",
    ]
    record_fieldnames = [
        "patient_id",
        "study_id",
        "assessment_id",
        "task_name",
        "expected_rows_per_record",
        "device_location",
        "file_name",
        "channels",
        "units",
    ]

    write_csv(OUTPUT_DIR / "movement_sessions.csv", session_fieldnames, session_rows)
    write_csv(OUTPUT_DIR / "movement_records.csv", record_fieldnames, record_rows)
    return session_rows, record_rows


def empty_sensor_stats():
    return {
        "row_count": 0,
        "time_start": "",
        "time_end": "",
        "duration_seconds": "",
        "accelerometer_x_mean": "",
        "accelerometer_x_min": "",
        "accelerometer_x_max": "",
        "accelerometer_y_mean": "",
        "accelerometer_y_min": "",
        "accelerometer_y_max": "",
        "accelerometer_z_mean": "",
        "accelerometer_z_min": "",
        "accelerometer_z_max": "",
        "gyroscope_x_mean": "",
        "gyroscope_x_min": "",
        "gyroscope_x_max": "",
        "gyroscope_y_mean": "",
        "gyroscope_y_min": "",
        "gyroscope_y_max": "",
        "gyroscope_z_mean": "",
        "gyroscope_z_min": "",
        "gyroscope_z_max": "",
    }


def summarize_timeseries_file(file_path):
    sensor_names = [
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    ]
    totals = {name: 0.0 for name in sensor_names}
    minimums = {name: None for name in sensor_names}
    maximums = {name: None for name in sensor_names}
    row_count = 0
    time_start = None
    time_end = None

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split(",")
            if len(values) != 7:
                continue

            time_value = float(values[0])
            if time_start is None:
                time_start = time_value
            time_end = time_value

            for index, sensor_name in enumerate(sensor_names, start=1):
                sensor_value = float(values[index])
                totals[sensor_name] += sensor_value
                current_min = minimums[sensor_name]
                current_max = maximums[sensor_name]
                minimums[sensor_name] = (
                    sensor_value if current_min is None else min(current_min, sensor_value)
                )
                maximums[sensor_name] = (
                    sensor_value if current_max is None else max(current_max, sensor_value)
                )

            row_count += 1

    if row_count == 0:
        return empty_sensor_stats()

    summary = {
        "row_count": row_count,
        "time_start": round(time_start, 6),
        "time_end": round(time_end, 6),
        "duration_seconds": round(time_end - time_start, 6),
    }

    for sensor_name in sensor_names:
        summary[f"{sensor_name}_mean"] = round(totals[sensor_name] / row_count, 6)
        summary[f"{sensor_name}_min"] = round(minimums[sensor_name], 6)
        summary[f"{sensor_name}_max"] = round(maximums[sensor_name], 6)

    return summary


def clean_timeseries(record_rows):
    timeseries_rows = []

    for record in record_rows:
        relative_file_name = record["file_name"]
        file_path = BASE_DIR / relative_file_name
        sensor_stats = summarize_timeseries_file(file_path)

        timeseries_rows.append(
            {
                "patient_id": record["patient_id"],
                "study_id": record["study_id"],
                "assessment_id": record["assessment_id"],
                "task_name": record["task_name"],
                "device_location": record["device_location"],
                "file_name": relative_file_name,
                **sensor_stats,
            }
        )

    fieldnames = [
        "patient_id",
        "study_id",
        "assessment_id",
        "task_name",
        "device_location",
        "file_name",
        "row_count",
        "time_start",
        "time_end",
        "duration_seconds",
        "accelerometer_x_mean",
        "accelerometer_x_min",
        "accelerometer_x_max",
        "accelerometer_y_mean",
        "accelerometer_y_min",
        "accelerometer_y_max",
        "accelerometer_z_mean",
        "accelerometer_z_min",
        "accelerometer_z_max",
        "gyroscope_x_mean",
        "gyroscope_x_min",
        "gyroscope_x_max",
        "gyroscope_y_mean",
        "gyroscope_y_min",
        "gyroscope_y_max",
        "gyroscope_z_mean",
        "gyroscope_z_min",
        "gyroscope_z_max",
    ]

    write_csv(OUTPUT_DIR / "timeseries_summary.csv", fieldnames, timeseries_rows)
    return timeseries_rows


def write_data_dictionary():
    rows = [
        {
            "table_name": "patients.csv",
            "grain": "One row per patient",
            "join_key": "patient_id",
            "description": "Patient-level demographics and clinical background.",
        },
        {
            "table_name": "questionnaire_answers.csv",
            "grain": "One row per patient per questionnaire question",
            "join_key": "patient_id",
            "description": "Long-format questionnaire answers for symptom analysis.",
        },
        {
            "table_name": "questionnaire_summary.csv",
            "grain": "One row per patient",
            "join_key": "patient_id",
            "description": "Questionnaire-level response totals and yes-answer rate.",
        },
        {
            "table_name": "movement_sessions.csv",
            "grain": "One row per patient per movement task",
            "join_key": "patient_id",
            "description": "Assessment session metadata including expected rows and device details.",
        },
        {
            "table_name": "movement_records.csv",
            "grain": "One row per patient per task per wrist file",
            "join_key": "patient_id",
            "description": "File manifest linking each task and wrist to the raw timeseries source.",
        },
        {
            "table_name": "timeseries_summary.csv",
            "grain": "One row per patient per task per wrist file",
            "join_key": "patient_id",
            "description": "Aggregated statistics from each raw sensor file for BI dashboards.",
        },
    ]

    fieldnames = ["table_name", "grain", "join_key", "description"]
    write_csv(OUTPUT_DIR / "data_dictionary.csv", fieldnames, rows)


def main():
    ensure_output_dir()
    patient_rows = clean_patients()
    questionnaire_rows, questionnaire_summary_rows = clean_questionnaires()
    movement_sessions, movement_records = clean_movement()
    timeseries_rows = clean_timeseries(movement_records)
    write_data_dictionary()

    print("Cleaning complete.")
    print(f"Patients: {len(patient_rows)}")
    print(f"Questionnaire answers: {len(questionnaire_rows)}")
    print(f"Questionnaire summary rows: {len(questionnaire_summary_rows)}")
    print(f"Movement sessions: {len(movement_sessions)}")
    print(f"Movement records: {len(movement_records)}")
    print(f"Timeseries summaries: {len(timeseries_rows)}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
