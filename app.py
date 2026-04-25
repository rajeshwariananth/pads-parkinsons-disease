import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

CSV_DIR = BASE_DIR / "pads_cleaned_csv"
RAW_DIR = BASE_DIR / "pads-parkinsons-disease-smartwatch-dataset"

sns.set_theme(style="whitegrid", context="talk")


def map_patient_group(condition: str) -> str:
    if condition == "Healthy":
        return "Healthy Control"
    if condition == "Parkinson's":
        return "Parkinson's Disease"
    return "Differential Diagnosis"


def classify_bmi(bmi: float) -> str:
    if pd.isna(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def classify_symptom_category(question_id: str, question_text: str) -> str:
    question_id = str(question_id).zfill(2)
    if question_id in {"22", "23", "24", "25", "26"}:
        return "Sleep"
    if question_id in {"01", "03", "04", "05", "06", "07", "08", "09"}:
        return "Gastrointestinal"
    if question_id in {"13", "16", "17", "18", "19"}:
        return "Mood"
    if question_id in {"12", "14", "15", "30"}:
        return "Cognitive"

    text = str(question_text).lower()
    if "sleep" in text or "dream" in text:
        return "Sleep"
    if any(word in text for word in ["bowel", "swallow", "constipation", "urine", "nausea"]):
        return "Gastrointestinal"
    if any(word in text for word in ["sad", "anxious", "interest", "panicky"]):
        return "Mood"
    if any(word in text for word in ["remember", "concentrating", "believing", "hearing"]):
        return "Cognitive"
    return "Other"


def average(values):
    clean = [value for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else np.nan


def figure_canvas(figsize=(8, 4.8)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig, ax


@st.cache_data(show_spinner=False)
def load_data():
    patients = pd.read_csv(CSV_DIR / "patients.csv", dtype={"patient_id": str})
    questionnaire_answers = pd.read_csv(CSV_DIR / "questionnaire_answers.csv", dtype={"patient_id": str, "question_id": str})
    questionnaire_summary = pd.read_csv(CSV_DIR / "questionnaire_summary.csv", dtype={"patient_id": str})
    movement_sessions = pd.read_csv(CSV_DIR / "movement_sessions.csv", dtype={"patient_id": str})
    movement_records = pd.read_csv(CSV_DIR / "movement_records.csv", dtype={"patient_id": str})
    timeseries_summary = pd.read_csv(CSV_DIR / "timeseries_summary.csv", dtype={"patient_id": str})

    patients["patient_group"] = patients["condition"].map(map_patient_group)
    patients["bmi"] = patients["weight_kg"] / ((patients["height_cm"] / 100) ** 2)
    patients["bmi_category"] = patients["bmi"].apply(classify_bmi)
    patients["disease_duration"] = patients["current_age"] - patients["age_at_diagnosis"]
    patients["gender"] = patients["gender"].str.title()

    questionnaire_answers["answer_flag"] = questionnaire_answers["answer_flag"].astype(int)
    questionnaire_answers["symptom_category"] = questionnaire_answers.apply(
        lambda row: classify_symptom_category(row["question_id"], row["question_text"]),
        axis=1,
    )
    questionnaire_joined = questionnaire_answers.merge(
        questionnaire_summary[["patient_id", "yes_answers", "yes_answer_rate"]],
        on="patient_id",
        how="left",
    ).merge(
        patients[["patient_id", "patient_group", "condition", "current_age", "gender"]],
        on="patient_id",
        how="left",
    )

    sensor = timeseries_summary.merge(
        movement_records,
        on=["patient_id", "study_id", "assessment_id", "task_name", "device_location", "file_name"],
        how="left",
        suffixes=("", "_record"),
    ).merge(
        movement_sessions[["patient_id", "task_name", "device_id", "sampling_rate_hz", "expected_rows_per_record"]],
        on=["patient_id", "task_name"],
        how="left",
    ).merge(
        patients[["patient_id", "patient_group", "condition", "current_age", "bmi"]],
        on="patient_id",
        how="left",
    )

    for axis in ["x", "y", "z"]:
        sensor[f"accelerometer_{axis}_range"] = sensor[f"accelerometer_{axis}_max"] - sensor[f"accelerometer_{axis}_min"]
        sensor[f"gyroscope_{axis}_range"] = sensor[f"gyroscope_{axis}_max"] - sensor[f"gyroscope_{axis}_min"]

    sensor["acceleration_range_mean"] = sensor[
        ["accelerometer_x_range", "accelerometer_y_range", "accelerometer_z_range"]
    ].mean(axis=1)
    sensor["gyroscope_range_mean"] = sensor[
        ["gyroscope_x_range", "gyroscope_y_range", "gyroscope_z_range"]
    ].mean(axis=1)
    sensor["movement_intensity"] = np.sqrt(
        sensor["accelerometer_x_range"] ** 2
        + sensor["accelerometer_y_range"] ** 2
        + sensor["accelerometer_z_range"] ** 2
    )
    sensor["tremor_intensity"] = np.sqrt(
        sensor["gyroscope_x_range"] ** 2
        + sensor["gyroscope_y_range"] ** 2
        + sensor["gyroscope_z_range"] ** 2
    )
    sensor["movement_variance"] = sensor[
        ["accelerometer_x_range", "accelerometer_y_range", "accelerometer_z_range"]
    ].pow(2).mean(axis=1)
    sensor["motion_energy"] = sensor["movement_intensity"] + 0.1 * sensor["tremor_intensity"]
    sensor["tremor_frequency_proxy"] = sensor["tremor_intensity"] / sensor["duration_seconds"].replace(0, np.nan)
    sensor["movement_speed"] = sensor["movement_intensity"] / sensor["duration_seconds"].replace(0, np.nan)

    patient_sensor = (
        sensor.groupby("patient_id", as_index=False)
        .agg(
            tremor_amplitude=("tremor_intensity", "mean"),
            tremor_frequency_proxy=("tremor_frequency_proxy", "mean"),
            movement_variance=("movement_variance", "mean"),
            movement_speed=("movement_speed", "mean"),
        )
        .merge(
            patients[["patient_id", "patient_group", "condition", "current_age", "bmi"]],
            on="patient_id",
            how="left",
        )
    )

    symptom_totals = (
        questionnaire_joined[questionnaire_joined["answer_flag"] == 1]
        .groupby("patient_id", as_index=False)
        .agg(symptom_count=("answer_flag", "sum"))
    )
    patient_analysis = patient_sensor.merge(symptom_totals, on="patient_id", how="left")
    patient_analysis["symptom_count"] = patient_analysis["symptom_count"].fillna(0)

    return {
        "patients": patients,
        "questionnaire_joined": questionnaire_joined,
        "movement_sessions": movement_sessions.merge(
            patients[["patient_id", "patient_group", "condition"]], on="patient_id", how="left"
        ),
        "movement_records": movement_records,
        "sensor": sensor,
        "patient_analysis": patient_analysis,
    }


@st.cache_data(show_spinner=False)
def load_raw_timeseries(relative_file: str) -> pd.DataFrame:
    raw = pd.read_csv(
        RAW_DIR / relative_file,
        header=None,
        names=[
            "time",
            "accelerometer_x",
            "accelerometer_y",
            "accelerometer_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ],
    )
    raw["acceleration_magnitude"] = np.sqrt(
        raw["accelerometer_x"] ** 2 + raw["accelerometer_y"] ** 2 + raw["accelerometer_z"] ** 2
    )
    raw["gyroscope_magnitude"] = np.sqrt(
        raw["gyroscope_x"] ** 2 + raw["gyroscope_y"] ** 2 + raw["gyroscope_z"] ** 2
    )
    return raw


def draw_simple_treemap(ax, values_df: pd.DataFrame, label_col: str, value_col: str):
    total = values_df[value_col].sum()
    x_start = 0.0
    palette = sns.color_palette("flare", len(values_df))
    for idx, row in enumerate(values_df.itertuples()):
        width = row.__getattribute__(value_col) / total if total else 0
        rect = patches.Rectangle((x_start, 0), width, 1, facecolor=palette[idx], edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(
            x_start + width / 2,
            0.5,
            f"{row.__getattribute__(label_col)}\n{row.__getattribute__(value_col)}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
        )
        x_start += width
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def show_kpi(col, title, value, help_text):
    col.metric(title, value)
    col.caption(help_text)


def section_demographics(patients: pd.DataFrame):
    parkinsons_count = (patients["condition"] == "Parkinson's").sum()
    healthy_count = (patients["condition"] == "Healthy").sum()
    cols = st.columns(5)
    show_kpi(cols[0], "Total Patients", f"{len(patients):,}", "All rows in patients.csv")
    show_kpi(cols[1], "Parkinson's Patients", f"{parkinsons_count:,}", "Exact Parkinson's condition")
    show_kpi(cols[2], "Healthy Patients", f"{healthy_count:,}", "Healthy controls")
    show_kpi(cols[3], "Average Age at Diagnosis", f"{patients['age_at_diagnosis'].mean():.1f}", "Mean recorded age at diagnosis")
    pd_duration = patients.loc[patients["condition"] != "Healthy", "disease_duration"].dropna()
    show_kpi(cols[4], "Ave Disease Duration", f"{pd_duration.mean():.1f}", "Current age minus age at diagnosis")

    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas()
      age_bins = pd.cut(patients["current_age"], bins=np.arange(20, 101, 10), right=False)
      age_counts = age_bins.value_counts().sort_index().rename_axis("age_group").reset_index(name="count")
      age_counts["age_group"] = age_counts["age_group"].astype(str)
      sns.barplot(data=age_counts, x="age_group", y="count", color="#b85c38", ax=ax)
      ax.set_title("Age Distribution")
      ax.set_xlabel("Age Group")
      ax.set_ylabel("Patients")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas()
      gender_counts = patients["gender"].value_counts()
      ax.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
      ax.set_title("Gender Distribution")
      st.pyplot(fig, use_container_width=True)

    c3, c4, c5 = st.columns(3)
    with c3:
      fig, ax = figure_canvas()
      bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
      bmi_counts = patients["bmi_category"].value_counts().reindex(bmi_order).fillna(0)
      ax.pie(bmi_counts.values, labels=bmi_counts.index, autopct="%1.1f%%", wedgeprops={"width": 0.45}, colors=sns.color_palette("crest", 4))
      ax.set_title("BMI Category Distribution")
      st.pyplot(fig, use_container_width=True)
    with c4:
      fig, ax = figure_canvas()
      sns.scatterplot(
          data=patients,
          x="height_cm",
          y="weight_kg",
          hue="patient_group",
          palette="Set1",
          alpha=0.8,
          ax=ax,
      )
      ax.set_title("Height vs Weight")
      ax.set_xlabel("Height (cm)")
      ax.set_ylabel("Weight (kg)")
      st.pyplot(fig, use_container_width=True)
    with c5:
      fig, ax = figure_canvas()
      group_counts = patients["patient_group"].value_counts().reindex(
          ["Parkinson's Disease", "Differential Diagnosis", "Healthy Control"]
      )
      sns.barplot(x=group_counts.index, y=group_counts.values, palette="Set2", ax=ax)
      ax.set_title("Patient Group Distribution")
      ax.set_xlabel("")
      ax.set_ylabel("Patients")
      plt.xticks(rotation=20)
      st.pyplot(fig, use_container_width=True)


def section_clinical(questionnaire_joined: pd.DataFrame):
    positive = questionnaire_joined[questionnaire_joined["answer_flag"] == 1].copy()
    symptom_counts = (
        positive.groupby("question_text", as_index=False)
        .agg(yes_count=("answer_flag", "sum"))
        .sort_values("yes_count", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas(figsize=(8, 8))
      sns.barplot(data=symptom_counts, x="yes_count", y="question_text", color="#1f4e5f", ax=ax)
      ax.set_title("Non-Motor Symptoms Frequency")
      ax.set_xlabel("Yes Responses")
      ax.set_ylabel("")
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas(figsize=(8, 4))
      category_counts = (
          positive.groupby("symptom_category", as_index=False)
          .agg(yes_count=("answer_flag", "sum"))
          .sort_values("yes_count", ascending=False)
      )
      draw_simple_treemap(ax, category_counts, "symptom_category", "yes_count")
      ax.set_title("Symptom Category Distribution")
      st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
      fig, ax = figure_canvas()
      by_group = (
          positive.groupby(["patient_group", "symptom_category"], as_index=False)
          .agg(yes_count=("answer_flag", "sum"))
      )
      pivot = by_group.pivot(index="patient_group", columns="symptom_category", values="yes_count").fillna(0)
      pivot.loc[["Parkinson's Disease", "Differential Diagnosis", "Healthy Control"]].plot(
          kind="bar", stacked=True, ax=ax, colormap="flare"
      )
      ax.set_title("Symptoms by Patient Group")
      ax.set_xlabel("")
      ax.set_ylabel("Yes Responses")
      ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
      st.pyplot(fig, use_container_width=True)
    with c4:
      fig, ax = figure_canvas(figsize=(8, 6))
      top_10 = symptom_counts.head(10).sort_values("yes_count")
      sns.barplot(data=top_10, x="yes_count", y="question_text", color="#b85c38", ax=ax)
      ax.set_title("Top 10 Most Common Symptoms")
      ax.set_xlabel("Yes Responses")
      ax.set_ylabel("")
      st.pyplot(fig, use_container_width=True)

    fig, ax = figure_canvas(figsize=(10, 8))
    heat = (
        positive.groupby(["question_text", "patient_group"], as_index=False)
        .agg(yes_count=("answer_flag", "sum"))
        .pivot(index="question_text", columns="patient_group", values="yes_count")
        .fillna(0)
    )
    heat = heat.reindex(columns=["Parkinson's Disease", "Differential Diagnosis", "Healthy Control"])
    sns.heatmap(heat, cmap="YlOrRd", ax=ax)
    ax.set_title("Symptom Heatmap")
    ax.set_xlabel("Patient Group")
    ax.set_ylabel("Symptoms")
    st.pyplot(fig, use_container_width=True)


def section_movement(movement_sessions: pd.DataFrame, sensor: pd.DataFrame):
    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas()
      task_counts = sensor["task_name"].value_counts().rename_axis("task_name").reset_index(name="recordings")
      sns.barplot(data=task_counts, x="task_name", y="recordings", color="#2a9d8f", ax=ax)
      ax.set_title("Task Frequency")
      ax.set_xlabel("")
      ax.set_ylabel("Recordings")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas()
      accel = (
          sensor.groupby(["task_name", "device_location"], as_index=False)
          .agg(acceleration=("acceleration_range_mean", "mean"))
      )
      sns.barplot(data=accel, x="task_name", y="acceleration", hue="device_location", palette="Set2", ax=ax)
      ax.set_title("Average Acceleration by Task")
      ax.set_xlabel("")
      ax.set_ylabel("Average Acceleration Range")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
      fig, ax = figure_canvas(figsize=(10, 5))
      sns.boxplot(data=sensor, x="task_name", y="movement_intensity", color="#ddb15d", ax=ax)
      ax.set_title("Movement Intensity per Task")
      ax.set_xlabel("")
      ax.set_ylabel("Movement Intensity")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)
    with c4:
      fig, ax = figure_canvas()
      duration = sensor.groupby("task_name", as_index=False).agg(duration=("duration_seconds", "mean"))
      sns.barplot(data=duration, x="task_name", y="duration", color="#1f4e5f", ax=ax)
      ax.set_title("Task Duration Comparison")
      ax.set_xlabel("")
      ax.set_ylabel("Average Duration (s)")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)

    fig, ax = figure_canvas(figsize=(10, 5))
    task_vs_group = (
        movement_sessions.groupby(["task_name", "patient_group"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    pivot = task_vs_group.pivot(index="task_name", columns="patient_group", values="count").fillna(0)
    pivot = pivot[["Parkinson's Disease", "Differential Diagnosis", "Healthy Control"]]
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Task vs Patient Group")
    ax.set_xlabel("Task")
    ax.set_ylabel("Session Rows")
    plt.xticks(rotation=30)
    st.pyplot(fig, use_container_width=True)


def section_sensor(sensor: pd.DataFrame):
    patients = sorted(sensor["patient_id"].unique().tolist())
    tasks = sorted(sensor["task_name"].unique().tolist())
    wrists = sorted(sensor["device_location"].unique().tolist())

    sel1, sel2, sel3 = st.columns(3)
    patient_id = sel1.selectbox("Patient", patients, index=0)
    task_name = sel2.selectbox("Task", tasks, index=tasks.index("Relaxed") if "Relaxed" in tasks else 0)
    wrist = sel3.selectbox("Wrist", wrists, index=wrists.index("RightWrist") if "RightWrist" in wrists else 0)

    selected_row = sensor[
        (sensor["patient_id"] == patient_id)
        & (sensor["task_name"] == task_name)
        & (sensor["device_location"] == wrist)
    ].iloc[0]
    raw = load_raw_timeseries(selected_row["file_name"]).iloc[::5].copy()

    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas()
      sns.lineplot(data=raw, x="time", y="acceleration_magnitude", color="#b85c38", linewidth=1.5, ax=ax)
      ax.set_title("Accelerometer Trend Over Time")
      ax.set_xlabel("Time (s)")
      ax.set_ylabel("Acceleration Magnitude")
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas()
      sns.lineplot(data=raw, x="time", y="gyroscope_magnitude", color="#1f4e5f", linewidth=1.5, ax=ax)
      ax.set_title("Gyroscope Rotation Over Time")
      ax.set_xlabel("Time (s)")
      ax.set_ylabel("Gyroscope Magnitude")
      st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
      fig, ax = figure_canvas()
      xyz = raw.melt(
          id_vars="time",
          value_vars=["accelerometer_x", "accelerometer_y", "accelerometer_z"],
          var_name="axis",
          value_name="value",
      )
      sns.lineplot(data=xyz, x="time", y="value", hue="axis", ax=ax)
      ax.set_title("X vs Y vs Z Acceleration")
      ax.set_xlabel("Time (s)")
      ax.set_ylabel("Acceleration")
      st.pyplot(fig, use_container_width=True)
    with c4:
      fig, ax = figure_canvas()
      variance_task = sensor.groupby("task_name", as_index=False).agg(signal_variance=("movement_variance", "mean"))
      sns.barplot(data=variance_task, x="task_name", y="signal_variance", color="#d96c6c", ax=ax)
      ax.set_title("Signal Variance by Task")
      ax.set_xlabel("")
      ax.set_ylabel("Variance Proxy")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)

    fig, ax = figure_canvas()
    sns.histplot(sensor["motion_energy"], bins=30, color="#2a9d8f", ax=ax)
    ax.set_title("Motion Energy Distribution")
    ax.set_xlabel("Motion Energy Proxy")
    ax.set_ylabel("Recordings")
    st.pyplot(fig, use_container_width=True)


def section_tremor(sensor: pd.DataFrame, patient_analysis: pd.DataFrame):
    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas()
      sns.boxplot(data=patient_analysis, x="patient_group", y="tremor_amplitude", palette="Set2", ax=ax)
      ax.set_title("Tremor Intensity by Patient Group")
      ax.set_xlabel("")
      ax.set_ylabel("Tremor Intensity")
      plt.xticks(rotation=20)
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas()
      sns.histplot(patient_analysis["tremor_frequency_proxy"].dropna(), bins=30, color="#ddb15d", ax=ax)
      ax.set_title("Tremor Frequency Distribution")
      ax.set_xlabel("Tremor Frequency Proxy")
      ax.set_ylabel("Patients")
      st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
      fig, ax = figure_canvas()
      sns.scatterplot(data=patient_analysis, x="current_age", y="tremor_amplitude", hue="patient_group", palette="Set1", ax=ax)
      ax.set_title("Tremor vs Age")
      ax.set_xlabel("Age")
      ax.set_ylabel("Tremor Intensity")
      st.pyplot(fig, use_container_width=True)
    with c4:
      fig, ax = figure_canvas()
      tremor_task = sensor.groupby("task_name", as_index=False).agg(tremor_intensity=("tremor_intensity", "mean"))
      sns.barplot(data=tremor_task, x="task_name", y="tremor_intensity", color="#b85c38", ax=ax)
      ax.set_title("Tremor by Task Type")
      ax.set_xlabel("")
      ax.set_ylabel("Average Tremor Intensity")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)

    fig, ax = figure_canvas()
    bins = patient_analysis["tremor_amplitude"].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    bins = np.unique(bins)
    labels = ["Low", "Moderate", "High", "Very High"][: max(len(bins) - 1, 1)]
    severity = pd.cut(patient_analysis["tremor_amplitude"], bins=bins, labels=labels, include_lowest=True)
    severity_counts = severity.value_counts().reindex(labels).fillna(0)
    ax.pie(
        severity_counts.values,
        labels=severity_counts.index,
        autopct="%1.1f%%",
        wedgeprops={"width": 0.45},
        colors=sns.color_palette("flare", len(severity_counts)),
    )
    ax.set_title("Tremor Severity Categories")
    st.pyplot(fig, use_container_width=True)


def section_wrist(sensor: pd.DataFrame):
    paired = sensor.pivot_table(
        index=["patient_id", "task_name", "patient_group"],
        columns="device_location",
        values=["acceleration_range_mean", "motion_energy", "tremor_intensity"],
    )
    paired.columns = [f"{metric}_{wrist}" for metric, wrist in paired.columns]
    paired = paired.reset_index().dropna()

    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas()
      sns.scatterplot(
          data=paired,
          x="acceleration_range_mean_LeftWrist",
          y="acceleration_range_mean_RightWrist",
          hue="patient_group",
          palette="Set1",
          alpha=0.7,
          ax=ax,
      )
      ax.set_title("Left vs Right Acceleration")
      ax.set_xlabel("Left Wrist Acceleration")
      ax.set_ylabel("Right Wrist Acceleration")
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas()
      wrist_energy = sensor.groupby(["task_name", "device_location"], as_index=False).agg(motion_energy=("motion_energy", "mean"))
      sns.barplot(data=wrist_energy, x="task_name", y="motion_energy", hue="device_location", palette="Set2", ax=ax)
      ax.set_title("Average Movement by Wrist")
      ax.set_xlabel("")
      ax.set_ylabel("Motion Energy Proxy")
      plt.xticks(rotation=30)
      st.pyplot(fig, use_container_width=True)

    fig, ax = figure_canvas(figsize=(10, 5))
    paired["tremor_asymmetry"] = paired["tremor_intensity_RightWrist"] - paired["tremor_intensity_LeftWrist"]
    asymmetry = paired.groupby("task_name", as_index=False).agg(tremor_asymmetry=("tremor_asymmetry", "mean"))
    asymmetry = asymmetry.sort_values("tremor_asymmetry")
    sns.barplot(data=asymmetry, x="tremor_asymmetry", y="task_name", palette="coolwarm", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Tremor Asymmetry")
    ax.set_xlabel("Right Minus Left Tremor Intensity")
    ax.set_ylabel("Task")
    st.pyplot(fig, use_container_width=True)


def section_correlation(patient_analysis: pd.DataFrame):
    corr_df = patient_analysis[["current_age", "bmi", "tremor_amplitude", "movement_variance"]].rename(
        columns={
            "current_age": "Age",
            "bmi": "BMI",
            "tremor_amplitude": "Tremor amplitude",
            "movement_variance": "Movement variance",
        }
    )

    c1, c2 = st.columns(2)
    with c1:
      fig, ax = figure_canvas()
      sns.heatmap(corr_df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
      ax.set_title("Correlation Matrix")
      st.pyplot(fig, use_container_width=True)
    with c2:
      fig, ax = figure_canvas()
      sns.scatterplot(data=patient_analysis, x="bmi", y="movement_speed", hue="patient_group", palette="Set1", ax=ax)
      ax.set_title("BMI vs Movement Speed")
      ax.set_xlabel("BMI")
      ax.set_ylabel("Movement Speed Proxy")
      st.pyplot(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="PADS Python Dashboard", layout="wide")
    st.title("PADS Smartwatch Dataset Dashboard")
    st.caption("Python dashboard using pandas for joins, seaborn for analysis visuals, and Streamlit for the interactive layout.")

    data = load_data()
    patients = data["patients"]
    questionnaire_joined = data["questionnaire_joined"]
    movement_sessions = data["movement_sessions"]
    sensor = data["sensor"]
    patient_analysis = data["patient_analysis"]

    st.info(
        "This app uses `patients.csv` as the main dimension table, joins questionnaire tables on `patient_id`, and joins movement and sensor tables on patient/task/wrist metadata."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "TAB1 Demographics",
            "TAB2 Clinical Symptoms Analysis",
            "TAB3 Movement Task Analysis",
            "TAB4 Sensor Data Analysis",
            "TAB5 Tremor Analysis",
            "TAB6 Left vs Right Wrist Comparison",
            "TAB7 Correlation & Pattern Analysis",
        ]
    )

    with tab1:
        section_demographics(patients)
    with tab2:
        section_clinical(questionnaire_joined)
    with tab3:
        section_movement(movement_sessions, sensor)
    with tab4:
        section_sensor(sensor)
    with tab5:
        section_tremor(sensor, patient_analysis)
    with tab6:
        section_wrist(sensor)
    with tab7:
        section_correlation(patient_analysis)


if __name__ == "__main__":
    main()
