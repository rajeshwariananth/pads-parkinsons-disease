# PADS — Parkinson's Disease Smartwatch Dataset

A comprehensive dataset of smartwatch sensor recordings from Parkinson's disease patients and healthy controls, with accompanying clinical questionnaires and demographic data.

## Project Structure

```
pads-parkinsons-disease/
├── app.py                      # Streamlit dashboard for data exploration
├── index.html                  # Standalone HTML dashboard with D3/Plotly
├── clean_pads_dataset.py       # Data cleaning pipeline (JSON → CSV)
├── requirements.txt           # Python dependencies
├── pads_cleaned_csv/           # Cleaned CSV outputs
│   ├── data_dictionary.csv     # Schema documentation
│   ├── patients.csv            # Patient demographics
│   ├── questionnaire_answers.csv
│   ├── questionnaire_summary.csv
│   ├── movement_sessions.csv
│   ├── movement_records.csv
│   └── timeseries_summary.csv
└── pads-parkinsons-disease-smartwatch-dataset/
    ├── patients/               # Raw patient JSON files
    ├── questionnaire/           # Questionnaire responses
    ├── movement/               # Movement task metadata
    └── timeseries/             # Raw sensor timeseries data
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 3. Explore the HTML Dashboard

Open `index.html` in a browser — no server required.

### 4. Re-run Data Cleaning (Optional)

```bash
python clean_pads_dataset.py
```

Regenerates the cleaned CSVs from raw JSON files.

## Dataset Overview

| File | Description |
|------|-------------|
| `patients.csv` | Demographics, clinical background, disease status |
| `questionnaire_answers.csv` | Long-format symptom questionnaire responses |
| `questionnaire_summary.csv` | Aggregated questionnaire totals and yes-rates |
| `movement_sessions.csv` | Movement task session metadata |
| `movement_records.csv` | File manifest linking tasks to raw timeseries |
| `timeseries_summary.csv` | Aggregated sensor statistics per recording |

### Patient Groups

- **Healthy Control** — No Parkinson's diagnosis
- **Parkinson's Disease** — Confirmed PD diagnosis
- **Differential Diagnosis** — Pending or uncertain diagnosis

## Technologies

- **Python**: Data processing (pandas, numpy)
- **Streamlit**: Interactive web dashboard
- **D3.js + Plotly**: Standalone HTML visualization
- **Seaborn/Matplotlib**: Static plotting

## License

This dataset is provided for research purposes. Please cite the original source if used in publications.
