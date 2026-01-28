import pandas as pd
import os

# ======================
# CONFIG
# ======================
CSV_PATH = "sint1.csv"

# ======================
# LOAD CSV
# ======================
df = pd.read_csv(CSV_PATH)

# Renombrar primera columna (ruta MIDI)
df = df.rename(columns={df.columns[0]: "midi_path"})

# ======================
# EXTRAER NOMBRE DEL DATASET
# ======================
def extract_dataset(path):
    """
    Extrae el nombre del dataset desde la ruta o el nombre del archivo.
    - 'midi_generados/<DATASET>' → toma <DATASET>
    - 'RVAE/...mid' → "RVAE"
    - archivos tipo '1OG.mid', '2REAM.mid', '3REARM.mid' → OG o REAM
    """
    parts = os.path.normpath(path).split(os.sep)
    
    # Caso midi_generados
    if "midi_generados" in parts:
        idx = parts.index("midi_generados")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    
    # Caso RVAE
    if "RVAE" in parts:
        return "RVAE"
    
    # Archivos sueltos: mirar el nombre
    fname = os.path.basename(path).upper()
    if "OG" in fname:
        return "OG"
    if "REAM" in fname:
        return "REAM"
    
    # Por defecto: carpeta padre
    return os.path.basename(os.path.dirname(path))

df["dataset"] = df["midi_path"].apply(extract_dataset)

# ======================
# METRICS A ANALYZE
# ======================
metrics= {
    # Dissonance
    "Vertical Minor Seconds": "Vertical_Minor_Seconds",
    "Vertical Tritones": "Vertical_Tritones",
    "Vertical Sevenths": "Vertical_Sevenths",
    "Vertical Dissonance Ratio": "Vertical_Dissonance_Ratio",

    # Harmonic richness
    "Standard Triads": "Standard_Triads",
    "Dominant Seventh Chords": "Dominant_Seventh_Chords",
    "Seventh Chords": "Seventh_Chords",
    "Non-Standard Chords": "Non-Standard_Chords",
    "Complex Chords": "Complex_Chords",

    # Unpredictability / evolution
    "Distance Between Two Most Common Vertical Intervals": "Distance_Between_Two_Most_Common_Vertical_Intervals",
    "Prevalence Ratio of Two Most Common Vertical Intervals": "Prevalence_Ratio_of_Two_Most_Common_Vertical_Intervals",
    "Variability of Number of Simultaneous Pitch Classes": "Variability_of_Number_of_Simultaneous_Pitch_Classes"
}


# ======================
# COMPUTE STATS
# ======================
def summarize_dataset(df_subset):
    rows = []
    for label, col in metrics.items():
        if col not in df_subset.columns:
            continue
        mean = df_subset[col].mean()
        std = df_subset[col].std(ddof=0)  # poblacional
        rows.append({
            "Metric": label,
            "Mean": mean,
            "Std": std
        })
    return pd.DataFrame(rows)

# ======================
# PROCESS EACH DATASET
# ======================
for dataset_name, df_group in df.groupby("dataset"):
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    summary = summarize_dataset(df_group)

    for _, row in summary.iterrows():
        print(
            f"{row['Metric']:28s} "
            f"{row['Mean']:.4f} ± {row['Std']:.4f}"
        )
