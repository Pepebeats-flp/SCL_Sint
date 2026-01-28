import pandas as pd
import numpy as np

# =========================
# Datos originales
# =========================
data = {
    "Métrica": [
        "Vertical Minor Seconds", "Vertical Tritones", "Vertical Sevenths", "Vertical Dissonance Ratio",
        "Standard Triads", "Dominant Seventh Chords", "Seventh Chords", "Non-Standard Chords",
        "Complex Chords", "Distance Between Two Most Common Vertical Intervals",
        "Prevalence Ratio of Two Most Common Vertical Intervals", "Variability of Number of Simultaneous Pitch Classes"
    ],
    "OG": [0.0, 0.0062, 0.0062, 0.0133, 0.9792, 0.0208, 0.0208, 0.0, 0.0, 1.0, 0.9815, 0.0551],
    "REAM": [0.0, 0.0424, 0.1240, 0.3140, 0.4375, 0.0417, 0.1667, 0.3958, 0.25, 2.5, 0.8577, 0.2913],
    "RVAE": [0.0, 0.0728, 0.0, 0.1777, 0.6667, 0.3, 0.3, 0.0, 0.0, 1.3333, 0.7917, 0.46],
    "SCL_Default": [0.0556, 0.0684, 0.0043, 0.2917, 0.6806, 0.0278, 0.0278, 0.2917, 0.0278, 1.1667, 0.6296, 0.5278],
    "SCL_Sigma_2.83": [0.0, 0.0123, 0.0123, 0.1167, 0.8333, 0.0, 0.0, 0.1667, 0.0833, 1.0, 0.6833, 0.4083]
}

# =========================
data = {
    "Métrica": [
        "Vertical Minor Seconds",
        "Vertical Tritones",
        "Vertical Sevenths",
        "Vertical Dissonance Ratio",
        "Standard Triads",
        "Dominant Seventh Chords",
        "Seventh Chords",
        "Non-Standard Chords",
        "Complex Chords",
        "Distance Between Two Most Common Vertical Intervals",
        "Prevalence Ratio of Two Most Common Vertical Intervals",
        "Variability of Number of Simultaneous Pitch Classes"
    ],

    "SCL_Default": [
        0.0556,
        0.0684,
        0.0043,
        0.2917,
        0.6806,
        0.0278,
        0.0278,
        0.2917,
        0.0278,
        1.1667,
        0.6296,
        0.5278
    ],

    "SCL_Sigma_2.54": [
        0.0222,
        0.0284,
        0.0062,
        0.1417,
        0.8333,
        0.0,
        0.0,
        0.1667,
        0.0417,
        1.0,
        0.6667,
        0.3486
    ],

    "SCL_Sigma_2.81": [
        0.0,
        0.0123,
        0.0123,
        0.1167,
        0.8333,
        0.0,
        0.0,
        0.1667,
        0.0833,
        1.0,
        0.6833,
        0.4083
    ],

    "SCL_Sigma_2.82": [
        0.0,
        0.0123,
        0.0123,
        0.1167,
        0.8333,
        0.0,
        0.0,
        0.1667,
        0.0833,
        1.0,
        0.6833,
        0.4083
    ],

    "SCL_Sigma_2.83": [
        0.0,
        0.0123,
        0.0123,
        0.1167,
        0.8333,
        0.0,
        0.0,
        0.1667,
        0.0833,
        1.0,
        0.6833,
        0.4083
    ]
}
df = pd.DataFrame(data)
df.set_index("Métrica", inplace=True)

# =========================
# Bloques de métricas
# =========================
bloques = {
    "dissonance": [
        "Vertical Minor Seconds", "Vertical Tritones", "Vertical Sevenths", "Vertical Dissonance Ratio"
    ],
    "chord_diversity": [
        "Standard Triads", "Dominant Seventh Chords", "Seventh Chords", "Non-Standard Chords", "Complex Chords"
    ],
    "distribution": [
        "Distance Between Two Most Common Vertical Intervals",
        "Prevalence Ratio of Two Most Common Vertical Intervals",
        "Variability of Number of Simultaneous Pitch Classes"
    ]
}

# =========================
# Invertir métricas donde mayor valor significa menos complejidad
# =========================
invertir = ["Standard Triads", "Prevalence Ratio of Two Most Common Vertical Intervals"]
for metric in invertir:
    df.loc[metric] = 1 - df.loc[metric]

# =========================
# Normalización min-max por métrica
# =========================
df_norm = (df - df.min(axis=1).values.reshape(-1,1)) / (df.max(axis=1).values.reshape(-1,1) - df.min(axis=1).values.reshape(-1,1) + 1e-8)

# =========================
# Función para calcular puntuación agregada
# =========================
def puntuacion_agregada(df_norm, bloques, pesos):
    scores = {}
    for col in df_norm.columns:
        score = 0
        for bloque_name, metrics in bloques.items():
            bloque_score = df_norm.loc[metrics, col].mean()
            score += bloque_score * pesos[bloque_name]
        scores[col] = score
    return scores

# =========================
# Varias configuraciones de pesos para probar
# =========================
configuraciones = {
    "Más disonancia": {"dissonance":0.5, "chord_diversity":0.3, "distribution":0.2},
    "Más diversidad acordes": {"dissonance":0.2, "chord_diversity":0.5, "distribution":0.3},
    "Más distribución": {"dissonance":0.2, "chord_diversity":0.3, "distribution":0.5},
    "Equilibrado": {"dissonance":0.33, "chord_diversity":0.33, "distribution":0.34},
    "Muy enfocado en disonancia": {"dissonance":0.7, "chord_diversity":0.15, "distribution":0.15},
    "Muy enfocado en diversidad acordes": {"dissonance":0.15, "chord_diversity":0.7, "distribution":0.15},
    "Muy enfocado en distribución": {"dissonance":0.15, "chord_diversity":0.15, "distribution":0.7}
}

# =========================
# Calcular y mostrar resultados
# =========================
for nombre, pesos in configuraciones.items():
    scores = puntuacion_agregada(df_norm, bloques, pesos)
    print(f"\n--- Configuración: {nombre} ---")
    for modelo, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{modelo}: {score:.3f}")
