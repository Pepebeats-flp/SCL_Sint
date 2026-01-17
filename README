# CVAE-LSTM Chord Progression Enrichment (Inference)

This repository provides the **inference pipeline** for a **CVAE-LSTM (Conditional Variational Autoencoder with LSTM)** model designed to enrich chord progressions extracted from MIDI files.

Given a MIDI input, the system:
- Extracts chord progressions
- Encodes them as discrete tokens
- Samples multiple candidates from the latent space
- Explicitly rejects any generation that is identical to the input progression
- Evaluates tonal coherence
- Selects the best generated progression

**Note:** This repository supports **inference only**. Model training is not included.

---

## Directory Structure

```
.
├── main.py
├── cvae_lstm_model.pth
├── token_to_id_train.pkl
├── id_to_token_train.pkl
├── environment.yml
├── README.md
└── instances/
    └── 1.mid
```

---
## Requirements

- Conda (Miniconda or Anaconda)
- Python 3.10

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate cvae
```

The environment is intentionally minimal and installs only the dependencies required for inference, ensuring fast setup and reproducibility.

---

## How to Run

The main script is `main.py`. It processes MIDI files located in the `instances/` directory. The command-line arguments are as follows:

```bash
python main.py <midi_input> <seed> <sigma> <N>
```

Where:
- `<midi_input>`: Path to the input MIDI file (e.g., `1.mid`).
- `<seed>`: Random seed for reproducibility.
- `<sigma>`: Latent noise scaling factor.
- `<N>`: Number of generated candidates to evaluate.

### Example Command
```bash
python main.py 1.mid 42 0.8 20
```

---

## Output

The script prints the following to standard output:
```
Input tokens: ['0_4_7', '5_9_0', '7_11_2']
Best tokens: ['0_4_7_11', '5_9_0', '7_11_2_5']
0.0833
```

Where:
- `Input tokens`: The original chord progression tokens extracted from the MIDI file.
- `Best tokens`: The enriched chord progression tokens selected based on tonal coherence.
- The final line is the tonal coherence loss score (0 to 1 scale), with lower values indicating better coherence with the input.

No files are written to disk.

---

## Musical Evaluation

To select the best generated progression, the system:

- Automatically detects the tonal scale using `music21`.
- Penalizes notes outside the detected scale.
- Selects the candidate with minimal tonal coherence loss.

The loss is defined as:

$$ 
\text{loss} = 1 - \left(\frac{\text{notes in scale}}{\text{total notes}}\right)

$$
---
