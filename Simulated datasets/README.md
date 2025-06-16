# Synthetic Datasets Documentation

This repository contains artificially generated datasets for **Circadian Oscillation** and **Differential Rhythmicity** tasks. All sample counts are measured in **thousands (K)** (1K = 1,000 samples).

---

## Dataset Overview

### 1. Circadian Oscillation Task

| Dataset     | Duration | Interval   | Training (K) | Validation (K) | Test (K) |
|:-----------:|:--------:|:----------:|:------------:|:--------------:|:--------:|
| SynthDST-1  | 24h      | 2h         | 480          | 80             | 240      |
| SynthDST-2  | 24h      | 3h         | 480          | 80             | 240      |
| SynthDST-3  | 24h      | 4h         | 480          | 80             | 240      |
| SynthDST-4 | 24h      | Non-uniform| 1920         | 320            | 960      |
| SynthDST-5  | 48h      | 1h         | 480          | 80             | 240      |
| SynthDST-6  | 48h      | 2h         | 480          | 80             | 240      |
| SynthDST-7  | 48h      | 3h         | 480          | 80             | 240      |
| SynthDST-8  | 48h      | 4h         | 480          | 80             | 240      |
| SynthDST-9  | 48h      | 5h         | 480          | 80             | 240      |
| SynthDST-10 | 48h      | 6h         | 480          | 80             | 240      |

---

### 2. Differential Rhythmicity Task

| Dataset     | Duration | Interval   | Training (K) | Validation (K) | Test (K) |
|:-----------:|:--------:|:----------:|:------------:|:--------------:|:--------:|
| SynthDST-11 | 24h      | 2h         | 192          | 32             | 96       |
| SynthDST-12 | 24h      | 3h         | 192          | 32             | 96       |
| SynthDST-13 | 24h      | 4h         | 192          | 32             | 96       |
| SynthDST-14 | 24h      | Non-uniform| 384          | 64             | 192      |
| SynthDST-15 | 48h      | 2h         | 192          | 32             | 96       |
| SynthDST-16 | 48h      | 4h         | 192          | 32             | 96       |
| SynthDST-17 | 48h      | 5h         | 192          | 32             | 96       |
| SynthDST-18 | 48h      | 6h         | 192          | 32             | 96       |

---

## File Structure

Each dataset consists of three time-series files in the following structure:

**Examples**:  

├── SynthDST-1/   # Dataset folder
│ ├── TRAIN.ts  
│ ├── VALIDATION.ts
│ └── TEST.ts
│
│ ├── SynthDST-2/ # Dataset folder
│ │ ├── TRAIN.ts
│ │ ├── VALIDATION.ts
│ │ └── TEST.ts
│
└── ... # (18 Dataset total)

## Data Availability:
Due to file size limitations, the datasets are hosted on figshare and can be accessed via the following link:
[https://figshare.com/s/625915ef49abac604d76](https://figshare.com/s/625915ef49abac604d76)
