# Real Datasets Documentation

This repository contains two comprehensive real biological datasets (RealDST1 and RealDST2) for circadian rhythm research. The datasets provide experimentally validated biological measurements with detailed annotations.

---

## Datasets Overview

| Dataset | Number of Sub-datasets | Number of Genes | Key Characteristics |
|:-------:|:----------------------:|:---------------:|:-------------------:|
| **RealDST1** | 38 | 480,983 | Contains positive/negative instances  |
| **RealDST2** | 19 | 108,835 | Contains period change, amplitude change, sphase shifts, base shifts, combined differences, and unchanged genes |

---

## Detailed Breakdown

### RealDST1 Characteristics

- **Positive instances**: 329,384  
- **Negative instances**: 151,599  

> **Note**: Positive instances indicate genes exhibiting circadian oscillation patterns, while negative instances show no oscillation.

### RealDST2 Characteristics

- **Period change**: 6,585  
- **Amplitude change**: 4,888  
- **Phase shift**: 18,875  
- **Base shift**: 1,239  
- **Combined difference**: 6,440  
- **No change**: 70,808  

> **Note**: These categories represent different types of differential rhythmicity patterns observed under experimental conditions.

---

## File Structure

Each dataset is organized with sub-datasets containing time-series measurements:

├── RealDST1/ # Main dataset folder
│ ├── RealDST1-1/ # Sub-dataset folder
│ │ ├── TRAIN.ts
│ │ ├── VALIDATION.ts
│ │ └── TEST.ts
│ │
│ ├── RealDST1-2/ # Sub-dataset folder
│ │ ├── TRAIN.ts
│ │ ├── VALIDATION.ts
│ │ └── TEST.ts
│ │
│ └── ... # (38 sub-datasets total)
│
└── RealDST2/ # Main dataset folder
│ ├── RealDST2-1/ # Sub-dataset folder
│ │ └── TEST.ts
│
│ ├── RealDST2-2/ # Sub-dataset folder
│ │ └── TEST.ts
│
│ └── ... # (19 sub-datasets total)

## Data Availability:
Due to file size limitations, the datasets are hosted on figshare and can be accessed via the following link:
[https://figshare.com/s/625915ef49abac604d76](https://figshare.com/s/625915ef49abac604d76)
