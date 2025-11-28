# Quantum Reservoir Computing for Options Pricing & Swaption Time-Series Imputation

This repository contains an end-to-end workflow that uses **quantum reservoir computing (QRC)** to analyse financial time-series, learn deviation signals in option prices, and impute missing values in swaption datasets.  
The approach integrates:

- **Deviation-based preprocessing** using Black–Scholes baselines  
- **Heisenberg/Ising-based quantum reservoirs** implemented in **myQLM**  
- **Qubit-reuse reservoir architecture** with disorder fields  
- **Classical readout layers** (MLPs) trained on extracted ⟨σᶻ⟩ features  
- **Imputation pipeline** for swaption price matrices  

A high-level schematic and conceptual background are provided in the slide deck *Reservoir Computing for Financial Time-Series*.

---

## 📂 Repository Structure

### **1. Data Generation & Option-Deviations Pipeline**

- **`gen_dat.py`** – Loads option-market files, merges them with underlying asset data, computes mid-prices, and constructs **Black–Scholes deviations**.

- **Sample input file** (small example):  
  Located in **`opt_data/`**, alongside the **imputed swaption file** produced by the imputation pipeline.

- **Full historical option-chain dataset**  
  The full dataset used for experiments is **not included in this repo** due to size but is **available upon request**.

---

### **2. Quantum Reservoir Implementation**

- **`reserve_mem.py`** – Core implementation of the quantum reservoir:
  - Trotterized Heisenberg & Ising blocks  
  - Random disorder fields  
  - Qubit reuse architecture  
  - Extraction of ⟨σᶻ⟩ expectation values  
  - Lag-feature construction for time-series learning  

---

### **3. End-to-End Learning Script**

- **`reserve_end-to-end.py`** – Full learning workflow:
  - Load options data via `generate_data()`  
  - Construct deviation time series  
  - Sweep over memory sizes  
  - Train a neural readout layer  
  - Produce MAE / RMSE / R² metrics  
  - Plot training loss curves  

---

### **4. Parallel Experiments & Metric Logging**

- **`reserve_multip.py`** – Multiprocessing wrapper for running memory-sweep experiments in parallel.  
  Saves:
  - `loss_curves_*.npy`  
  - `error_metrics_*.json`

- **`reserve_plot.py`** – Plotting utilities for aggregating results across all runs.

---

### **5. Swaption Time-Series Imputation Pipeline**

- **`swaptions_aid.py`** – Step-wise imputation routine:
  - Selects clean windows  
  - Extracts QRC features  
  - Trains a lightweight MLP readout  
  - Performs forward-filling via quantum reservoir predictions  

- **`swaptions_run.py`** – End-to-end imputation execution:
  - Parallel imputation of multiple columns  
  - Writes output to  
    **`opt_data/sample_Simulated_Swaption_Price_imputed.xlsx`**

**Both the sample input file and the imputed file are provided in the `opt_data/` directory.**

---

## 🔍 Tasks Supported

### **1. Options Price Deviation Learning**
- Learn deviation dynamics for a chosen strike/expiry.  
- Compare performance across reservoir memory sizes.  
- Study effect of disorder and Trotterisation depth.

### **2. Swaption Matrix Imputation**
- Handle missing values with minimal data.  
- Produce fully imputed Excel outputs.  
- Demonstrate the utility of quantum reservoirs for structural financial datasets.

---

## 📦 Requirements

- **myQLM**  
- **scikit-learn**  
- **numpy / pandas**  
- **matplotlib**  
- **multiprocessing**

---

## ▶️ How to Run

### **Option-Deviation Learning**
```
python reserve_end-to-end.py
```

### **Parallel Memory Sweep**
```
python reserve_multip.py
```

### **Swaptions Imputation**
```
python swaptions_run.py
```

Output will be written to:
```
opt_data/sample_Simulated_Swaption_Price_imputed.xlsx
```

---

## 📬 Data Availability

- **Sample files** (options & swaptions) are included in `opt_data/`.  
- **Imputed swaption file** is generated automatically in the same folder.  
- **Full-scale options dataset** is **available upon request**.

---

## 📘 Citation

If you use this repository, please cite the accompanying slide deck:
*Effect of Reservoir Memory on Learning Options Price*  
Harshit Verma

