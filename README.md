# ✈️ Fighter Aircraft Pilot Ejection Success Prediction

## 📌 Project Overview  
This project predicts the **success rate of a fighter pilot’s ejection** using a **logistic regression model** implemented from scratch. Ejection safety is critical in aviation, but **real-world datasets are rarely available** due to confidentiality. To address this limitation, the project uses **synthetically generated data** that simulates realistic aviation scenarios involving altitude, airspeed, G-force, ejection system type, and pilot posture.  

The objective is to determine whether a given ejection attempt will be classified as a **success** or **failure**, while also analyzing which factors most strongly influence the outcome.  

---

## 📂 Dataset  
- **File:** `Fighter_Pilot_Ejection_Success.csv`  
- **Type:** Synthetic dataset (due to unavailability of real-world aviation data)  
- **Features:**
  - `Altitude_ft`: Height (ft) at the time of ejection (capped at 48,000 ft)  
  - `Airspeed_knots`: Aircraft speed in knots  
  - `G_Force`: G-forces acting on the pilot (capped at 12G)  
  - `Ejection_Type`: `0 = Zero-Zero`, `1 = Conventional`  
  - `Pilot_Posture`: `1 = Optimal`, `0 = Slouched`  
- **Target:**
  - `Ejection_Success`: `1 = Success`, `0 = Failure`  

👉 The synthetic dataset **overcomes the data availability constraint** by simulating plausible operational conditions.  

---

## 🎯 Problem Statement  
From the official problem definition (see `Problem Statement_1001 and 991.pdf`):  

- Predict the likelihood of successful ejection based on flight and human parameters.  
- Identify which factors (altitude, airspeed, posture, system type) most affect the outcome.  
- Test the reliability of logistic regression on synthetic aviation datasets.  
- Understand how operational/environmental conditions influence ejection results.  

---

## ⚙️ Implementation Details  

### 🔹 `regress_impl.py`  
- Preprocesses the dataset (encoding + normalization).  
- Implements logistic regression **from scratch**:
  - **Sigmoid function**  
  - **Cost function (log-loss)**  
  - **Gradient descent optimizer**  
- Trains the model (`alpha = 0.1`, `iterations = 1000`).  
- Saves parameters to `.npy` files (`Trained_theta.npy`, `Mean.npy`, `Standard_dev.npy`).  
- Interactive **prediction loop**: takes user input and outputs success probability.  
- Plots **cost vs iterations** to visualize convergence.  

---

### 🔹 `Project.py`  
- A more **refined version** of the pipeline:  
  - Data cleaning: removes duplicates, handles missing values.  
  - Clips extreme values (altitude ≤ 48,000 ft, G-force ≤ 12).  
  - Numerical stability: `np.clip` in sigmoid, `EPSILON` in log-loss.  
  - Option to **train from scratch** or load pre-trained weights.  
  - Reports **final cost** and **training accuracy**.  
  - Provides user-friendly input prompts and probability predictions.  
  - Visualizes **training progress** with cost plots.  

---

### 🔹 Jupyter Notebooks  
- **`regress_impl.ipynb`** & **`ML_Project.ipynb`**  
  - Interactive development and experimentation.  
  - Show step-by-step preprocessing, normalization, and logistic regression math.  
  - Plot and explain convergence behavior.  
  - Serve as **exploration + documentation notebooks** before final `.py` scripts.  

---

### 🔹 Supporting Files  
- **`Mean.npy` & `Standard_dev.npy`** → Store feature scaling parameters.  
- **`Trained_theta.npy`** → Stores trained model weights.  
- **`cost_plot.png`** → Graph of cost function vs iterations (training progress).  
- **`Problem Statement_1001 and 991.pdf`** → Official project problem statement.  

---

## 🔄 Workflow  
1. **Dataset** → Preprocessing (encoding + normalization).  
2. **Model Training** → Logistic regression via gradient descent.  
3. **Trained Parameters** → Saved for reuse (`.npy` files).  
4. **Prediction** → User inputs flight conditions, model outputs probability of success.  
5. **Visualization** → Cost plots track training progress.  
6. **Notebooks** → Demonstrate the research and experimentation process.  

---

## 📊 Results  
- Logistic regression successfully trained on synthetic aviation data.  
- Training accuracy and cost convergence demonstrate feasibility of prediction.  
- Factors like **altitude, posture, and ejection type** were shown to have a strong influence.  
- Synthetic data, while not replacing real data, provides a **valuable approximation** to study critical aviation safety mechanisms.  

---

## 🚀 Conclusion  
This project demonstrates how **logistic regression** can be applied to safety-critical aviation problems even when **real-world data is unavailable**.  
By designing a synthetic dataset, the project builds a complete **machine learning pipeline**:  
- Data preprocessing  
- Logistic regression model from scratch  
- Training and evaluation  
- Real-time user prediction  

It provides both a **predictive tool** and an **analytical framework** for understanding pilot ejection success factors.  

---

## 🛠️ Tech Stack  
- **Python**  
- **NumPy, Pandas, Matplotlib**  
- **Jupyter Notebook**  

---

## 👨‍💻 Authors  
- **S Harikesh**  
- **Joshua Ayush Kerketta**  

---
