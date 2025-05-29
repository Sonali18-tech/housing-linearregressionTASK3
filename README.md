# 🧠 Linear Regression - Housing Price Prediction

## 📌 Task 3: AI & ML Internship | Simple & Multiple Linear Regression

This project is a part of the AI & ML internship task series where I implemented and interpreted **Simple** and **Multiple Linear Regression** using Python. The goal was to understand regression modeling, evaluate it using performance metrics, and visualize insights using real-world data.

---

## 📂 Dataset

**Kaggle Housing Price Prediction Dataset**  
🔗 [Download Link](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

This dataset contains details such as area, number of bedrooms, bathrooms, furnishing status, etc., and the corresponding house prices. It’s ideal for linear regression as we aim to predict a continuous variable: `Price`.

---

## 🛠️ Tools & Libraries Used

- Python
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – model training & evaluation

---

## 📊 Workflow

### ✅ Step 1: Data Preprocessing

- Loaded CSV dataset
- Handled categorical variables with **one-hot encoding**
- Checked for and handled missing/null values
- Feature scaling was considered but skipped since Linear Regression in `sklearn` handles this reasonably for well-distributed data.

### ✅ Step 2: Exploratory Data Analysis (EDA)

- Correlation heatmap to check feature relevance
- Pairplots and distribution plots for understanding spread and relationships
- Outlier detection (optional for standout submissions)

### ✅ Step 3: Model Building

- **Simple Linear Regression**: Used `area` as a single feature to predict `price`.
- **Multiple Linear Regression**: Used all numerical and one-hot encoded features.
- Split data into training (80%) and testing (20%) sets.

### ✅ Step 4: Model Evaluation

Used the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score (Coefficient of Determination)**

```python
MAE:  970043.403920164
MSE:   1754318687330.6643
R² Score: 0.6529242642153184

Visualizations
📌 Regression line over actual scatter plot (Simple Regression)

📉 Residual plots to validate assumptions

📊 Feature Coefficients bar plot (Multiple Regression)

🔍 Optional: Pairplot and correlation matrix to highlight relationships

Key Learnings & Highlights
1.Understood core concepts of linear regression including its assumptions.
2.Learned the difference between simple and multiple regression and when to use which.
3.Gained hands-on experience in visualizing model predictions and diagnostics.
4.Practiced real-world debugging (e.g., plotting issues, categorical encoding).

Author
Sonali18-tech
