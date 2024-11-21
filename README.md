# Sales Prediction using Python

## Overview
This project demonstrates a **Sales Prediction System** using Python and machine learning. The goal is to predict future sales based on advertising budgets across different media channels like TV, Radio, and Newspapers.

### Key Objective
Predict how much of a product people will buy based on:
- Advertising budget allocation.
- Target segment preferences.
- Advertising platform efficiency.

This project uses machine learning models to provide accurate sales forecasts, which are crucial for data-driven decision-making in product and service-based businesses.

---

## Dataset
The dataset used for this project contains advertising budgets for three mediums (TV, Radio, Newspaper) and the corresponding sales figures.  
- **Download Dataset**: [Advertising.csv](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)

---

## Tools and Libraries
- **Python Libraries**:
  - `pandas` & `numpy` – Data manipulation and analysis.
  - `matplotlib` & `seaborn` – Data visualization.
  - `sklearn` – Machine learning models and evaluation.

---

## Project Workflow
1. **Data Exploration**:
   - Inspected dataset using `.info()` and `.describe()`.
   - Checked for missing values and dropped unnecessary columns.

2. **Data Visualization**:
   - Plotted pairwise relationships between features and sales.

3. **Model Building**:
   - Split data into training (70%) and testing (30%) sets.
   - Used the following regression models:
     - **Linear Regression**
     - **Decision Tree Regressor**
     - **Random Forest Regressor**

4. **Model Evaluation**:
   - Calculated performance metrics:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - R-squared (R² Score)

5. **Result Visualization**:
   - Compared actual vs. predicted sales using scatter plots.

---

## Results
| **Model**                | **MSE** | **RMSE** | **R² Score** |
|--------------------------|---------|----------|--------------|
| Linear Regression        | 3.80    | 1.95     | 0.86         |
| Decision Tree Regressor  | 1.55    | 1.25     | 0.94         |
| Random Forest Regressor  | 0.46    | 0.68     | 0.98         |

- **Best Model**: Random Forest Regressor with an R² Score of 0.98.

---

## Visualization
### Advertising Budget vs. Sales
Pairwise scatter plots showed the relationship between advertising mediums and sales.

### Actual vs. Predicted Sales
A scatter plot illustrated the accuracy of the predictions compared to actual sales.

---

## How to Run the Code
1. **Install Required Libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Clone the Repository**:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

3. **Run the Notebook**:
   Open the `Sales Prediction using Python.ipynb` file in Jupyter Notebook or Google Colab and execute cells step-by-step.

---

## Author
This project was created to demonstrate the application of machine learning in sales prediction. If you have questions or feedback, feel free to reach out!
