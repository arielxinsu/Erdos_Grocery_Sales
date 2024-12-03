# Table of Contents

- [Table of Contents](#table-of-contents)
- [Executive Summary](#executive-summary)
  - [Project Description](#project-description)
  - [Data Collection](#data-collection)
  - [KPIs](#kpis)
  - [Model Selection and Evaluation](#Model-Selection-and-Evaluation)
  - [Repository structure](#Repository-structure)

---

## Executive Summary

### Project Description
This project addresses the **[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)** challenge. The goal is to predict daily sales for thousands of product families sold at Favorita grocery stores in Ecuador. Accurate forecasting can improve inventory management, reduce waste, and enhance customer satisfaction.

Key tasks include:
- Understanding historical sales data trends.
- Incorporating external factors such as holidays and oil prices.
- Building predictive models to forecast future sales.


---

### Data Collection
The dataset comprises multiple files with key details:
1. **`train.csv`**: Historical sales data with:
   - `date`: Sales date.
   - `store_nbr`: Store identifier.
   - `family`: Product family.
   - `sales`: Total sales for a product family (can be fractional).
   - `onpromotion`: Count of promotional items.
2. **`test.csv`**: Contains the same features as `train.csv`, excluding `sales`. Predictions are required for these dates.
3. **`stores.csv`**: Metadata about stores, including location, type, and cluster grouping.
4. **`oil.csv`**: Daily oil prices, relevant due to Ecuador’s oil-dependent economy.
5. **`holidays_events.csv`**: Holiday and event details with classifications like `type`, `transferred`, and `locale`. The `transferred` column indicates when a holiday was officially moved to another date by the government, affecting how it impacts sales. 

For further details, refer to the **[competition dataset description](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)**.


Key observations about the data:
- **Store details**: Stores are categorized into clusters based on similar characteristics (e.g., location and type). 
- **Sales data**: Includes fractional sales values as products can be sold in non-integer quantities.
- **External factors**: Oil prices and holidays influence sales and must be integrated into predictive models.



---


**Feature Engineering Highlights**:
- **Lagged sales features**: Creating features that capture past sales for a given time period to predict future sales.
- **Oil price adjustments**: Incorporating oil price data as an important external variable influencing sales.
- **Holiday adjustments**: Including holidays and special events that can cause sales spikes or drops.


---

### KPIs
To evaluate model performance, the following metrics were prioritized:

1. **Root Mean Squared Logarithmic Error (RMSLE)**:
  !<p align="left">
  <img src="Visuals/RMSLE.png" alt="RMSLE" title="RMSLE" style="max-width:75%;">
</p>
 
   - **Target**: Achieve RMSLE < 0.50.

3. **Training Efficiency**:  
   - **Target**: Model training time should be under 2 hours on Kaggle’s environment.

4. **Feature Contribution**:  
   - **Target**: Demonstrate at least a 10% improvement in RMSLE with engineered features.

The **RMSLE** metric is crucial because it penalizes large errors more heavily than small ones, which is important for sales data, where extreme values can disproportionately affect business operations.

---

### Model Selection and Evaluation

We tested several modeling approaches, including baseline models and more advanced models. The performance was evaluated based on an appropriate score, and the results are as follows:

#### **Baseline Models:**
1. **Mean by Years for the Same Date**: Score = 0.902  
   A simple model that predicts the mean value for each date based on historical data from the same date in previous years. It performs well but lacks any predictive power beyond basic historical averages.
  
2. **Rolling Average**: Score = 0.46141  
   A model that calculates the rolling average over a certain number of past data points. It smooths out fluctuations but doesn’t account for any seasonal or complex patterns.
  
#### **Advanced Models:**
3. **Random Forest Regressor**: Score = 0.51442  
   An ensemble model based on decision trees. It handles non-linearity well and can capture complex relationships in the data, but the score shows it may not fully capture the underlying patterns.
  
4. **Prophet Model**: Score = 0.48433  
   A time-series forecasting model developed by Facebook, specifically designed for handling seasonal data with trends. Although it performs better than some models, its score indicates it might not fit the data well for this task.
  
5. **SARIMAX**: Score = 0.47600  
   A classical time-series model that accounts for seasonality, trends, and residuals. It’s effective in many situations but shows lower performance on this dataset.

---

### Insights:
- **Mean by Years** provides the highest score (0.902), indicating it is the least accurate model, which is good since a low score indicates better performance.
- **Rolling Average** scores much lower, showing it has a better fit compared to the mean-based approach, but it still does not capture complex patterns.
- **Random Forest** and **Prophet** show moderate performance, suggesting that more complex models are still not capturing all aspects of the data as well as they could.
- **SARIMA** has the lowest score among all models, suggesting it’s the most accurate and better suited to the task at hand.

Overall, the SARIMAX model has performed the best, but you might want to explore further optimization or hybrid approaches to improve accuracy even more.


- **Models Runtime**: Efficiently trained in under 2  hours.
- **Feature Engineering Impact**: The inclusion of lagged sales, oil prices, and holiday adjustments significantly improved prediction accuracy.

**Impact**:
- Improved forecasting accuracy highlights the potential for better inventory management, reduced food waste, and higher customer satisfaction.
- The model can serve as a decision-support tool for retailers in inventory and promotions planning.

---


## Repository Structure

The repository is organized into the following folders:

- **Data**: Contains the original dataset and the cleaned, preprocessed data.
  - **competition_data**: This folder holds the original data provided by Kaggle, including the `train.csv`, `test.csv`, `oil.csv`, `stores.csv`, `holidays_events.csv`, and other related files.
  - **processed_data**: Contains the cleaned and merged data after preprocessing steps, ready for model training.

- **Master**: The main folder for code, models, and exploratory data analysis.
  - **Cleaning**: This folder includes notebooks and scripts used to preprocess and clean the raw data.
  - **EDA**: This folder contains exploratory data analysis notebooks and scripts to understand the data and derive insights.
  - **Models**: This folder stores the trained models, including the baseline models (e.g., Random Forest Regressor) and any other custom models developed for the competition.
  - **Predictions**: This folder contains the output files from the trained models, including predictions submitted to Kaggle and comparison results with the leaderboard metrics.  
 Predictions
- **README.md**: The project documentation file containing an overview, setup instructions, and other useful information about the project.

---




