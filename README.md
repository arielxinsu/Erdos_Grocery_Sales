# Table of Contents

- [Table of Contents](#table-of-contents)
- [Executive Summary](#executive-summary)
  - [Project Description](#project-description)
  - [Data Collection](#data-collection)
  - [Model Selection](#model-selection)
  - [KPIs](#kpis)
  - [Results](#results)
- [Changelog](#changelog)
- [XinSu Branch Update](#xinsu-branch-update)

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

### Model Selection
We explored several modeling approaches, including:
1. **Time-Series Models**: Leveraging historical patterns with lagged features, rolling averages, and autoregressive models.
2. **Machine Learning Models**:
   - Gradient Boosting (XGBoost, LightGBM, CatBoost).
   - Random Forests.


**Feature Engineering Highlights**:
- **Lagged sales features**: Creating features that capture past sales for a given time period to predict future sales.
- **Oil price adjustments**: Incorporating oil price data as an important external variable influencing sales.
- **Holiday adjustments**: Including holidays and special events that can cause sales spikes or drops.


---

### KPIs
To evaluate model performance, the following metrics were prioritized:

1. **Root Mean Squared Logarithmic Error (RMSLE)**:
   \[
   \text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2}
   \]
   - **Target**: Achieve RMSLE < 0.50.

2. **Training Efficiency**:  
   - **Target**: Model training time should be under 2 hours on Kaggle’s environment.

3. **Feature Contribution**:  
   - **Target**: Demonstrate at least a 10% improvement in RMSLE with engineered features.

The **RMSLE** metric is crucial because it penalizes large errors more heavily than small ones, which is important for sales data, where extreme values can disproportionately affect business operations.

---

### Results
- **RMSLE Achieved**: 0.48, surpassing the target threshold.
- **Model Runtime**: Efficiently trained in under 2  hours.
- **Feature Engineering Impact**: The inclusion of lagged sales, oil prices, and holiday adjustments significantly improved prediction accuracy.

**Impact**:
- Improved forecasting accuracy highlights the potential for better inventory management, reduced food waste, and higher customer satisfaction.
- The model can serve as a decision-support tool for retailers in inventory and promotions planning.

---


# Previous Readme

# Table of contents

- [Table of contents](#table-of-contents)
- [Executive Summary  ](#executive-summary--)
  - [Project Description ](#project-description-)
  - [Data Collection ](#data-collection-)
  - [Model Selection ](#model-selection-)
  - [KPIs ](#kpis-)
  - [Results ](#results-)
- [Changelog ](#changelog-)
- [XinSu Branch Update:](#xinsu-branch-update)


# Executive Summary  <a name="executivesummary"></a>

## Project Description <a name="description"></a>
Our primary aim is to predict unit sales for families of products sold at 52 Favorita grocery stores in Ecuador. 

## Data Collection <a name="datacollection"></a>

## Model Selection <a name="modelselection"></a>

## KPIs <a name="kpi"></a>

## Results <a name="results"></a>

- data_visual_GUI.py creates an interactive GUI to check the train and test data for the choosen store and family type. 
  - To activate the GUI, use 'python data_visual_GUI.py'



# Changelog <a name="chagelog"></a>
Tues Nov 03 2024  Fernando Liu Lopez <fcl2@rice.edu>

    * Master: created Master folder to consolidate all our results and clean notebooks

    * Master/1. Data Collection and Preliminary Analysis.ipynb: lists our sources and initial observations of our data sets
    
    * Master/2. Cleaning.ipynb: cleans and merges our datasets

    * Data/merged_train.csv: merged training data


Mon Nov 11 2024  Fernando Liu Lopez  <fcl2@rice.edu>

    * Master/2. Cleaning.ipynb: added holiday data to training set

    * Data/merged_train.csv: updated training set based on cleaning done above

Thurs Nov 14 2024  Fernando Liu Lopez <fcl2@rice.edu>

    * Master/2. Cleaning.ipynb: added an alternative way to merge data in case people want to keep all the rows (see the Merging (Alternative) section in the notebook). 

    * Data/merged_train_alt.csv: alternative training set with all 3 million rows but some missing values.

    * Master/3. EDA.ipynb: created notebook to contain EDA, visuals, and other findings. populated notebook with some visuals.

Sun Nov 17 2024 Dongyu 

    * Master/2. Clearning.ipynb: added a new data set X_new, fixed the "transferred" error, excluded event from "national holidays" and set event as a separate column. 
    * Master/holidays.ipynb: copied the holiday script to main branches. Will update to the EDA notebook once completed. 


Thurs Nov 21 2024  Fernando Liu Lopez <fcl2@rice.edu>

    * Master/2. Cleaning (version 2).ipynb: cleaned up reading notebook, added merged testing set
    * Master/3. EDA.ipynb added more visuals

Fri Nov 22 2024  Maksim Kosmakov, Fernando Liu Lopez <fcl2@rice.edu>

    * Master/4. Baseline Models.ipynb: added notebook with baseline models. This is a copy of Maksim's baseline-models.ipynb notebook, with minor edits.





# XinSu Branch Update:
1.  EDA: Sales Trends by Family Types
2.  Major findings:

    (1)Missing data identified in the following categories: Baby Care, Books, Celebration Transactions, Home & Kitchen I, Home & Kitchen II, Home Care, Ladieswear, Magazines, Pet Supplies, Electronics, Produce, and School & Office Supplies.
    
    (2) Unusual spike observed around July 2016.



