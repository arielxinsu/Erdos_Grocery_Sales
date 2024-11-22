# Table of contents

- [Table of contents](#table-of-contents)
- [Executive Summary ](#executivesummary-)
    - [Project Description ](#description-)
    - [Data Collection ](#datacollection-)
    - [Model Selection ](#modelselection-)
    - [KPIs ](#kpi-)
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







# XinSu Branch Update:
1.  EDA: Sales Trends by Family Types
2.  Major findings:

    (1)Missing data identified in the following categories: Baby Care, Books, Celebration Transactions, Home & Kitchen I, Home & Kitchen II, Home Care, Ladieswear, Magazines, Pet Supplies, Electronics, Produce, and School & Office Supplies.
    
    (2) Unusual spike observed around July 2016.



