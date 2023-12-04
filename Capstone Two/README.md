# Capstone Two - AUDL Win Model(s)

## Contents

 - [Submissions](#Submissions)
 - [Python Environment](#python-environment--310-311)
 - [Project Overview](#Overview)

### Submissions

 1. [Project Proposal](/Capstone%20Two/7.1_Project%20Propsal.pdf)
 2. [Data Wrangling Notebook](/Capstone%20Two/7.6_Wrangling.ipynb)
 3. [EDA Notebook](/Capstone%20Two/11.6_EDA.ipynb)  | [Outlier Detection](/Capstone%20Two/11.6_EDA_outlier-detection.ipynb) *actual assignment is "11.5"*
 4. [Preprocessing Notebook](/Capstone%20Two/16.3_Preprocessing-Training.ipynb)
 5. [Modeling Notebook](/Capstone%20Two/18.3_Modeling.ipynb)
 6. [Project Report](/Capstone%20Two/Report)
	- [slides](/Capstone%20Two/Report/capstone_two_audl_slides.pdf)
	- [report](/Capstone%20Two/Report/capstone_two_audl_report.pdf)
	- [model metrics](/Capstone%20Two/final_model_info.csv)
   
**| [Data Folder](/Capstone%20Two/data/) | [Graphs Folder](/Capstone%20Two/graphs/) |**

### Python Environment | 3.10, 3.11

<details><summary>show</summary>

**Packages**
 - pandas 
 - numpy
 - requests
 - scikit-learn
 - catboost
 - xgboost
 - lightgbm
 - scipy
 - matplotlib
 - seaborn
 - plotly
 - pyarraow
 - tqdm
 - *separate environment used for PyCaret*
   - [pycaret_requirements](/Capstone%20Two/pycaret_requirements.txt) for list of packages installed in my venv for Pycaret
   - fresh installation without requirements file recommended

</details>

 
   
### Overview

*see the [report](/Capstone%20Two/Report/capstone_two_audl_report.pdf) for more details*

**Background**
 - American Ultimate Disc League (AUDL) is a men’s professional Ultimate Frisbee league established 2013 in North America, currently consisting of 24 teams across four regional divisions
 - use the most basic game summary statistics to predict the outcome of the game: the winning team and the difference in the two teams’ scores
   - final models should provide a basis for understanding the remaining summary statistics
   - data pipeline should exemplify collecting, cleaning, and exploring AUDL data for future studies.
   
**Data Collection**
 - data collected via AUDL API [docs](https://www.docs.audlstats.com/)
   - official endpoint for game lists, [all games 2011-2023](https://www.backend.audlstats.com/api/v1/games?date=2011:2023)
   - unofficial endpoint for game summaries, [example game](https://www.backend.audlstats.com/web-api/game-stats?gameID=2023-05-19-LA-SLC)
   
**Data Cleaning**
 - some 0 values are real, some indicate missing data 
 - newer statistics (redzone, hucks) not recorded until midway through 2020
 - *bring in list of checks from EDA notebook?*
 - Remaining Questions
   - blocks vs turnovers vs incompletions
   - *bring in more from EDA notebook*

<details><summary>Data Cleaning graphs</summary>

<br>**Feature Distributions after data collection**<br>
![Initial](/Capstone%20Two/graphs/data_cleaning/initial_distributions.png "Feature distributions after data collection") 
<br>**Feature Distributions after data cleaning**<br>
![Final](/Capstone%20Two/graphs/data_cleaning/clean_1_distributions.png "Feature distributions after data cleaning") 

</details>
 
**Features**
 - only basic stats used for modeling: **throws, completions, blocks, turnovers**
 - used to engineer additional Features
   - completion rate, completion rate difference, block-turnover difference
 - target features used the final score for definition
   - **home margin** = `home_score - away_score`
   - **home win** = `if home_score > away_score`
   
<details><summary>EDA graphs</summary>

<br>**Feature Distributions, relation to Home Margin**<br>
![Distribution, Margin](/Capstone%20Two/graphs/EDA/hist_vs_margin.png "Features vs home margin") 
<br>**Feature Distributions, relation to Home Win**<br>
![Distribution, Win](/Capstone%20Two/graphs/EDA/all_features_hist_vs_win.png "Features vs home win chance") 
<br>**Feature+Target Correlations**<br>
![Correlation](/Capstone%20Two/graphs/EDA/corr_heatmap.png "Correlation Heat Map") 

*see more thresholds and outlier detection based on PCA components in [folder](/Capstone%20Two/graphs/Outlier%20Detection)*

<br>**Isolation Forest**<br>
![Isolation Forest](/Capstone%20Two/graphs/Outlier%20Detection/engineered%20features/IsoForest_0.05.png "Isolation Forest - outlier detection") 
<br>**Local Outlier Factor**<br>
![Local Outlier Factor](/Capstone%20Two/graphs/Outlier%20Detection/engineered%20features/LocalOutlierFactor_0.05.png "Local Outlier Factor - outlier detection") 

</details>

**Modeling**
 - separate studies for each target, separated into two notebooks
 - PyCaret used to streamline initial studies, scikit-learn + manual loops were then used to evaluate and train final models
 - **Regression for Margin**
   - CatBoost, GBR, XGB (tree), kNN all performed well. Voting Regressor blend of first two selected for final model
 - **Binary Classification for Home win**
   - kNN, Extra Trees, SGD, CatBoost classifiers all performed well. kNN selected for final model.

<details><summary>Regression Model graphs</summary>

<br>**Predicted vs Actual Home Margin**<br>
![residuals_1](/Capstone%20Two/graphs/Model/model-selection_RMSE-vs-MAE.png "Model evaluation by RMSE, R2, MAE") 
<br>**Predicted vs Actual Home Margin**<br>
![residuals_1](/Capstone%20Two/graphs/Model/final_predicted-vs-actual.png "Predicted Home Margin vs Actual Home Margin") 
<br>**Feature Importance**<br>
![residuals_2](/Capstone%20Two/graphs/Model/feature_importance.png "Final model's component feature importances") 

</details>


**Future Work**
 - repeat efforts with team and matchup specific models
   - predict current model's inputs --> predict winning team and score difference?
 - data expansion: game events + player data + position data 
   - established detailed game record collection + persistence
   - frame many other studies from this collection
 



