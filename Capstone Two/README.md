# Capstone Two - AUDL Win Model(s)

## Contents

 - [Submissions](#Submissions)
 - [Python Environment](#python-environment--310-311)
 - [Project Overview](#Overview)

## Submissions

 1. [Project Proposal](./7.1_Project%20Propsal.pdf)
 2. [Data Wrangling Notebook](./7.6_Wrangling.ipynb)
 3. [EDA Notebook](./11.6_EDA.ipynb)  | [Outlier Detection](./11.6_EDA_outlier-detection.ipynb) *actual assignment is "11.5"*
 4. [Preprocessing Notebook](./16.3_Preprocessing-Training.ipynb)
 5. [Modeling Notebook](./18.3_Modeling.ipynb) | [Classification Notebook](./18.3_Modeling_Classification.ipynb)
 6. Project Documentation
	- [slides](./slides.pdf)
	- [report](./report.pdf)
	- [model metrics](./final_model_info.csv)
   
**| [Data Folder](.data/) | [Graphs Folder](./graphs/) |**

## Python Environment | 3.10, 3.11

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
   - [pycaret_requirements](./pycaret_requirements.txt) for list of packages installed in my venv for Pycaret
   - fresh installation without requirements file recommended

</details>

 
   
## Overview

 1. [Background, Problem Statement](#background,-problem-statement)
 2. [Data Collection](#data-collection)
 3. [Data Cleaning](#data-cleaning)
 4. [EDA and Feature selection, engineering](#features)
 5. [Modeling](#modeling)
 6. [Future Work](#future-work)
 7. [Win Classification](#win-classification) *alternate problem statement*

### Background, Problem Statement
 - American Ultimate Disc League (AUDL) is a men’s professional Ultimate Frisbee league established 2013 in North America, currently consisting of 24 teams across four regional divisions
 - use the most basic game summary statistics to predict the outcome of the game: **the winning team and the difference in the two teams’ scores**
   - final models should provide a basis for understanding the remaining summary statistics
   - data pipeline should exemplify collecting, cleaning, and exploring AUDL data for future studies.
   - this page focuses on regression for `home_margin`, see [win classification]() work below
   
### Data Collection
 - data collected via AUDL API [docs](https://www.docs.audlstats.com/)
   - official endpoint for game lists, [all games 2011-2023](https://www.backend.audlstats.com/api/v1/games?date=2011:2023)
   - unofficial endpoint for game summaries, [example game](https://www.backend.audlstats.com/web-api/game-stats?gameID=2023-05-19-LA-SLC)
   
### Data Cleaning
 - some 0 values are real, some indicate missing data 
 - newer statistics (redzone, hucks) not recorded until midway through 2021
 - see [report](./report.pdf) appendix for detailed data checks
 - still unsure of: blocks vs turnovers vs incompletions

<details><summary>Data Cleaning graphs</summary>

<br>**Feature Distributions after data collection**<br>
![Initial](/Capstone%20Two/graphs/data_cleaning/initial_distributions.png "Feature distributions after data collection") 
<br>**Feature Distributions after data cleaning**<br>
![Final](/Capstone%20Two/graphs/data_cleaning/clean_1_distributions.png "Feature distributions after data cleaning") 

</details>
 
### Features
 - only basic stats used for modeling: **throws, completions, blocks, turnovers**
 - these were used to engineer additional features
   - **completion rate, completion rate difference, block-turnover difference**
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

**Automated Outlier Detection**<br>
*see more thresholds and outlier detection based on PCA components in [folder](/Capstone%20Two/graphs/Outlier%20Detection)*

<br>**Isolation Forest**<br>
![Isolation Forest](/Capstone%20Two/graphs/Outlier%20Detection/engineered%20features/IsoForest_0.05.png "Isolation Forest - outlier detection") 
<br>**Local Outlier Factor**<br>
![Local Outlier Factor](/Capstone%20Two/graphs/Outlier%20Detection/engineered%20features/LocalOutlierFactor_0.05.png "Local Outlier Factor - outlier detection") 

</details>

<details><summary>Feature Engineering graphs</summary>

*completion rate example*

`completion rate = completions / throws`

`completion rate difference` = `home completion rate` - `away completion rate`

<br>**Completion Rate Distributions**<br>
![Completion Rate Difference](/Capstone%20Two/graphs/EDA/feature%20engineering/distributions_2.png "completion rate difference more normally distributed than either rate itself") 
<br>**Away vs Home completion rate, colored by Home Margin**<br>
![Distribution, Win](/Capstone%20Two/graphs/EDA/feature%20engineering/comp_rate_study.png "completion rates trend slightly with margin") 
<br>**Completion Rate Difference vs Home Margin**<br>
![Correlation](/Capstone%20Two/graphs/EDA/feature%20engineering/comp_rate_diff-vs-margin.png "completion rate difference more clearly defined relationship with margin") 

*see also: `block turnover difference`*
 - [distributions](/Capstone%20Two/graphs/EDA/feature%20engineering/distributions_1.png)
   - home/away blocks are added to away/home turnovers to get a total for a given team (`block turnover`'s), rates are the total / number of points played.
 - [individual vs margin](/Capstone%20Two/graphs/EDA/feature%20engineering/turnover_rates.png), [difference vs margin](/Capstone%20Two/graphs/EDA/feature%20engineering/block_turnover_diff-vs-margin.png)
   - as shown above, the difference between the two teams correlates more strongly to `home margin` than either itself


</details>

### Modeling
 - separate studies for each target, separated into two notebooks
 - PyCaret used to streamline initial studies, scikit-learn + manual loops were then used to evaluate and train final models
   - final hyperparameter tuning with RandomizedSearch, [help with distributions](https://nbpub.pythonanywhere.com/)
 - **Regression for Margin**
   - CatBoost, GBR, XGB (tree based boosting), kNN all performed well. Voting Regressor blend of CatBoost+GBR selected for final model
     - residuals analyzed and shown below. two worst errors were games with faulty records that should have been removed during cleaning.
 - **Binary Classification for Home win**
   - kNN, Extra Trees, SGD, CatBoost classifiers all performed well. kNN selected for final model
     - kNN better recall, slightly worse precision. not as good at identifying narrow lossess?
	 - results from one more false positive than others, possible faulty record?

<details><summary>Regression Model graphs</summary>

<br>**Tuned Model Selection**<br>
![model_selection](/Capstone%20Two/graphs/Model/model-selection_RMSE-vs-MAE.png "Model evaluation by RMSE, R2, MAE") 
<br>**Predicted vs Actual Home Margin**<br>
![residuals_1](/Capstone%20Two/graphs/Model/residual%20analysis/final_predicted-vs-actual.png "Predicted Home Margin vs Actual Home Margin") 
<br>**Feature Importance**<br>
![feature_importance](/Capstone%20Two/graphs/Model/feature_importance.png "Final model's component feature importances") 

</details>


### Future Work
 - repeat efforts with team and matchup specific models
   - predict current model's inputs --> predict winning team and score difference?
 - data expansion: game events + player data + position data 
   - established detailed game record collection + persistence
   - frame many other studies from this collection
 
 
## Win Classification 

*see [Classification Notebook](./18.3_Modeling_Classification.ipynb) for details*

 - same process for data preparation as used for `home_margin` regression, different target variable: `home_win`
   - `home_win` is True if the home score is greater than the away score at the end of the game
     - slight home-field advantage noted, see [report](./report.pdf)
	 - ties should not occur by game design, and two existed in the dataset. they were treated as losses: `home_win=False`
 - tested a wide variety of classifcation models, **progressed kNN, SGD, and ExtraTrees classifiers** for hyperparameter tuning and final comparison
   - determined feature selection (`SelectKBest`) and normalization conditions for linear and kNN models
   - did not employ feature selection for ensemble model
   - training and testing scores similar for all models, SGD may be [overfitting slightly](/Capstone%20Two/graphs/Model/classification/overfitting_check.png)
 - kNN better recall, but worse precision than ExtraTrees and SGD classifiers. kNN higher in aggregate scores: F1, Balanaced Accuracy, ROC-AUC
   - in more words, kNN better at home wins correctly but worse at predicting home losses
 - Voting/Stacking used to blend final three models (tuned) and CatBoost classifier (untuned)
   - kNN + CatBoost + ExtraTrees blend (`VotingClassifier`) may provide improvement over kNN or ExtraTrees alone
   
   
| Model | precision | recall | F1 | Balanced Accuracy | ROC-AUC |
|-------|-----------|--------|----|-------------------|---------|
|KNeighborsClassifier|	0.949|	0.943|	0.946|	0.937	|0.937|
|kNN+Cat+ET|	0.954	|0.938	|0.946	|0.938|	0.938|
|ExtraTreesClassifier|	0.953|	0.926|	0.939|	0.932|	0.932|
|SGDClassifier	|0.953|	0.926	|0.939	|0.932	|0.932|
|kNN+Cat|	0.959|	0.920|	0.939	|0.933	|0.933|
|CatBoostClassifier|	0.948|	0.926	|0.937|	0.928|	0.972|
|kNN+Cat+SGD	|0.948	|0.926	|0.937	|0.928	|0.928|


<details><summary>Classification graphs</summary>

<br>**Model Selection 1**<br>
![classification_selection_1](/Capstone%20Two/graphs/Model/classification/F1-vs-ROCAUC.png "Model selection: F1 vs ROC-AUC") 
<br>**Model Selection 2**<br>
![classification_selection_2](/Capstone%20Two/graphs/Model/classification/recall-vs-precision.png "Model selection: Recall vs Precision") 
<br>**Top 3 classification models: SGD, kNN, ExtraTrees**<br>
![classification_top_1](/Capstone%20Two/graphs/Model/classification/ROC-curve.png "") 
![classification_top_2](/Capstone%20Two/graphs/Model/classification/PR-curve.png "") 
![classification_top_3](/Capstone%20Two/graphs/Model/classification/DET-curve.png "") 
![classification_top_4](/Capstone%20Two/graphs/Model/classification/Calibration-curve.png "") 
<br>
![classification_top_5](/Capstone%20Two/graphs/Model/classification/top_3_estimator_parameters.png "") 

[see more](/Capstone%20Two/graphs/Model/classification/)

</details>


