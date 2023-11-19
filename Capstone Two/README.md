# Capstone Two - AUDL Win Model(s)

## Contents

 1. [Project Proposal](/Capstone%20Two/propsal.pdf)
 2. [Data Wrangling Notebook](/Capstone%20Two/7.6_Wrangling.ipynb)
 3. [EDA Notebook](/Capstone%20Two/11.6_EDA.ipynb) *actual assignment is "11.5"*
 4. [Preprocessing Notebook](/Capstone%20Two/16.3_Preprocessing-Training.ipynb)
 5. [Modeling Notebook](/Capstone%20Two/18.3_Modeling_final.ipynb) | [Model Selection](/Capstone%20Two/18.3_Modeling_selection.ipynb)
 6. [Project Report](/Capstone%20Two/Report)
	- [slides](/Capstone%20Two/Report/slides.pdf)
	- [report](/Capstone%20Two/Report/report.pdf)
   
**| [Data Folder](/Capstone%20Two/data/) | [Graphs Folder](/Capstone%20Two/graphs/) |**

### Python Environment | 3.10, 3.11

**Packages**
 - pandas 
 - numpy
 - requests
 - scikit-learn
 - scipy
 - matplotlib
 - seaborn
 - plotly
 - pyarraow
 - tqdm
 - ydata_profiling
 
### PyCaret Virtual Environment

See [pycaret_requirements](/Capstone%20Two/pycaret_requirements.txt) for list of packages installed in my venv for Pycaret. Fresh installation without requirements file recommended.
 
   
## Overview

. . .


### Selected Graphs

<details><summary>Data Cleaning</summary>

<br>**Feature Distributions after data collection**<br>
![Initial](/Capstone%20Two/graphs/data_cleaning/initial_distributions.png "Feature distributions after data collection") 
<br>**Feature Distributions after data cleaning**<br>
![Final](/Capstone%20Two/graphs/data_cleaning/clean_1_distributions.png "Feature distributions after data cleaning") 

</details>

<details><summary>EDA - Feature Distributions and Relationships</summary>

<br>**Feature Distributions, relation to Home Margin**<br>
![Distribution, Margin](/Capstone%20Two/graphs/EDA/hist_vs_margin.png "Features vs home margin") 
<br>**Feature Distributions, relation to Home Win**<br>
![Distribution, Win](/Capstone%20Two/graphs/EDA/all_features_hist_vs_win.png "Features vs home win chance") 
<br>**Feature+Target Correlations**<br>
![Correlation](/Capstone%20Two/graphs/EDA/corr_heatmap.png "Correlation Heat Map") 

</details>

<details><summary>Automated Outlier Detection</summary>

*see more thresholds and outlier detection based on PCA components in [folder](/Capstone%20Two/graphs/Outlier%20Detection)*

<br>**Isolation Forest**<br>
![Isolation Forest](/Capstone%20Two/graphs/Outlier%20Detection/engineered%20features/IsoForest_0.05.png "Isolation Forest - outlier detection") 
<br>**Local Outlier Factor**<br>
![Local Outlier Factor](/Capstone%20Two/graphs/Outlier%20Detection/engineered%20features/LocalOutlierFactor_0.05.png "Local Outlier Factor - outlier detection") 


</details>

<details><summary>Preprocessing</summary>

<br>**Linear Model Feature Selection**<br>
![feature selection](/Capstone%20Two/graphs/PreProc/linear-models_feature-selection-zoomed.png "various linear models performance vs number of features selected") 
<br>**Model Selection after HyperParameter Tuning**
![model selection](/Capstone%20Two/graphs/PreProc/model-selection_RMSE-vs-MAE.png "Tuned model metrics") 

</details>

<details><summary>Final Model</summary>

<br>**Predicted vs Actual Home Margin**<br>
![residuals_1](/Capstone%20Two/graphs/Model/final_predicted-vs-actual.png "Predicted Home Margin vs Actual Home Margin") 
<br>**Residuals vs Season**<br>
![residuals_2](/Capstone%20Two/graphs/Model/final_residual-by-year.png "Residuals by season") 
<br>**Residuals vs Home/Away Teams**<br>
![residuals_3](/Capstone%20Two/graphs/Model/final_residual-by-team.png "Resiudals by home/away team")

</details>




