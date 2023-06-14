# Red Wine Regression Case Study

*completed Tier 3 and deleted other notebooks*

### Process Notes


**1. Sourcing and loading** 

- Exploring the data
- Choosing a dependent variable
  - fixed acidity chosen for wine reasons
  - quality not chosen as it was a discrete variable
    - ultimately dropped

**2. Cleaning, transforming, and visualizing**
- Visualizing correlations
  - correlation tables
  - pair plots
  - heat maps of correlation tables
  
  
**3. Modeling** 
- Train/Test split
- Making a Linear regression models: 
  - first model: Ordinary Least Squares (OLS) 
    - via **sklearn** package
  - second model: Ordinary Least Squares (OLS) 
    - via **statsmodels** package
  - third model: multiple linear regression
    - all continuous features included
  - fourth model: avoiding redundancy
    - remove some features that are strongly correlated to each other
	- *ex: citric acid and pH closely related, chose to only keep pH*

**4. Evaluating and concluding** 
- Reflection 
  - see notebook
- Which model was best?
  - model 3 scored best, but model 4 is more elegant
- Other regression algorithms
  - statsmodels documentation [link](https://www.statsmodels.org/dev/examples/index.html#regression)
  - sklearn linear models [link](https://scikit-learn.org/stable/modules/linear_model.html#linear-model)


### Question-Answers

> Throughout this case study, questions will be asked in the markdown cells. Try to answer these yourself in a simple text file when they come up. Most of the time, the answers will become clear as you progress through the notebook.

*I did not keep up with this here, see notebook*