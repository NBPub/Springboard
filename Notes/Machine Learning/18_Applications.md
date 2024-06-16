## 18 ML Evaluation and Optimization

*see also*
 - [14 Supervised Learning]()
 - [14b Time Series Analysis]()
 - [15 Unsupervised Learning]()
 - [16 Feature Engineering]()
 - [16b Feature Selection]()

### Overview

 - Evaluation metrics
   - Regression
     - R2 vs adj R2 vs MAE vs RMSE
   - Classification
     - accuracy, class imbalance
	 - confusion matrix
	   - precision, recall, f1 score
	   - ROC-AUC, true positive/negative rates
	 - log loss
 - Model optimization, Hyperparameter Tuning
   - Grid Search, Random Search
     - how can Random Search outperform Grid Search? why is grid search usually not best choice?
   - [Bayesian Optimization](#bayes-optimization)

### Resources

#### Course Reading

 - Hyperparameter Tuning article
   - Techniques
     - *classical:* Grid Search, Random Search
	   - Bergstra and Bengio paper showed Random usually did as well as Grid with at least 60 points
	 - *"smarter methods"*
	   - tend to be iterative and less parallelizable 
	   - good performance often requires an outter tuning step of the optimization method
	   - [Snoek, Larochelle, and Adams paper](https://papers.nips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html)
	     - Gaussian process to model response function and *Expected Improvement*, then used to determine next set of hyperparameters
	   - [Hutter, Hoos, and Leyton-Brown paper](https://proceedings.mlr.press/v32/hutter14.html)
	     - train random forest to approximate response surface, which is then used to sample optimal regions
		 - dubbed **SMAC**, linked below. article author suggests it works well for categorical hyperparameters
	   - **derivative-free optimzation**
	     - employ heuristics to determine where to sample next. see **Nelder-Mead** method
		   - scipy provides Nelder-Mead in its [`optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) function, [demonstration article](https://aicorespot.io/how-to-leverage-nelder-mead-optimization-in-python/)
	   - **Bayes-optimzation**
	     - see stuff below and linked packages
	   - **random forest smart tuning**
	     - similar to Bayes in that the response surface is modeled with another function, and then that is used to sample more points
   - see paper notes for "nested" cross-validation/hyperparameter tuning
   - Packages
     - [Hyperopt](https://github.com/hyperopt/hyperopt) Bayes Optimization using tree-based parzen estimators
	 - [SMAC aka SMAC3](https://github.com/automl/SMAC3) Bayes optimization, Random forest
	 - *not maintained?*: Spearmint, hypergrad

#### Bayes Optimization

read/watch more. why are black box approaches good for hyperparameter tuning?

 - Bayes Optimization
   - see [package](https://github.com/bayesian-optimization/BayesianOptimization)
   - [lecture](https://www.youtube.com/watch?v=C5nqEHpdyoE)
   - [short explanation](https://www.youtube.com/watch?v=M-NTkxfd7-8)
 - Background?
   - [stat quest - bayes theorem](https://www.youtube.com/watch?v=9wCnvr7Xw4E)
   - [stat quest - naive bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)


