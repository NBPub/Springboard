Course by Michele Vallisneri

Other Resources
 - [gapminder.org](https://www.gapminder.org/)
 - plotting libraries I haven't used
   - plotnine (similar to R's ggplot)
   - Bokeh (web interactive, similar to Plotly?)


## Statistical Inference

 - **Bootstrapping** AKA bagging
   - example, poll 100 people to grade some political figure 1-10
   - cannot describe sampling distribution
 -  therefore estimate uncertainty of mean value by generating large family of samples from the existing ones
   - sample with replacement from data, evaluate mean, repeat a lot
   - from resulting distribution, can get confidence interval of value
     - is sample representative?
   - `scipy.stats` package used to evaluate statistically
     - did not follow this exercise, drew a bimodal distribution with two peaks
 - **Hypothesis Testing**
   - Null Hypothesis [Test](https://en.wikipedia.org/wiki/Null_hypothesis)
     - observe statistic from data
	 - compute sampling distribution of statistic under null hyptohesis (no relationship)
	 - quantile of observed statistic provides **P value**, likelihood that result would happen given null hypothesis
	   - what should cutoff value be for significance?
   - can bootstrapping be applied to compute **P values**?
     - bootstrap samples only represents true distribution, not distribution under null hyptohesis
	 - would need to modify values
	   - problem specific, not always straightforward and/or possible

## Statistical Modeling

 - fitting models
   - examples with  **[statsmodels](https://www.statsmodels.org/stable/index.html)** package
     - as with most examples start with using overall mean to set baseline for performance
	 - model formula uses **R** syntax?
	 - define explanatory, target variables
	   - can also allow interactions
 - goodness of fit
   - statsmodels allows plotting of residuals and things below from *model* object
     - OLS linear regression, [docs](https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html#Ordinary-Least-Squares)
     - see summary of a model
   - mean squared error
   - correlation coefficients
   - F statistic
     - measures how much, on average, each parameter contributes to the growth of R<sup>2</sup>, compared to a hypothethical *random* model parameter
	   - if parameter has no contribution `F > 1`
	   - larger F indicates more explanation from parameter
	 - helps caution against overfitting
   - ANOVA tables
     - **AN**alysis **O**f** **VA**riance
	   - `df` degrees of freedom, number of parameters for a model or datapoints/parameters for **Residual** row
	   - `sum_sq`, `mean_sq` sum and mean squared error
	   - `F` F statistic
	   - `PR(>F)` p-value for a hypothetical model with same number of parameters but all *random* terms
	 - can easily compare different parameters' contributions
 - cross-validation
   - plenty of notes in previous learnings
 - logistic regression
   - regression for categorical or binary responses
     - used in place of ordinary least squares (OLS) for whether smoking impacts life expectancy, `yes/no`
   - transform unconstrained linear model with logistic transformation
     - `exp(y) / (1+exp(y))`
     - response bounds: `[-inf, inf]` to `[0,1]` 
 - bayesian inference
   - estimate using entire probability distributions, not just population parameters
     - relatively intense computation, became more popular with computing
   - have established probabilities, make observations, use results to update *prior to posterior* probabilities
   - **[pymc3](https://www.pymc.io/welcome.html)** package used 
     - > probabilistic programming library for Python that allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods
	 - see also: [ArviZ](https://python.arviz.org/en/latest/index.html), exploratory analysis of Bayesian models, diagnose and visualze Bayesian inference
  - coinflip example, 
    - prior: chance of getting heads with bias (40-80%) 
    - model observations with probability distributions, used Binomial for **k** events over **n** trials with each event having **p** probability
	  - *ex: Normal, Binomial*
	- "sample posterior" - generate population parameters approximately distributed according to posterior. **trace**  
