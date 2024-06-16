# Contents

 - [Ch 1,2](#chapters-12)
   - categorical data and percentages
   - summarizing and communicating numbers
 - [Ch 3](#chapter-3), [Ch 4](#chapter-4),  [Ch 5](#chapter-5)
   - populations and measurement, what causes what?, modeling with regression
 - [Ch 6](#chapter-6) | **Algorithms, Analytics, and Prediction**
  - [titanic example](#titanic-example)
  - [overfitting](#over-fitting)
  - [regression  models](#regression-models)
  - [other techniques](#other-techniques)
    - Random Forests, SVM, Neural Networks, Nearest Neighbors
  - [challenges](#challenges)
    - four main concerns
  - [artificial intelligence](#artificial-intelligence)
    - narrow vs general
 - [Ch 7](#chapter-7)
   - estimates and intervals
 - [Ch 8,9](#chapter-89)
   - Probability, together with Statistics
 - [Ch 10](#chapter-10)
   - answering questions and claiming discoveries
 - [Ch 11](#chapter-11)
   - Bayesian inference, statistics
   - [R van de Schoot](https://osf.io/wdtmc/download)
 - [Ch 12,13, 14](#chapter-121314)
   - how things go wrong, what we can do better
   - conclusion
   
**The Art of Statistics and Learning from Data** - *David Spiegelhalter*
<br> ISBN: `978-0-241-25875-0`

*chapter takewaway notes stored locally*

## Chapters 1,2

## Chapter 3

## Chapter 4

## Chapter 5

## Chapter 6

**Algorithms, Analytics, and Prediction**

 - two broad tasks:
   - **Classification**
     - .
   - **Prediction**
     - .	 
 - narrow vs general AI
 - finding patterns
 
### Titanic Example

 - classification and prediction
 - classification tree

### Assessing Performance

 - define all the terms, provide "elegant interpretations"
 - Accuracy, Sensitivity, Specificity
 - **ROC Curve**
   - ...
 - **Mean Squared Error (MSE) AKA Brier Score**
   - provide context to a model's MSE by calculating comparing to MSE from simply using (long-term) average
     - weather forecast score vs using historical records
     - Titanic survival algorithm vs using mean survival rate
   
### Over-fitting

 - model becomes specific to training set, will perform worse overall. model is fitting noise, not signal
   - Titanic example: 
     - many branches added to descision tree
	 - training set accuracy improved, test set accuracy had minor drop (MSE increase more clear)
 - **bias/vairance trade-off**
   - over-fitting seeks to reduce bias at a cost of uncertainty in the estimates
 - protect against over-fitting with **cross-validation** during model construction
   - cycle training/test sets based on chosen proportions of data
   - test model performance, tune parameters *ex: various numbers of brances*
   
### Regression Models

 - classification trees vs regression models
   - trees make rules to identify groups with similar outcomes
   - regression provides varying weight to specific features
 - logistic regression could be fit to Titanic data, example trained one with *[boosting](https://en.wikipedia.org/wiki/Gradient_boosting)*
   - model training gives extra weight to incorrectly classified cases with each iteration of cross-validation**
 - various parameters contribute positive or negative scores for survival prediction
   - each parameter's contribution can be interpreted as its importance for predicting the target variable
 - more complex models, too
   - non-linear regression, *ex: __kinetics*
   - [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) technique, *scikit-learn [docs](https://scikit-learn.org/stable/modules/linear_model.html#lasso)*
   
### Other Techniques

Brief model descriptions, see wikipedia and scikit-learn links for more.

 - [Random Forests](https://en.wikipedia.org/wiki/Random_forest#See_also) | [docs](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
   - large number of trees with classifications, majority "vote" for final decision *[bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)*
 - [Suport Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) | [docs](https://scikit-learn.org/stable/modules/svm.html#support-vector-machines)
   - linear combinations of features that split different outcomes
 - [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network) | [docs](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised), [unsupervised](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#neural-networks-unsupervised)
   - layers of nodes, depending on previous layers by optimized weights
     - > series of logistic regressions piled on top of each other . . . weights are learned. . . like random forests
   - many-layered models are "deep-learning"
 - [K-nearest-neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) | [docs](https://scikit-learn.org/stable/modules/neighbors.html)
   - classifies according to majority outcome among close cases 
  
***Applying to Titanic . . .***

 - model accuracy not good measure of performance
   - recall over-fit model had similar accuracy score (81%) to "properly fit" model, but higher MSE
   - models listed above had similar accuracy to simple rule: females survive, males don't
 - can use area under ROC or MSE
   - **Random Forest** had best (highest) ROC integral
   - **Simple Classification Tree** had best (lowest) MSE
   - no real winner, are more complex models worth it here?
 - negative consequences of "black-box" model <-- more likely with increasing complexity
   - high effort to implement and upgrade
   - uncertain of a prediction's confidence
   - cannot investigate implicit, systematic biases
     - [How to make a racist AI without really trying](http://blog.conceptnet.io/posts/2017/how-to-make-a-racist-ai-without-really-trying/)

### Challenges

Four main concerns

 - lack of robustness
   - ...
 - not accounting for statistical variability
   - ...
 - implicity bias
   - ...
 - lack of transparency
   - ...
   
### Artificial Intelligence

 - narrow vs general

## Chapter 7

## Chapter 8,9

## Chapter 10

## Chapter 11

## Chapter 12,13,14