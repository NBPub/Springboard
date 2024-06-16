# Feature Selection Notes

**Contents**

 - model based, sequential scikit-learn guide
 - permutation scikit-learn guide
 - SHAP *external package*
 - LIME *external package*

### Model Based and Sequential Feature Selection 

[link](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#selecting-features-based-on-importance)

#### Approach 1: Ridge regression estimator coefficients

 - use **[RidgeRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV)** to fit a model to data
   - examine coefficients for each feature
 - use SelectFromModel to choose features above a certain threshold
   - top 2 features selected, threshold set to be higher than 3rd highest coefficient
 -**my notes**
   - can use variety of estimators to evaluate feature importance/coefficients
   - ex: RandomForest
   
#### Approach 2: Sequential Feature Selection

  - "greedy" procedure, choose best new features at each iteration based on CV score
    - start with 0 and choose the single best feature, repeat until desired number of features
	- can also perform in the reverse direction, start with all and drop worst.
  - **[SquentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel)** used
    - `direction=backward`, `tol<0` for reverse search
	- requires an estimator that has feature_importance or coefficient attributes

### Permutation Feature Importance

[link](https://scikit-learn.org/stable/modules/permutation_importance.html), 
**[sklearn.inspection.permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance)**

 - can be used for any fitted estimator when data is tabular
 - useful for non-linear or opaque estimators
 - *how much does model score decrease when a single feature is effectively removed?*
   - break relationship between feature and target
   - drop in model score indicative of feature importance to model
 - model agnostic, can be calculated many times with different permutations
   - using multiple scorers/metrics more efficient than running several times with different scorers/metrics
   - metric choice particularly important for imbalanced classification problems
 - can lose information with intercorrelated features, see feature clustering strategy
   - [user guide](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py)
> Permutation importances can be computed either on the training set or on a held-out testing or validation set. Using a held-out set makes it possible to highlight which features contribute the most to the generalization power of the inspected model. Features that are important on the training set but not on the held-out set might cause the model to overfit.

**vs Decision Trees** [see more](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)
>  Impurity is quantified by the splitting criterion of the decision trees (Gini, Log Loss or Mean Squared Error). However, this method can give high importance to features that may not be predictive on unseen data when the model is overfitting. Permutation-based feature importance, on the other hand, avoids this issue, since it can be computed on unseen data.

