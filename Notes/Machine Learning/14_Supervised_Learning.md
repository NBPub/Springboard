## 14 Supervised Learning

### Resources

 - Videos
   - [Classification, kNN, Cross-Validation, Dimensionality Reduction](https://youtu.be/uhHqzqj5Pio?t=495)
     - [part 2 timestamp](https://youtu.be/uhHqzqj5Pio?t=2512)
	 - Harvard CS109 lecture ~72 mins
   - [Introduction to Machine Learning](https://youtu.be/h0e2HAPTGF4?t=100)
     - MIT OCW, Introduction to Computational Thinking and Data Science
   - [Bias and Regression](https://youtu.be/sf_xR4oWEgU)
     - Harvard CS109 lecture, ***find different video? parts of presentation / boardwork not shown***
   - [Decision Trees, Random Forests](https://youtu.be/AI8VWsQTMFk)
     - Harvard CS109 lecture
   - [Using Random Forests](https://youtu.be/6O4kASc-SDE)
     - lecture from PyData conference, only first 20 mins part of assignment
   - [Ensemble Methods - Bagging and Boosting](https://youtu.be/ccqNeWQJC0c)
     - Harvard CS109 lecture
   - [Suppport Vector Machines 1/3](https://youtu.be/efR1C6CvhmE)
     - StatQuest video, illustrative
   - [SVM Performance Evaluation, PR ROC](https://youtu.be/npnLohAiISc)
     - Harvard CS109 lecture
   - [Short SVM with Polynomial Kernel Visualization](https://youtu.be/3liCbRZPrZA)
     - Consider this a visual demonstration of the kernel trick in SVM
   - [Time Series Analaysis](https://youtu.be/Prpu_U5tKkE)
     - intro lecture from Jordan Kern

### Notes

 - Logistic Regression
 - Descision Trees
 - Random Forest
 - Gradient Boosting
	 

#### Imbalanced Data | [link](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

<details><summary>. . . click to expand . . .</summary>

*ex: Classification problem (`A` vs `B`) where source data contains much more `A` cases, therefore may end up overfitting to `A`.*

**Accuracy Paradox [wiki](https://en.wikipedia.org/wiki/Accuracy_paradox)**

> a simple model may have a high level of accuracy but be too crude to be useful. For example, if the incidence of category A is dominant, being found in 99% of cases, then predicting that every case is category A will have an accuracy of 99%. [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) are better measures in such cases.

simple model, in this case, could be as simple as *always* outputting `A` 

  1. Collect more data
Useful if possible!

  2. Change performance metric
As mentioned above, accuracy isn't ideal. Some ideas mentioned:
	- **Confusion Matrix**
	- **Precision and Recall**
	- **F1 Score (weighted average of above)**
	- **Kappa / Cohen's kappa**
	- **ROC Curves**

  3. Resample dataset | [wiki](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)
Artificially change dataset to impart more balance on the model, either:
 - **over-sample**: add copies of the under-represented class `B`
   - better with less data
 - **under-sample**: delete instances of over-represented class `A`
   - better with more data
 - *Advice*
   - Test random and non-random/stratified sampling schemes
   - Test different resampled ratios (don't only try 1:1 for binary problem)

  4. Generate Synthetic Samples, **SMOTE**
  - can sample empirically
  - utilize Naive Bayes
  - various systematic algorithms . . .
    - **SMOTE**, synthetic minority over-sampling technique
	  - [original 2002 publication](https://arxiv.org/pdf/1106.1813.pdf)
	  - Randomly sample attributes from instances in the minority class to generate synthetic samples.
	- see *scikit-learn-contrib* package, [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)

  5. Change algorithm
Always use cross-validation to try various models. Author suggests decision trees often perform well with imbalanced data.
*suggestions:* **C4.5, C5.0, CART, Random Forest**
  
  6. Penalized Classification
Use *penalized* version of a classification algorithm that imposes additional cost for mistakes on the minority class. 
Trial and error with a variety of *penalty schemes* often required.

  7. Change Perspective / Get Creative
*ex: Instead of detecting rare events,* **Anomaly Detection,** *for a particular problem, consider as* **Change Detection**.
*could be useful for a security camera or something.*

Break down problem into more tractable, smaller problems. Get inspiration from other problems.

</details> 
 
### kNN | [Harvard CS109 lecture 09](https://youtu.be/uhHqzqj5Pio?t=495)

<details><summary>. . . click to expand . . .</summary>

**Basic Idea**

**Training vs Testing complexity**

**Bias, Variance, selection of `k`** 

One nearest neighbor is low bias, high variance. With each new training point, more boundaries are added.

As neighbors are increased, bias is introduced and variance decreases. Smoother boundaries, may not be exact.

**Optimizing `k`, distance function, voting parameters via Cross-Validation**

AKA **Hyper-Parameter evaluation**. 5 and 10-fold are typical choices for CV. Depends on size of dataset and choice of classifier.
For example, smaller datasets can't fold as much. Make sure test-data is untouched until final evaluation.

**Distance Calculations**, Training Classifier

> choice of feature is one of most important things in classification, self-driving car example . . . many different "detectors" to provide many different features to provide best decisions


 - very basic for image classifier: pixel-by-pixel distance distance
   - euclidean
   - L1
   - L2
 - SIFT
   - Rotation, Scale Invariant
   
Feature additions may help training accuracy, but then encounter [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), 
the space is now too far and neighbors are far apart. Therefore . . .

</details> 

### Dimensionality Reduction

**Idea:** Bring down dimensionality of vectors generated from pixels while preserving the distance between neighbors.
*watch this part again as a review [timestamp](https://youtu.be/uhHqzqj5Pio?t=3814)

Also useful for compression and visualization (PCA music example).

- techniques
  - linear models *why are they more commonly used?*
  - non-linear methods
  
**PCA** | [previous notes](https://github.com/NBPub/DataScienceGuidedCapstone#principal-component-analysis-1)
  
 - Post-Office Handwriting Recognition Example
 - Acoustic patterns in music: *project to expand on my library, explore particular genre grouping*



**Multi-Dimensional Scaling (MDS)** | [wiki](https://en.wikipedia.org/wiki/Multidimensional_scaling) | [other](https://dept.stat.lsa.umich.edu/~jerrick/courses/stat701/notes/mds.html#:~:text=Multidimensional%20Scaling%20(MDS)%20is%20a,to%20find%20patterns%20or%20groupings.)

... notes ...

### Metrics

 - Confusion Matrix for Classification
   - Precision, Recall, F1 Score
   - scikit learn --> `classification_report`
 - ROC curve for Logistic Regression
   - see/take statistics book notes
 - Hyperparameter tuning 
   - many examples elsewhere (k for kNN, alpha for ridge/lasso regression)
   - Grid Search CV, [guided capstone notes](https://github.com/NBPub/DataScienceGuidedCapstone#hyperparameter-search-using-gridsearchcv)
   - Randomized Search CV
   
### Preprocessing, Pipelines

 - 

### Regularized Regression

#### Ridge

#### Lasso


