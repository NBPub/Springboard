 - precision: tp / (tp+fp)
   - model's ability to avoid false positive labels
 - recall: tp / (tp+fn)
   - model's ability to find all positive samples
 - Receiver Operating Characteristic Curve
   - performance of binary classifier at different discrimination thresholds
   - true positive rate (*sensitivity*) vs false positive rate (*1-specificity*) AKA true negative rate
   
---

 - [pandas timeseries](https://pandas.pydata.org/docs/getting_started/intro_tutorials/09_timeseries.html)
 
---

Dataset was pulled from 1st July and has users with the following information. 
Users are "retained" if they took a ride within the previous 30 days.

|  **Feature**  | **Description** |
|--------------:|----------------:|
|  city | city this user signed up in |
|  phone | primary device for this user |
|  signup_date | date of account registration; in the form ‘YYYYMMDD’ |
|  last_trip_date | the last time this user completed a trip; in the form ‘YYYYMMDD’ |
|  avg_dist | the average distance in miles per trip taken in the first 30 days after signup |
|  avg_rating_by_driver | the rider’s average rating over all of their trips |
|  avg_rating_of_driver | the rider’s average rating of their drivers over all of their trips |
|  surge_pct | the percent of trips taken with surge multiplier > 1 |
|  avg_surge | The average surge multiplier over all of this user’s trips |
|  trips_in_first_30_days | the number of trips this user took in the first 30 days after signing up |
|  ultimate_black_user | `TRUE` if the user took an Ultimate Black in their first 30 days; `FALSE` otherwise |
|  weekday_pct | the percent of the user’s trips occurring during a weekday |

 - preprocessing | [model data](./model_data.parquet)
   - dropped rating features, date features
   - one-hot-encoded city and phone
   - combined surge features into one by multiplying together
   - standardized numerical features
 - selected from a few classification models via scikit-learn and cross-validation
 - one round of hyperparameter tuning with RandomizedSearchCV
 - final model: tuned GradientBoostingClassifier
 
 
![model selection]("https://raw.githubusercontent.com/NBPub/Springboard/main/27.2.2%20Take%20Home%20Challenge/model_selection_CV.png")

![final model results]("https://raw.githubusercontent.com/NBPub/Springboard/main/27.2.2%20Take%20Home%20Challenge/final_model_results.png")

![feature importance, sample predictions]("https://raw.githubusercontent.com/NBPub/Springboard/main/27.2.2%20Take%20Home%20Challenge/results_interpretation_table_img.png")