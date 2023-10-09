## 14b Time Series Analysis

*see also*
 - [14 Supervised Learning]()
 - [15 Unsupervised Learning]()
 - [16 Feature Engineering]()
 - [16b Feature Selection]()
 - [18 Model Evaluation]()

### Notes

 - **ARIMA** = auto-regressive integrated moving average
   - general class of time series models
     - most others are special cases of ARIMA
     - systematic set of rules for choosing with case should be used for predicting a given time series
   - therefore, only really need to consider **ARIMA**
     - dicsussed in second course reading (see links below)
     - [seasonal vs nonseasonal, intro slides](https://people.duke.edu/~rnau/Slides_on_ARIMA_models--Robert_Nau.pdf)



### Additional Resources

 - [Robert Nau - Statistical Forecasting, notes on regression and time-series analysis](https://people.duke.edu/~rnau/411home.htm)
   - detailed notes from Duke course
   - other course reading sourced from here
     - [table with selection guidance for data transformations, forecasting models](https://people.duke.edu/~rnau/whatuse.htm)
	   - deflation by price index, deflation at fixed rate, Logarithm, First difference, Seasonal difference,
	   Seasonal adjustment, Random walk, Linear trend, Moving average, Exponential smoothing, Linear exponential
	   smoothing (Brown's, Holt's), Seasonal random walk, Winter's season smoothing, Multiple regression, **ARIMA**
 - Sites from other course reading:
   - [Time Series Tutorial in R](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/),
     [Guide to create forecast](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
   - [Rules for identifying ARIMA models](https://people.duke.edu/~rnau/arimrule.htm)  
     - R tutorial said to have more thorough discussion
   - [Notebook from article](https://github.com/seanabu/seanabu.github.io/blob/master/Seasonal_ARIMA_model_Portland_transit.ipynb)
     - note that this was written in 2016 and packages have seen significant updates