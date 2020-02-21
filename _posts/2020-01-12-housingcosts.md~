---
layout: post
author: roberto
---

## INTRODUCTION

In this project, the monthly housing cost is predicted by applying several supervised machine learning models available in the Python scikit-learn library. A potential home buyer could benefit from this machine learning application by providing the user a cost estimate of a property before closing the deal. Government agencies could use this application to identify homes in financial distress in a given region, thereby influencing their policy decisions.

## DATA
The data set used is called the Housing Affordability Data System (HADS) which consists of individual datasets spanning the years 1985 to 2013. The data sets are available for download here: [American Housing Survey: Housing Affordability Data System](https://www.huduser.gov/portal/datasets/hads/hads.html). The HADS data are sourced from the American Housing Survey (AHS) national sample microdata and the AHS metropolitan sample microdata. Therefore, more details can be obtained by referring to the AHS data. Each row in the data is an observation of a housing unit. The features fall under 4 categories:

* Cost measures: Utility costs, mortgage payments, HOA fees
* Housing unit characteristics: number of bedrooms, year built, location, structure type
* Household characteristics: income, number of people, rent/owner status
* Local market conditions: median income, fair market rent, poverty level income

To avoid data leakage issues, the features corresponding to cost measures are not fed into the machine learning models. The table below lists the features used to predict the monthly housing cost.

<table style="width:50%">
  <tr>
    <th>Feature Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>AGE</td>
    <td>Age of head of household</td>
  </tr>
  <tr>
    <td>BEDRMS</td>
    <td>Number of bedrooms in unit</td>
  </tr>
  <tr>
    <td>FMR</td>
    <td>Fair market rent</td>
  </tr>
  <tr>
    <td>INCRELAMIPCT</td>
    <td>Household income relative to area median income (percent)</td>
  </tr>
  <tr>
    <td>IPOV</td>
    <td>Poverty income</td>
  </tr>
  <tr>
    <td>LMED</td>
    <td>Area median income (average)</td>
  </tr>
  <tr>
    <td>NUNITS</td>
    <td>Number of units in building</td>
  </tr>
  <tr>
    <td>PER</td>
    <td>Number of persons in household</td>
  </tr>
  <tr>
    <td>ROOMS</td>
    <td>Number of rooms in unit</td>
  </tr>
  <tr>
    <td>VALUE</td>
    <td>Current market value of unit</td>
  </tr>
  <tr>
    <td>REGION</td>
    <td>Census region</td>
  </tr>
  <tr>
    <td>YEAR</td>
    <td>Year of housing survey</td>
  </tr>
  <tr>
    <td>FMTBUILT</td>
    <td>Year unit was built</td>
  </tr>
  <tr>
    <td>FMTASSISTED</td>
    <td>Assisted housing</td>
  </tr>
  <tr>
    <td>FMTSTRUCTURETYPE</td>
    <td>Structure type</td>
  </tr>
  <tr>
    <td>FMTMETRO</td>
    <td>Central city/suburban status</td>
  </tr>
  <tr>
    <td>FMTZADEQ</td>
    <td>Adequacy of unit</td>
  </tr>
</table> 

The HADS data files provide more features which are not included in the following analysis. These discarded variables are related to either LMED or INCRELAMIPCT.

Note that the features starting with string FMT (for formatted) are categorical variables. Therefore one-hot encoding was applied to these, ending up with with a 29-dimensional feature space. Following various data clearning/wrangling steps, the data set ends up having 486,785 observations.

For the analysis that follows, Python code was written in various Jupyter notebooks. The notebook files can be found [here](https://github.com/lambertgarrido/spb-cap)

## MODELS

Two supervised learning algorithms are applied to predict the monthly housing cost (column name is ZSMHC in data files): ridge linear and random forest. The data is split into a test and training set. The training set is used to fit the model and the test data is used to calculate the model’s score. The table below lists the scores for the different models<sup><a href="#fn1" id="ref1">1</a></sup>:

<table style="width:30%">
  <tr>
    <th>Model</th>
    <th>Score</th>
  </tr>
  <tr>
    <td>Linear Ridge</td>
    <td>0.554</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.606</td>
  </tr>
</table>

### Ridge Linear Model

This model is applied due to its simplicity and easy interpretability of the output coefficients. At first, the hyperparameter alpha is tuned over the range from 0.01 to 100. The best value was found to be an alpha of 10. However, the score value was roughly the same across all values of alpha used (score was about 0.57). In order to compare the relative importance of the coefficients, the data was scaled to mean 0 and variance 1, and the train scaled data was fitted to a linear ridge instance with alpha set to 10. The corresponding coefficients for the scaled data are plotted below.

![coefficients scaled linear model]({{ site.baseurl }}/assets/images/coeffs_scaled_linear.png "coefficients scaled linear model")

### Random Forest
The next model used is the random forest. This is a popular model that can account for nonlinear effects in the data, unlike the ridge linear model. The training data is fed into an instance of a random forest model with default parameters. The score on the test data is around 0.61. Next the training data set is used for 3-fold cross validation, obtaining scores in the 0.60-0.62 range. The feature importance values are plotted below. The importance values add up to one. The higher a feature’s importance value, the more important that feature is.

![random forest importance]({{ site.baseurl }}/assets/images/importance_rf.png "random forest importance")

It is to be noted that parameter tuning was performed over the following parameter grid<sup><a href="#fn2" id="ref2">2</a></sup>:

<table style="width:30%">
  <tr>
    <th colspan="2">Grid 1</th>
  </tr>
  <tr>
    <th>Parameter</th>
    <th>Values</th>
  </tr>
  <tr>
    <td>bootstrap</td>
    <td>True, False</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>10, 12, 14, 16, 18, 20, 22, 24, 26,  28, 30, None</td>
  </tr>
  <tr>
    <td>max_features</td>
    <td>auto, sqrt, log2</td>
  </tr>
  <tr>
    <td>min_samples_leaf</td>
    <td>1, 2, 4</td>
  </tr>
  <tr>
    <td>min_samples_split</td>
    <td>2, 5, 10</td>
  </tr>
  <tr>
    <td>n_estimators</td>
    <td>20, 50, 100, 150, 200</td>
  </tr>
</table>

Instead of scanning the entire parameter space, 100 parameter combinations were chosen at random and the score was evaluated over these points. Over these 100 points, the score varied from 0.26 up to 0.62. The best score obtained was 0.619 with best parameter values set to:

<table style="width:30%">
  <tr>
    <th colspan="2">Best Parameters Grid 1</th>
  </tr>
  <tr>
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>bootstrap</td>
    <td>False</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>18</td>
  </tr>
  <tr>
    <td>max_features</td>
    <td>sqrt</td>
  </tr>
  <tr>
    <td>min_samples_leaf</td>
    <td>1</td>
  </tr>
  <tr>
    <td>min_samples_split</td>
    <td>10</td>
  </tr>
  <tr>
    <td>n_estimators</td>
    <td>200</td>
  </tr>
</table>

Next, a second grid around the best parameter values listed above was created: 

<table style="width:30%">
  <tr>
    <th colspan="2">Grid 2</th>
  </tr>
  <tr>
    <th>Parameter</th>
    <th>Values</th>
  </tr>
  <tr>
    <td>bootstrap</td>
    <td>False</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>16, 18, 20</td>
  </tr>
  <tr>
    <td>max_features</td>
    <td>sqrt</td>
  </tr>
  <tr>
    <td>min_samples_leaf</td>
    <td>1, 2</td>
  </tr>
  <tr>
    <td>min_samples_split</td>
    <td>2, 5, 10</td>
  </tr>
  <tr>
    <td>n_estimators</td>
    <td>100, 200, 500</td>
  </tr>
</table>

The score was evaluated for all the points in this grid, with scores ranging from 0.616 up to the best score of 0.619. The best score parameters turned out to be:

<table style="width:30%">
  <tr>
    <th colspan="2">Best Parameters Grid 2</th>
  </tr>
  <tr>
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>bootstrap</td>
    <td>False</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>20</td>
  </tr>
  <tr>
    <td>max_features</td>
    <td>sqrt</td>
  </tr>
  <tr>
    <td>min_samples_leaf</td>
    <td>1</td>
  </tr>
  <tr>
    <td>min_samples_split</td>
    <td>10</td>
  </tr>
  <tr>
    <td>n_estimators</td>
    <td>500</td>
  </tr>
</table>

For the results that follow in the discussion section, n_estimators (number of trees) was set to 100 and the other parameters were set to their default values. These settings result in a score of 0.61 which is quite close to the best scores of 0.62 obtained from the parameter grid searches.

Finally, the plot below shows the average fit time versus n_estimators parameter for the points from the second grid. Note the positive linear relationship. For our purposes, a 100-tree random forest might not give the best score (although comes close), but it does run about 5 times faster than a 500-tree random forest.

![time vs number of trees]({{ site.baseurl }}/assets/images/time_vs_trees.png "time vs number of trees")


### Support Vector Machine
Finally, a support vector machine (SVM) model is used to predict whether or not a housing unit monthly cost is in the top 10%. For this part, a new column was created indicating whether or not a housing unit is in the top 10% of monthly housing cost for a given survey year. This column becomes the variable that the support vector machine needs to predict. The results of this model will be presented in the next section. Again the training data is used to fit the model and the test data is used to quantify the prediction accuracy.

## DISCUSSION

In the table below, the top 10 features are listed for the ridge linear and random forest model. Both models agree that the housing unit market value (feature name VALUE) is the most important factor in determining the monthly housing cost.

For random forest, the second most important predicter is AGE (age of head of household), followed by INCRELAMIPCT (income). Next most important predictors for this model are variables that measure local market conditions IPOV (poverty level income), FMR (fair market rent), and LMED (median income). The least important variables include the survey year (YEAR) and variables describing the housing characteristics (ROOMS, BEDRMS, PER (number of people)).

For the linear model, the same set of variables (except for BEDRMS) are considered important; however the order of importance is different than the random forest model. Whether or not the housing unit was built in 2000-2009 was an important predictor for the linear model.

<table style="width:50%">
  <tr>
    <th colspan="2">Ridge Linear</th>
    <th colspan="2">Random Forest</th>
  </tr>
  <tr>
    <td>Feature</td>
    <td>Coefficent</td>
    <td>Feature</td>
    <td>Importance</td>
  </tr>
  <tr>
    <td>VALUE</td>
    <td>0.396</td>
    <td>VALUE</td>
    <td>0.483</td>
  </tr>
  <tr>
    <td>IPOV</td>
    <td>0.301</td>
    <td>AGE</td>
    <td>0.120</td>
  </tr>
  <tr>
    <td>PER</td>
    <td>0.179</td>
    <td>INCRELAMIPCT</td>
    <td>0.086</td>
  </tr>
  <tr>
    <td>AGE</td>
    <td>0.173</td>
    <td>IPOV</td>
    <td>0.066</td>
  </tr>
  <tr>
    <td>FMR</td>
    <td>0.146</td>
    <td>FMR</td>
    <td>0.064</td>
  </tr>
  <tr>
    <td>YEAR</td>
    <td>0.135</td>
    <td>LMED</td>
    <td>0.054</td>
  </tr>
  <tr>
    <td>INCRELAMIPCT</td>
    <td>0.125</td>
    <td>ROOMS</td>
    <td>0.027</td>
  </tr>
  <tr>
    <td>ROOMS</td>
    <td>0.104</td>
    <td>YEAR</td>
    <td>0.023</td>
  </tr>
  <tr>
    <td>LMED</td>
    <td>0.099</td>
    <td>BEDRMS</td>
    <td>0.012</td>
  </tr>
  <tr>
    <td>FMTBUILT_2000-2009</td>
    <td>0.088</td>
    <td>PER</td>
    <td>0.007</td>
  </tr>
</table> 

Next, the percent error is calculated for each model. As a baseline for comparison, a third model is defined which predicts the sample average, regardless of the feature values. Such a model would have a score of 0. The average percent error is plotted across survey years for linear ridge, random forest, and baseline model. The random forest has by far the smallest average percent error. For earlier survey years, the linear model performs better than the baseline model. For later survey years, the baseline model outperforms the linear model.

![average percent error]({{ site.baseurl }}/assets/images/avg_percent_error_all.png "average percent error")

The plot below lists only the linear ridge and random forest models:

![average percent error rf linear only]({{ site.baseurl }}/assets/images/avg_percent_error_linear_rf.png "average percent error rf linear only")

Below are boxplots of the percent error for random forest model (full plot and zommed in version):

![box plot rf]({{ site.baseurl }}/assets/images/percent_error_boxplot_rf.png "box plot rf")

![box plot rf zoomed]({{ site.baseurl }}/assets/images/percent_error_boxplot_rf_zoomedin.png "box plot rf zoomed")

For the support vector machine, a binary variable is defined if the housing unit is above the 90th percentile for the monthly housing cost. The accuracy, precision, and recall are calculated for this variable.<sup><a href="#fn3" id="ref3">3</a></sup> These values are listed below.


<table style="width:20%">
  <tr>
    <td>Accuracy</td>
    <td>0.925</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>0.747</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.421</td>
  </tr>
</table>

These values are plotted versus survey year below. Note that for 1985 and 1987, the second column of confusion matrix consists of zeroes. Therefore the precision is not defined and has been plotted as zero.

![svm]({{ site.baseurl }}/assets/images/svm.png "svm")


## FURTHER WORK

To improve the predictive power of these machine learning models, the following items will be considered:

* During the data cleaning steps, a large fraction of observations had no defined VALUE column. Therefore, these rows were dropped. Instead, these missing values can be imputed based on similar entries or via by averaging over the known values.
* Consider other data sets that contain more precise location data. The current data set only specifies the 4 census regions and whether or not the housing unit is in a central city. It is well known that housing prices are higher in coastal cities and cheaper in the interior of the United States. Perhaps knowing a more precise location will help the model learn better.
* Try hyperparameter tuning/cross validation for support vector machine model.

## CONCLUSION
In conclusion, a linear ridge model and random forest model were used to predict the monthly housing cost. Both models determined that the housing unit market value is an important indicator of the monthly housing cost. Both models almost had the same set of important predictor variables. However, the order of importance differed. The random forest model has on average a smaller percent error than the linear ridge model. A support vector machine is used to predict whether or not a housing unit is in the top 10% of monthly housing cost. The accuracy, precision, and recall are calculated for this model. Finally, ideas on improving the predictive power of these models are discussed.



<hr>

<sup id="fn1">1. The score function used is the r-squared value. Other score functions are available in sci-kit learn. For r-squared, a score of 1 indicates a model that exactly predicts the actual values<a href="#ref1" title="Jump back to footnote 1 in the text.">↩</a></sup>

<sup id="fn2">2. See sci-kit learn RadomForestRegressor documentation for definition of these parameters<a href="#ref2" title="Jump back to footnote 2 in the text.">↩</a></sup>

<sup id="fn3">3. Accuracy is the sum of true positives and true negatives divided by the total number of observations; precision is true positives divided by total predicted positive values; recall is true positives divided by total actual positive values<a href="#ref3" title="Jump back to footnote 3 in the text.">↩</a></sup>

