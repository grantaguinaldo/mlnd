 **Answer Question 1:**

In general, I have seen that the main driver of home prices is the quality of the school district.  Of course there are other attributes that factor into the selling price of a home, like the size, or in this case, the `'RM'` values, however, from what I have seen, the main driver is the quality of the school district. 

As to the `'RM'` values. All else being equal, I expect that a high `'RM'` value to have a higher value than a smaller one.  The reason for this, is that the `'RM'` value is proportional to the overall size of the home, and in general, large homes do sell for more. 

As to the 'LSTAT'` values.  Again, all else being equal, I expect that a home in a neighborhood with a lower 'LSTAT'` value to have a **higher value** than one with a higher value.  The reason for this is that neighborhoods are generally clusters of people in the same income bracket, therefore, a higher `'LSTAT'` value means that there is more people of a lower class in that neighborhood.  In addition, the home values of folks that are of a lower class are generally lower since their incomes are lower.

As to the `'PTRATIO'` values. School overcrowding can be one metric that gauges the cost/value of a home since overcrowded class rooms are typically seen in lower class/cost areas (this can be related to the overall quality of an education as well).  That being said, I expect that lower `'PTRATIO'` values to result in a higher home price since the quality of the education tends to also be higher in these situations.  Like I said at the start:

> In general, I have seen that the main driver of home prices is the quality of the school district. 

**Answer Question 2:**

The calculated `'r^2'` score (`'0.923'`), means that 92.3% of the variance in the dataset can be predicted by the independent variables. That being said, **I do** consider this model to have captured the variation of the target variable.  

**Answer Question 3:**

The benefit of using `'train_test_split'` in modeling is that it allows you to ensure that the model does not over- or under-fit the data since you will evaluate the model using data (i.e., the `'test'` set) that the model has not seen.  This in-turn ensures that the model is able to generalize well to out of sample data. 

**Answer Question 4:**

From the four charts presented, the one where `'max_depth=1'` seems to fit the data best.  The reason being that the for `'max_depth=1'`, the error of the `'training'` and `'testing'` sets are converging to a lower number, than say,  `'max_depth=3'`. In passing, the charts for `'max_depth=6'` and `'max_depth=10'` don't seem to be converging. 

If you add more points to the datasets, the training error should increase since there is more data to fit, however, the testing error should decrease since we have a better model that has been trained with more data. 

In general, having more training data increases the training error, however, in the long run, you will have a better model since has seen more points, and is more likely to generalize better since it has the ability to pick up all of the nuances in the dataset. 

It is noted that there is a point at which the training and the testing curves to approach each other (with a very small distance between them). In this case, adding more training data to the model, may not result in a better training score.  

**Answer Question 5:**

As to the chart with `'max_depth=1'`, I wold say that the is the sweet spot for the model since the error is at the minimum.  It seems that as the `'max_depth'` approaches four, that the model starts to show high bias (underfit), but as the depth increases, the model starts to show high variance or overfitting.

The key clues that show that the model is over fitting is the the distance between the training and testing scores, on the complexity chart, is getting larger. Visual clues that a model is underfit is that the delta between the training and testing scores are small, however, the point at where they converge is not at the minimum. 

**Answer Question 6:**

The model that uses  `'max_depth=1'` would generalize the best to unseen data.  The reason for that is two fold.  First, the delta between the training and testing scores are small, and second, the score at which the training and testing points converge to is at the minimum compared to all of the other `'max_depth'` values. 

This also makes sense given that the best models are the simplest models. 

**Answer Question 7:**

A grid search is a technique to optimize a model by changing the hyperparamters of the model, recording the performance of the change on the model (e.g., like the F1 score), and then selecting the set of hyper parameters that creates the best performing model. 

As an example, for a random forest, one of the hyper parameters that are available in scikit learn (SK Learn) is `'max_depth'`. 

All of the hyper parameters that can be used for a [random forest (from SK Learn)](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) is shown below.

```
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, 
splitter=’best’, max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
min_impurity_split=None, class_weight=None, presort=False)

```

The `'max_depth'` parameter specifies how many nodes or levels to have on the decision tree.  As it relates to model performance, having more nodes/levels can affect the ability of the model to generalize well.  Therefore, when changing the `'max_depth'` parameter, it is possible to see the impact of the change when looking at the `'F1'` score that is produced when the testing data is used on the model.  

Specifically, in some cases, increasing the `'max_depth'` value can lead to a higher `'F1'` score, which is an indication that the model is performing better on the testing data. 

**Answer Question 8:**

When assessing model performance, it is important that the model be tested on data that the model **has not** seen.  The reason for this is that you don't want to introduce bias into the model that could affect the ability to generalize. As a result, it is common practice to split the data up into training and testing sets as in the case of `'train_test_split'` in SK Learn. 

In `'train_test_split'`, the testing set, is always going to be the same data points. Since the goal of building a model is to have a model that generalizes on out of sample data, it is possible, that biases could be introduced into the dataset when doing `'train_test_split'` since the testing and training set will always include the same datapoint. 

To get around this potential source of bias, it is customary to test the performance of the model using different points in the testing and training set.  The problem here, however, is that we only have a set amount of data points, and the cardinal rule is to never test a model using data points that the model has already seen. 

To solve this problem, one can use k-fold cross validation.  In k-fold cross validation, the data set is sub divided into `'k'` divisions of equal size. Once the divisions have been made, the model is trained with `'k-1'` data points, and the then tested with data from the k<sup>th</sup> subdivision.  

This process is repeated for `'k'` times and an accuracy or `'F1'` score computed for each step.  Once the model has been trained and evaluated `'k'` times, the average performance is provided and used to describe the model.  Put in another way, k-fold cross validation ensures that the training and testing of the model is homogeneous across all of the data points, while still complying with the cardinal rule of next testing a model using data points that the model has already seen during the testing phase. 

As to the use of cross validation and grid search. As noted in the [SK learn documentation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation):

> When evaluating different settings (“hyper parameters”) for estimators, such as the C setting that must be manually set for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. 

To ensure that the model does not inadvertently learn about nuances of the dataset when going through a grid search, cross validation is used since model performance will be judged over a series of `'k'` times thereby removing the ability of the model to learn about nuances of the dataset. 

**Answer Question 9:**

From the results from grid search, it seems that the best model is one that has `'max_depth = 4'`.  As it relates to the *guess* that has made in Question 6, it seems that a `'max_depth = 4'` would not have been expected to produce the best results since the error has not been minimized. As noted from Question 6:

> The model that uses  `'max_depth=1'` would generalize the best to unseen data.  

**Answer Question 10:**

The model has predicted the three home prices for each client.

```
Predicted selling price for Client 1's home: $403,025.00
Predicted selling price for Client 2's home: $237,478.72
Predicted selling price for Client 3's home: $931,636.36
```

As it relates to the response provided in question 1, all it seems that my intuition and the model are in agreement for the following three reasons.

First, I noted that:

> [I] expect that lower `'PTRATIO'` values to result in a higher home price since the quality of the education tends to also be higher in these situations.

The home predictions follow that trend, since there is a 12-to-1 student ratio for client 3. 

Second, I noted that:

> [I] expect that a home in a neighborhood with a lower 'LSTAT'` value to have a **higher value** than one with a higher value.

Again, the home predictions follow this trend since there is a 3% poverty rate for client 3 and a 22% poverty rate for client 2.

Finally, I noted that:

> I expect that a high `'RM'` value to have a higher value than a smaller one.

The home prediction for client 3 follows this trend since it has the most bedrooms, as well as the highest predicted selling price.  Client 2 has the smallest house as well a the lowest selling price.

**Answer Question 11:**

* In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.  

**Hint:** Take a look at the range in prices as calculated in the code snippet above. Some questions to answering:

- How relevant today is data that was collected from 1978? How important is inflation?
- Are the features present in the data sufficient to describe a home? Do you think factors like quality of appliances in the home, square feet of the plot area, presence of pool or not etc should factor in?
- Is the model robust enough to make consistent predictions?
- Would data collected in an urban city like Boston be applicable in a rural city?
- Is it fair to judge the price of an individual home based on the characteristics of the entire neighborhood?

In this project, we trained a model to predict the selling home price in Boston.  While predictions are made from what seems to be a robust model, care should be exercised when solely relying on this model to make predictions for the following reasons.

First, the data used to train the model was collected in 1978.  Home prices have changed from 1978, due to inflation, and this must be considered when using this model.  One possible correction is to correct the predicted price to 2018 dollars using a standard rate of inflation. 

Second, calculating the actual price of a home is a complex endeavor that can use multiple inputs or features.  In this model, we have built a model that uses only three features to predict the home price, but there are indeed others that can be used. One of these additional features is the crime rate of the neighborhood.  While more features can be certainly added to the model, one must exercise care because not all features added to the model are valuable.  It is possible to add more features to the model and at the same time add more noise to the model.  To get around this, is is best to look at the importance of each feature to the model.

Third, while the model does make predictions, it should be noted that the model may be overfit given the `'max_depth=4'` value.  When looking at the chart from Question 6, it suggest that the model may be overfit relative to a `'max_depth=1'`.  To ensure that the model is not overfit, I would run the complexity analysis again, but use the results from a 10-fold cross validation as opposed to points generated from the `train_test_split`.

Fourth, the model does not consider the fact that Boston is a urban city and people may want to use the model to predict home prices in a rural setting. Home prices in a rural setting may be lower and therefore, this model could estimate higher prices in a rural setting. 

Fifth, in my opinion, it is not 100% fare to judge a home price solely by the characteristics of the entire neighborhood, although, in my experience, this does play a strong component in the overall price.  A new home in a poor neighborhood may be priced artificially low, even if it has a high amount of bedrooms.

In all, while models can make predictions about future events, care must be exercised when relying on the data.  It is important to have context about the data used to create the model as well as domain knowledge when using or building a model to ensure that the output of a model is used correctly. 