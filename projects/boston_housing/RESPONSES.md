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
