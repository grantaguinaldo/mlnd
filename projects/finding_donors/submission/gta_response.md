***

### Question 2 - Model Application
List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen

- Describe one real-world application in industry where the model can be applied. 
- What are the strengths of the model; when does it perform well?
- What are the weaknesses of the model; when does it perform poorly?
- What makes this model a good candidate for the problem, given what you know about the data?

**HINT:**

Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

**Answer: Question 2**

This problem requires the use of a binary classifier to predict if a given salary will be `<=50K` or `>50K`.  For this problem, I will propose to use the following machine learning models.

* **Logistic Regression**
* **Support Vector Machines**
* **Gaussian Naive Bayes**

*Logistic Regression*

**Describe one real-world application in industry where the model can be applied.** Predicting a binary outcome, to understand customer churn.  In this [paper,](https://analytics.ncsu.edu/sesug/2017/SESUG2017_Paper-191_Final_PDF.pdf) Sean Ankerbruck, outlines a method that uses logistic regression to, "[]identify important factors that influence churn and classify individuals based on their predicted likelihood
to churn."  

**What are the strengths of the model; when does it perform well?** Strengths of using logistic regression include the ability to output [probabilistic results](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba) of the model. This can be of value since in some cases, we are not solely interested in knowing if a specific data point is in one class or another (i.e., class zero or one), but rather, knowing what is the probability that it belongs to one class or another (i.e, there is a 75% chance that this datapoint belongs to class one).  Another strength is that we are able to look at the weights (coefficients) of the terms (features) to identify the features that are most relevant to the overall prediction. For example, it is possible to know that out of all of the features, the most important feature is the `captial-gains` when determining if a user will make `>$50K`.

**What are the weaknesses of the model; when does it perform poorly?** One **weakness** of this model is that it only be used to make binary classifications.  If you have a multi-class problem, then you're going to need to consider another model.  In addition, if your features are non-linear, then logistic regression may not work since it is a [linear model](https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier) and requires the input features to be linear.

**What makes this model a good candidate for the problem, given what you know about the data?** Logistic regression is a good candidate for this problem since we have a binary outcome, and an array of input features. As mentioned earlier, when using this model, we are not only able to know which class is being assigned to the datapoint, but also know the probability, or the strength that certain datapoint belongs to one class or another.  In other words, we will be able to find cases, where a user belongs to `>$50K` but by only small amount (e.g., 61%).  


*Support Vector Machines*

**Describe one real-world application in industry where the model can be applied.** One example of using a support vector machine in the real-world includes the development of a  bankruptcy prediction model. In this [paper](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=B21D947ACCF7DFA22075A071209C8836?doi=10.1.1.98.2804&rep=rep1&type=pdf), Shin, et. al., outlines the development of such a model using a "small" training dataset of 2,300 observations. 

**What are the strengths of the model; when does it perform well?** One strength of using support vector machine is that you are able to use the model on [small datasets](https://stats.stackexchange.com/questions/47209/what-are-good-techniques-for-modeling-small-datasets). Depending on the business case, you may not have access or have the ability to work with a *large* dataset, and this can have adverse impacts on the final results of the analysis. Knowing that a SVM works well with small datasets is of value in the case that you are trying to solve a classification problem but don't have access to a large dataset. 

**What are the weaknesses of the model; when does it perform poorly?** One problem that I have read about when using a SVM is that training the model can be computationally intensive as the dataset get large (i.e., *O(n<sup>3</sup>)* for [Kernel Methods](https://stats.stackexchange.com/questions/327646/how-does-a-random-kitchen-sink-work)).  In the case of large datasets, sometimes, performing a [dimensionality reduction](http://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) is needed to get the model to work efficiency.  In terms of performance, an [**SVM doesn't perform well** if you have highly skewed data](https://www.quora.com/For-what-kind-of-classification-problems-is-SVM-a-bad-approach) (like finding credit card fraud).

**What makes this model a good candidate for the problem, given what you know about the data?** 

An SVM can perform well in this case since we are making a [binary classification.
](https://www.quora.com/For-what-kind-of-classification-problems-is-SVM-a-bad-approach)

*Gaussian Naive Bayes*

**Describe one real-world application in industry where the model can be applied.** 

One real world example of Gaussian Naive Bayes in the prediction of bankruptcies (see [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.7345&rep=rep1&type=pdf)).  A Gaussian Naive Bayes would work in this case since the features (gross revenue, expenditures, etc.) are continuous, as opposed to discrete.  

**What are the strengths of the model; when does it perform well?** 

A few strengths of the Gaussian Naive Bayes model is that is can make probabilistic predictions similar to logistic regression. In general, the model performs well even if you have a small dataset (see [here](https://www.quora.com/What-are-the-advantages-of-using-a-naive-Bayes-for-classification))

**What are the weaknesses of the model; when does it perform poorly?** 

One weakness of the model is that there is an assumption about the type of distribution that is being used in the model (either Gaussian or Multinomial). Therefore, if you have a dataset that maybe using another type of distribution, your model may not work properly.

**What makes this model a good candidate for the problem, given what you know about the data?**

The Gaussian Naive Bayes model works well in this case since we are working with a feature space that is continuous as opposed to discrete.

***


### Question 3 - Choosing the Best Model

* Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than $50,000. 

** HINT: ** 
Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:
* metrics - F score on the testing when 100% of the training data is used, 
* prediction/training time
* the algorithm's suitability for the data.

**Answer**

Based on the three models selected, we recommend that the best model that is used for *CharityML* is **Logistic Regression**. This recommendation is based on the following.

**First**, as compared the SVC and the Gradient Boosted Classifier, the Logistic Regression model is the quickest to train on 100% of the dataset.   From the Model Training charts creating using the `vs.evaluate(results, accuracy, fscore)` function, there is an inverse relationship between the training times of the `SVC` and `LogisticRegression`.  That means that as the dataset gets larger, the time to train the dataset get shorter for logistic regression and larger for the SVC.

**Second,** there seems to be less overfitting with the logistic regression model as compared to the Gradient Boosting Classifier.  Specifically, the difference in the `F1` score, at 100% of the training data, is about 15 basis points for the Gradient Boosting Classifier, but about 5 points for the logistic regression model. In general, the degree to which a model overfits can be a cause of concern since overfitting is an indication of poor generalization (which is something that we strive to avoid when doing as a machine learning engineer). 

It is understood that the F1 score is higher for the Gradient Boosting Classifier, however, from the standpoint of overfitting, I have taken the position that is it is more important to have a model that does not overfit, even at the expense   of performance.  In an actual case, the decision to pick a model based on it's performance or over- or under-fitting is something that should be established before starting the project.  

***

### Question 4 - Describing the Model in Layman's Terms

* In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.

** HINT: **

When explaining your model, if using external resources please include all citations.

**Answer**

The logistic regression model is a classifier that can predict of a certain datapoint belongs to one class or another (think, is this a cat or a dog).  

How the model does this is quite complex, but it can be broken down into a few simple steps.

First, as in the cat/dog case, the model determines what are the most important features of what makes a cat a cat and what makes a dog a dog based on the data that you provide.  For example, an important feature can be the length of the muzzle.  Dogs **tend** to have longer muzzles than cats and so on. 

Second, once the model identifies the most important features, the model will look at a test point, or the point that you want to classify. Based on the features that the model thought was important, the model will look at the new data point and assign a probability (between 0-1) of  what class to place the point into. For example, if the new datapoint has a long muzzle, the model will likely assign a high probability (e.g, 0.8) to  classify the image as being a dog.  On the other hand, if the muzzle length is small, then the model may assign a smaller probability (e.g., 0.25) that the new point is a dog. When developing the model, the cut off point is typically 0.5 for classification.  This means that the model will assign the class if the probability is above or below 0.5 (> 0.5 will be assigned to class 1 and < 0.5 will be assigned to class 0). 

***

### Question 5 - Final Model Evaluation

* What is your optimized model's accuracy and F-score on the testing data? 
* Are these scores better or worse than the unoptimized model? 
* How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?  

**Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

During the model optimization step, we found that our model does worse as compared to the unoptimized variant.  For example, during the training step, we saw `F1` scores for the logistic regression model in the area of 0.8, however, during the optimization step, we saw `F1` value in the area of 0.68. In both cases, we set `random_state=100` to ensure that we are on the same playing field. 

Despite the optimized model performing worse than the unoptimized variant, the optimized model still does perform better than the baseline model that that we assessed in Question 1.  For reference, the baseline model is one that always predicts that an individual made more than \$50,000.  As compared to the baseline, the optimized model has an F1 score that is more than 100% higher that for the baseline model.  In terms of the accuracy of the optimized model, the optimized model has an accuracy score that is more than 300% that of the baseline model. 

***

### Question 6 - Feature Relevance Observation
When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

**Answer**

Of the 13 features listed in the dataset, I believe that the five most important features that drive the prediction of making more than \$50,000 is, in descending order of importance:

* Education Level
* Native County
* Hours-per-week worked
* Occupation
* Workclass

**Native County:** Out of all of the features, the one that is the largest driver if someone can make more than $50,000 per year is the country that they are located in. Since each county operates on a different economic system, the cost of living is different in each county.  I'd like to assume that companies pay according to to the value provided relative to the market of that country.  Therefore, in wealthier countries the criteria for earning more than $50,000 may be quite different than ones that are not as wealthy.  To get around this, it may be helpful to normalize all of the earnings to one county or a uniform metric as to remove the the dependencies of cost of living and specific economic factors from the analysis.

**Education Level:** In general, I feel that someone who has more education will be paid higher since there should be a relationship between the education level and the occupation of the individual. In this case, folks that have the occupation of `Exec-managerial` or `Prof-specialty` should have more education and therefore earnings.

**Occupation:** Similar to the point noted on education, I believe that occupation is a strong predictor of earnings, or the ability to earn more than \$50,000 per year. Again, I feel that folks that have the occupation of `Exec-managerial` or `Prof-specialty` should have more education and therefore higher earnings 

**Hours-per-week worked:** From experience, it seems that people who work long hours do end up making more in both management and non-management roles.  For managers, this is due to more responsibility which requires you to do more. In the non-management role, I have typically seen this be the result of working more overtime which does equate to a higher annual salary at the end of the year. 

**Workclass:** In my experience, work class is a good indicator if someone can earn more than \$50,000 per year.  I saw this because in certain industries, like engineering, it is possible to earn more money working for a state or local government than working in the private sector.  From what I have seen, the reason for this is that these types of employers have labor unions that fight for higher wages for the employees.  This type of setup is less prevalent in the private sector which is why I say that it can be easier for a public sector employee to earn more than $50,000 per year.

***

### Question 7 - Extracting Feature Importance

Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  

* How do these five features compare to the five features you discussed in **Question 6**?
* If you were close to the same answer, how does this visualization confirm your thoughts? 
* If you were not close, why do you think these features are more relevant?

**Answer:  Question 7**

In reviewing the chart created for Question 7, my predictions were very different than what the model came up with.  For example, the most important feature that the model predicted is `captial-gain` whereas I noted that `Native County`.  In fact, **none** of the features that I noted in my list appeared in the list of important features generated from the model. 

The single best explanation that I have for this, is that I selected discrete features , where as the ones that the model selected were continuous, except for the martial status. Although, I will add that it does make sense that a person will have a higher income if they have higher amounts of capital gains.

***

### Question 8 - Effects of Feature Selection

* How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
* If training time was a factor, would you consider using the reduced data as your training set?

**Answer**

This seems to be a good example of dimensionality reduction done correct. In this case, we have reduced our feature space from 13 down to five, and in doing so, we are able to have `F1` scores that are close to that of the full feature space. What these results suggests is that the variability of the full dataset are preserved quite well in a dataset that only has five features (sigue to  to [Principal Component Analysis](http://setosa.io/ev/principal-component-analysis/) in the unsupervised learning section).  In the case that there was a strong dependency between training time and feature space, I would consider using the smaller feature space in the model since it can help reduce the amount of time to get the overall model to production. 

***