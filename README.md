

Introduction: In this project, we are using a database containing the parameters related to several videos on the web and we are aiming to predict the **total transcoding time for transcoding**  **them.**

**Note:** Since we will be using heavy regressors like SVM, and regarding the limited capacity of our local hardware, we used google colab environment.

## Python Code explanation:

# Data Import And Preparation

Since we are using google colab, the primary cell of the code is to allow the colab to have access to my google drive to read the CSV file.

![](RackMultipart20211220-4-ep4s4u_html_e8309911adf17c.png)

Then, we import the required dictionaries and import the dataset on which we have to work.

![](RackMultipart20211220-4-ep4s4u_html_ab5acf21bb8abe8.png) ![](RackMultipart20211220-4-ep4s4u_html_3750caf784646d0c.png)

Looking at the shape of the dataset, we see: ![](RackMultipart20211220-4-ep4s4u_html_6f6b14ca58256e00.png)

Or having a look at some samples of the dataset to have a better overview of the data...

![](RackMultipart20211220-4-ep4s4u_html_a1acb54e7e37b966.png)

Now we should check the dataset in case there is a null value in it, and fortunately, there is not any here.

![](RackMultipart20211220-4-ep4s4u_html_d14328c5b8eba2eb.png)

Now we check all the data types in our dataset. We have 3 objects, while the others are numbers.

![](RackMultipart20211220-4-ep4s4u_html_e6ffd2573d168781.png)

We extract the categorical and numerical data from dataset into two different parts.

![](RackMultipart20211220-4-ep4s4u_html_548a0077425f57b0.png) ![](RackMultipart20211220-4-ep4s4u_html_d8b4de1d20cd172c.png)

Through the below code, we box plot the categorical data

![](RackMultipart20211220-4-ep4s4u_html_c956af533c8b8493.png)

![](RackMultipart20211220-4-ep4s4u_html_11bccca4c5d4d452.png)

![](RackMultipart20211220-4-ep4s4u_html_bb1700935c6bce35.png)

![](RackMultipart20211220-4-ep4s4u_html_480c77c4b9a9405e.png)

For example, most of the videos were of the format of h264, while flv obtains less portion.

Then, we attribute dummy values to these categorical data to make them undrestandble for the regressors.

![](RackMultipart20211220-4-ep4s4u_html_8e8038649d4f3253.png)

We plot the histogram of the data. Although some of them do not represent a good distribution, since we will delete them because of their strong correlation with other attributes, the changes like getting a logarithm is not necessary at this stage.

![](RackMultipart20211220-4-ep4s4u_html_74f67accba79ff74.png)

Plotting covariance matrix as a hitmap sows stronger correlations between some attributes. In this regard, In the next cell, we delete some attributes which have a correlation more than 80%.

![](RackMultipart20211220-4-ep4s4u_html_779a8fcfceeee38c.png)

![](RackMultipart20211220-4-ep4s4u_html_dca74f70c4e182d4.png)

After doing so, we can see that the new attributes are much less dependent.

![](RackMultipart20211220-4-ep4s4u_html_9fcac85ca65d528.png)

With the pair plotting the attributes with the target, there is no linear correlation between any of them and the target. So it is not reasonable to use a linear regressor using just one variable.

Now we should scale the data. We should consider that:

\*\*\*We delete the target from numerical values since the next dataset which we will use to predict will not contain the target, and we should use the same scaler, so the scaler should not include the target.

![](RackMultipart20211220-4-ep4s4u_html_6a858f11cb5d9296.png)

Now we separate training and test set. We leave 70% of the data for the train, leaving the other for the test.

![](RackMultipart20211220-4-ep4s4u_html_edefc09d0e595fce.png)

# Regressor

The first step to use a regressor is to define a function that can specify the best parameters of a regressor. We use the _ **negative** _ of mean absolute error, since the less the error, the better the result.

![](RackMultipart20211220-4-ep4s4u_html_fae82c15d19ade4a.png)

We start from linear regressor:

![](RackMultipart20211220-4-ep4s4u_html_f0d4b5ae814fb21.png)

Using ridge or lasso resoult in a little better outcome in terms of MAE:

We try different regulation and normalization terms as the parameters.

![](RackMultipart20211220-4-ep4s4u_html_ddfe7e0853e6d38f.png)

With KNeighborsRegressor we receive:

![](RackMultipart20211220-4-ep4s4u_html_9911e4b9375a8c06.png)

But it seems the best results are from Random Forrest Regressor. The MAE is about 3, which in comparison to the values that the target can achieve is reasonable. Besides, there is now a huge gap between Train and test, so we do not encounter overfitting. ![](RackMultipart20211220-4-ep4s4u_html_b427ec5e37f63aac.png)

Unfortunately, the SVM required too much computational time, and it was not possible to use it with the present Hardware.

Finally, Adaboost does not show better result in terms of MAE, so our final decision would we RandomForrestRegressor

![](RackMultipart20211220-4-ep4s4u_html_510ec780887da82f.png)

Then, we train the model with whole the data, with the parameters resulted from Gridsearch, to have a better result.

![](RackMultipart20211220-4-ep4s4u_html_f0bfc32514e0771.png)
