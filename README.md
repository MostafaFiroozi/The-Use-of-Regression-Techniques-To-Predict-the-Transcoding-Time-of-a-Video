

## Introduction: <br/>
In this project, we are using a database containing the parameters related to several videos on the web and we are aiming to predict the **total transcoding time for transcoding**  **them.**

**Note:** Since we will be using heavy regressors like SVM, and regarding the limited capacity of our local hardware, we used google colab environment.

## Python Code explanation:

### Data Import And Preparation

Since we are using google colab, the primary cell of the code is to allow the colab to have access to my google drive to read the CSV file.

![image001](https://user-images.githubusercontent.com/73081215/146787358-4cd6229a-5842-41ac-8d99-0324d611fcbf.png)


Then, we import the required dictionaries and import the dataset on which we have to work.

 ![image003](https://user-images.githubusercontent.com/73081215/146787437-97148527-22dd-4708-a154-0fd0b8f00cd8.png)



Looking at the shape of the dataset, we see:

![image007](https://user-images.githubusercontent.com/73081215/146787688-8a8bae5f-2f47-42cc-a360-b2baaa58329e.png)

Or having a look at some samples of the dataset to have a better overview of the data...

![image010](https://user-images.githubusercontent.com/73081215/146787920-a91ee200-28bf-4e46-a444-ebd4a3fa2922.jpg)


Now we should check the dataset in case there is a null value in it, and fortunately, there is not any here.

![image012](https://user-images.githubusercontent.com/73081215/146787994-0ffc54bc-0392-41c8-8b84-7359d77e7c1b.jpg)



Now we check all the data types in our dataset. We have 3 objects, while the others are numbers.

![image013](https://user-images.githubusercontent.com/73081215/146788598-2dc44796-0848-4a2c-995b-5783aa773a2d.png)



We extract the categorical and numerical data from dataset into two different parts.

![image014](https://user-images.githubusercontent.com/73081215/146788700-baf7da03-01fd-4c6e-b921-5eda726ea540.png)

![image016](https://user-images.githubusercontent.com/73081215/146788954-fa914a98-83de-4e89-8d70-e094afd5d3ec.png)

Through the below code, we box plot the categorical data
![image018](https://user-images.githubusercontent.com/73081215/146789066-8e8886cc-8242-4d7d-b35d-72e06f0b6e1a.png)
![image019](https://user-images.githubusercontent.com/73081215/146789074-873ffb5e-a46d-4e70-8a7f-a5a1816e2998.png)
![image020](https://user-images.githubusercontent.com/73081215/146789085-aecfdd9a-ce70-4393-a53a-849f928ee6b7.png)
![image021](https://user-images.githubusercontent.com/73081215/146789098-db9a9a1c-f535-405f-ab5f-b817f5b521be.png)



For example, most of the videos were of the format of h264, while flv obtains less portion.

Then, we attribute dummy values to these categorical data to make them undrestandble for the regressors.

![image022](https://user-images.githubusercontent.com/73081215/146789154-a628ba8c-c0c5-4749-a1a6-2c0c652efd11.png)


We plot the histogram of the data. Although some of them do not represent a good distribution, since we will delete them because of their strong correlation with other attributes, the changes like getting a logarithm is not necessary at this stage.
![image024](https://user-images.githubusercontent.com/73081215/146789212-a82c9b46-2243-4b74-8042-e00f24711eb5.png)



Plotting covariance matrix as a hitmap sows stronger correlations between some attributes. In this regard, In the next cell, we delete some attributes which have a correlation more than 80%.
![image026](https://user-images.githubusercontent.com/73081215/146789269-bb7b5a80-9625-4270-a027-e776f9d5d514.png)

![image028](https://user-images.githubusercontent.com/73081215/146789275-9a79a4bd-0de2-4fe1-9b47-63d27e74efbb.png)


After doing so, we can see that the new attributes are much less dependent.

![image030](https://user-images.githubusercontent.com/73081215/146789298-fe861c59-3165-4404-8ad9-eed2a53c5e16.png)


With the pair plotting the attributes with the target, there is no linear correlation between any of them and the target. So it is not reasonable to use a linear regressor using just one variable.

Now we should scale the data. We should consider that:

\*\*\*We delete the target from numerical values since the next dataset which we will use to predict will not contain the target, and we should use the same scaler, so the scaler should not include the target.

![image032](https://user-images.githubusercontent.com/73081215/146789594-6319979e-8fdf-4ac4-b02a-524b6aad9df4.png)


Now we separate training and test set. We leave 70% of the data for the train, leaving the other for the test.

![image033](https://user-images.githubusercontent.com/73081215/146789675-e601aa64-fbb5-45d5-ba46-d48e8feb9d04.png)


# Regressor

The first step to use a regressor is to define a function that can specify the best parameters of a regressor. We use the _ **negative** _ of mean absolute error, since the less the error, the better the result.
![image035](https://user-images.githubusercontent.com/73081215/146789739-212cd4ad-aa38-47b9-bfab-8e4249f839a6.png)



We start from linear regressor:

![image037](https://user-images.githubusercontent.com/73081215/146790629-ea60693d-e926-4f2f-a7f9-f0988dc2f0a2.png)


Using ridge or lasso resoult in a little better outcome in terms of MAE:
![image038](https://user-images.githubusercontent.com/73081215/146790806-b7d19143-b1c3-4429-a0c3-052cafb578a9.jpg)

We try different regulation and normalization terms as the parameters.
![image041](https://user-images.githubusercontent.com/73081215/146790917-d8dde341-ce29-4c31-8673-c034394b2182.png)



With KNeighborsRegressor we receive:

![image041](https://user-images.githubusercontent.com/73081215/146791147-72ca371c-38c3-4d0e-801d-ac15b94643f8.png)


But it seems the best results are from Random Forrest Regressor. The MAE is about 3, which in comparison to the values that the target can achieve is reasonable. Besides, there is now a huge gap between Train and test, so we do not encounter overfitting.
![image042](https://user-images.githubusercontent.com/73081215/146791076-ff609c8f-34cd-47bc-b494-7554c3120ca4.png)

**Unfortunately, the SVM required too much computational time, and it was not possible to use it with the present Hardware.


Finally, Adaboost does not show better result in terms of MAE, so our final decision would we RandomForrestRegressor

![image043](https://user-images.githubusercontent.com/73081215/146791304-6459776a-6c46-41aa-8f23-474212ac7374.png)


Then, we train the model with whole the data, with the parameters resulted from Gridsearch, to have a better result.

![image045](https://user-images.githubusercontent.com/73081215/146791321-3d8ed7e2-70f6-4de6-8200-eff6ed961d33.png)

