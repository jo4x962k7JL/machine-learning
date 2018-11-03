# Machine Learning
Content for Udacity's Machine Learning curriculum, which includes projects and their descriptions.

- [Project 1: Predicting Boston Housing Prices (**Model Evaluation and Validation**)](#project-1-predicting-boston-housing-prices)
- [Project 2: Finding Donors for CharityML (**Supervised Learning**)](#project-2-finding-donors-for-charityml)
- [Project 3: Dog breed classifier(**Deep
  Learning**)](#project-3-dog-breed-classifier)
- [Project 4: Customer Segments (**Unsupervised Learning**)](#project-4-customer-segments)
- [Project 5: Teach a Quadcopter How to Fly (**Reinforcement Learning**)](#project-5-teach-a-quadcopter-how-to-fly)
- [Project Capstone (**Kaggle competition**)](#project-capstone-kaggle-competition)

## Project 1: Predicting Boston Housing Prices
Check the jupyter notebook here: [boston_housing.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/boston_housing/boston_housing.ipynb)

- This is the project of the **"Model Evaluation and Validation"** section.
- In this project, we will go through the basic ML(machine learning) procedures and get the predicted selling price for clients' home. These procedures include:
  1. Data Exploration(statistically)
  2. Utilize techniques like GridSearch and CrossValidation to optimize our learning algorithms/models
  3. Analyzing Model Performance using Learning Curves and Complexity Curves
  4. Finally, train(fit) the model and predict the selling prices

<img width="807" alt="printscreen 2018-11-02 6 00 11" src="https://user-images.githubusercontent.com/26728779/47946565-87542980-deca-11e8-8870-f1daab5f6635.png">

- The dataset for this project originates from the "UCI Machine Learning Repository",
but for simplicity here, we only use 489 data points and 3 features(variables) to predict
the median value of clients' houses.
- At the first time I passed this project, the R2 score is 0.7435. After I completed whole Udacity Machine Learning Nanodegree, I redesign the model in 5 mins, and get the R2 score of 0.8269.
- You can check the improved version here(only coding part): [boston_housing_version2.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/boston_housing/boston_housing_version2.ipynb)

## Project 2: Finding Donors for CharityML
Check the jupyter notebook here: [finding_donors.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/finding_donors/finding_donors.ipynb)

- This is the project of the **"Supervised Learning"** section.
- After applying some data preprocessing techniques, we discussed and implemented:
  1. Create a training/predicting **pipeline**
  2. Evaluate the performance of models such as run time and f-beta scores.
  3. Model hyperparameters tuning
  4. Extract feature importance and/or feature selection
  
<img width="897" alt="printscreen 2018-11-02 6 03 25" src="https://user-images.githubusercontent.com/26728779/47946564-87542980-deca-11e8-996e-abd2c98f278d.png">

- The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income), and we will use 45,222 data points and 13 features to predicts whether an individual makes more than \$50,000. Due to the imbalance in the dataset, we used the F-beta score to evaluate our models.
- The main part of this project, is to evaluate the models we have learned, such as AdaBoost, Support Vector Machines, Random Forest, and Gradient boosting(LightGBM). By using training/predicting pipelines and visualization techniques, we can more confidently choose the best model and then tune hyperparameters.
- You can check the LightGBM version here(only coding part): [finding_donors_version2.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/finding_donors/finding_donors_version2.ipynb)

## Project 3: Dog breed classifier
Check the jupyter notebook here: [dog_app.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/dog-project/dog_app.ipynb)

- This is the project of the **"Deep Learning"** section.
- In this project, we build two detectors for human face and dog. Then, create CNNs in two ways, from scratch and transfer learning, to classify dog breeds. The numbers of training, validation, and testing dog images are 6680, 835 and 836 respectively. And there are 133 total dog categories. we discussed and implemented:
  1. Use dlib(HOG-features) and cv2.CascadeClassifier(Harr-features) to see if an image contains a human face. It achieves 99.30% accuracy of detecting human face correctly on 1000 images with a clearly presented face.
  2. Use a ResNet-50 model pre-trained on ImageNet to see if an image contains a dog. It achieves 98.40% accuracy of detecting dog(s) correctly on 1000 images.
  3. Classify dog breeds by creating a CNN from scratch. It achieves 40.79% test accuracy.
  4. Classify dog breeds by creating a CNN using **Transfer Learning** with pre-trained Xception bottleneck features. Finally, it achieves **85.53%** test accuracy.

<img width="810" alt="printscreen 2018-11-01 8 21 44" src="https://user-images.githubusercontent.com/26728779/47893430-2ec55380-de1a-11e8-8092-af4f1cf9ec36.png">
<img width="796" alt="printscreen 2018-11-01 8 21 54" src="https://user-images.githubusercontent.com/26728779/47893431-308f1700-de1a-11e8-9f14-c741d883a2cf.png">

- You can check the main coding part here: [dog_app_version2.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/dog-project/dog_app_version2.ipynb)(it might need around 3~4 hours for running), or you can quickly test it on CLI (it only takes around 30 secs):

```python
python3 quicktest.py yourimage.jpg
```

<img width="904" alt="printscreen 2018-11-03 12 52 30" src="https://user-images.githubusercontent.com/26728779/47956802-93da8f80-df67-11e8-8b9c-22a674b04816.png">

## Project 4: Customer Segments
Check the jupyter notebook here: [customer_segments.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/customer_segments/customer_segments.ipynb)

- This is the project of the **"Unsupervised Learning"** section.
- The dataset originates from the [UCI Machine Learning Repository：Wholesale customers Data Set](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). We will use 400 data samples with 6 features, the features 'Channel' and 'Region' will be excluded for the purposes of this project.
- Consider how the different groups of customers may be affected differently by a specific delivery scheme, we will:
  1. Analyze various customers' annual spending amounts of diverse product categories
  2. Apply one of the most famous "Feature Extraction" algorithms - **PCA(Principal Component Analysis)** to draw conclusions about the underlying structure of the wholesale customer data
  3. Use **GMM(Gaussian Mixture Model)** clustering algorithm to identify the various customer segments hidden in the data.
  4. Run A/B test to determine whether making delivery scheme change will affect its customers positively or negatively.
This will give the distributor insight into how to best structure their delivery service to meet the needs of each customer.

<img width="784" alt="printscreen 2018-11-02 6 06 08" src="https://user-images.githubusercontent.com/26728779/47946563-86bb9300-deca-11e8-8fda-b63e3838c7f3.png">

## Project 5: Teach a Quadcopter How to Fly
Check the jupyter notebook here: [Quadcopter_Project.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/RL-Quadcopter-2-master/Quadcopter_Project.ipynb)

- This is the project of the **"Reinforcement Learning"** section.

<img width="855" alt="printscreen 2018-11-02 6 04 40" src="https://user-images.githubusercontent.com/26728779/47946562-86bb9300-deca-11e8-8c4e-70dd25930fe7.png">

## Project Capstone: Kaggle competition
Check the repository here: [Kaggle-Home-Credit-Default-Risk](https://github.com/jo4x962k7JL/Kaggle-Home-Credit-Default-Risk)

- This is the final project.

<img width="1222" alt="printscreen 2018-11-02 7 02 10" src="https://user-images.githubusercontent.com/26728779/47947019-ee291100-ded1-11e8-9a0f-0f6cdb9f1e59.png">
<img width="901" alt="printscreen 2018-11-02 7 04 58" src="https://user-images.githubusercontent.com/26728779/47947031-3b0ce780-ded2-11e8-851b-690e132a99ba.png">

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.

