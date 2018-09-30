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

- This is the project after learning the **"Model Evaluation and Validation"** section.
- In this project, we will go through the basic ML(machine learning) procedures and get the predicted selling price for clients' home. These procedures include:
  1. Data Exploration(statistically)
  2. Utilize techniques like GridSearch and CrossValidation to optimize our learning algorithms/models
  3. Analyzing Model Performance using Learning Curves and Complexity Curves
  4. Finally, train(fit) the model and predict the selling prices
- The dataset for this project originates from the "UCI Machine Learning Repository",
but for simplicity here, we only use 489 data points and 3 features(variables) to predict
the median value of clients' houses.
- At the first time I passed this project, the R2 score is 0.7435. After I completed whole Udacity Machine Learning Nanodegree, I redesign the model in 5 mins, and get the R2 score of 0.8269.
You can check the improved version here(only coding part):[boston_housing_version2.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/boston_housing/boston_housing_version2.ipynb)

## Project 2: Finding Donors for CharityML
Check the jupyter notebook here: [finding_donors.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/finding_donors/finding_donors.ipynb)

- This is the project after learning the **"Supervised Learning"** section.
- After applying some data preprocessing techniques, we discussed and implemented:
  1. Create a training/predicting pipeline
  2. Evaluate the performance of models such as run time and f-beta scores.
  3. Model hyperparameters tuning
  4. Extract feature importance and/or feature selection
- The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income), and we will use 45,222 data points and 13 features to predicts whether an individual makes more than \$50,000. Due to the imbalance in the dataset, we used the F-beta score to evaluate our models.
- The main part of this project, is to evaluate the models we have learned, such as AdaBoost, Support Vector Machines, Random Forest, and Gradient boosting(LightGBM). By using training/predicting pipelines and visualization techniques, we can more confidently choose the best model and then tune hyperparameters.
You can check the LightGBM version here(only coding part):[finding_donors_version2.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/finding_donors/finding_donors_version2.ipynb)

## Project 3: Dog breed classifier
Check the jupyter notebook here: [dog_app.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/dog-project/dog_app.ipynb)

- This is the project after learning the **"Deep Learning"** section.
- to be continue...



## Project 4: Customer Segments
Check the jupyter notebook here: [customer_segments.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/customer_segments/customer_segments.ipynb)

- This is the project after learning the **"Unsupervised Learning"** section.
- to be continue...



## Project 5: Teach a Quadcopter How to Fly
Check the jupyter notebook here: [Quadcopter_Project.ipynb](https://github.com/jo4x962k7JL/udacity_MLND/blob/master/projects/RL-Quadcopter-2-master/Quadcopter_Project.ipynb)

- This is the project after learning the **"Reinforcement Learning"** section.
- to be continue...


## Project Capstone: Kaggle competition
Check the repository here: [Kaggle-Home-Credit-Default-Risk](https://github.com/jo4x962k7JL/Kaggle-Home-Credit-Default-Risk)

- This is the final project.
- to be continue...





<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.

