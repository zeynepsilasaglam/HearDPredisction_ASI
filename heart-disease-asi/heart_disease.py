# #Predicting heart disease
# Using [Heart Disease Dataset from Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

# Model choises
# We'll use different classification algorithms and comparing the result.
# * Logistic Regression
# * k-Nearest Neighbors
# * Random Forest
# * Decision Trees
# * Naive Bayes
# * Gradient Boosting

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from matplotlib.pyplot import figure
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

# Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Loading data

read_data = pd.read_csv("dataset/heart.csv")
data_shape = read_data.shape

# Exploring data
read_n_rows = read_data.head()

count_unique_values = read_data.target.value_counts()

origin_relative_frequency = read_data.target.value_counts(normalize=True)

def origin_bar_plot_target_column():
  read_data.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"]);
  plt.show()

def origin_summary():
  read_data.info();

origin_statistics = read_data.describe()


# Heart Disease Frequency according to Gender

count_gender = read_data.sex.value_counts()

def boxplot_age():
  plt.figure(figsize=(12, 10))
  plt.xlabel("age",fontsize=18)
  plt.ylabel("Agagee",fontsize=18)
  sns.boxplot(x='target',y='age',data=read_data,palette='winter')
  plt.show()


# Compare target column with sex column
compare_target_with_gender = pd.crosstab(read_data.target, read_data.sex)


# Let's make a simple heuristic.
# 
# Since there are around 300 women and 226 of them have a postive value of heart disease being present, we might infer, based on this one variable if the participant is a woman, there's a 72.5% chance she has heart disease.
# 
# As for males, there's about 700 total with 300 indicating a presence of heart disease. So we might predict, if the participant is male, 43% of the time he will have heart disease.
# 
# Averaging these two values, we can assume, based on no other parameters, if there's a person, there's a 54% chance they have heart disease.

# Create a plot
def plot_gender_heuristic():
  pd.crosstab(read_data.target, read_data.sex).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])
  plt.title("Heart Disease Frequency for Sex")
  plt.xlabel("0 = No Disease, 1 = Disease")
  plt.ylabel("Amount")
  plt.legend(["Female", "Male"])
  plt.xticks(rotation=0);
  plt.show()

# Modeling    
# methods which are useless to user, auto called by other functions.
class HeartDiseaseHiddenFromUser:

  X = read_data.drop("target", axis=1)
  y = read_data.target.values

  @staticmethod
  def fit_and_score(models, X_train, X_test, y_train, y_test):

    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    np.random.seed(42)
    model_scores={}
    for name, model in models.items():
      model.fit(X_train, y_train)
      model_scores[name] = model.score(X_test, y_test)
    return model_scores
  
  # Tuning models
  # Tune KNeighborsClassifier

  @staticmethod
  def tune_knn(neighbors, knn, train_scores, test_scores):
    for i in neighbors:
        knn.set_params(n_neighbors = i)
        knn.fit(X_train, y_train)

        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))



X_train, X_test, y_train, y_test = train_test_split(HeartDiseaseHiddenFromUser.X, HeartDiseaseHiddenFromUser.y, test_size=0.2)


length_X_train = len(X_train)
length_y_test = len(y_test)

# 1. Logistic Regression
# put models in dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(max_iter=1000),
          "Random Forest": RandomForestClassifier(),
          "Naive Bayes": GaussianNB()}


model_scores = HeartDiseaseHiddenFromUser.fit_and_score(models, X_train, X_test, y_train, y_test)

model_compare = pd.DataFrame(model_scores, index=['accuracy'])

def plot_model_comparison(): 
  model_compare.T.plot.bar()
  plt.show()

# As we can see the `Random Forest` model performs best.


# Tuning models
# Tune KNeighborsClassifier

def knn_tune_scores_plot():
  train_scores = []
  test_scores = []

  neighbors = range(1, 21)

  knn = KNeighborsClassifier()
  HeartDiseaseHiddenFromUser.tune_knn(neighbors, knn, train_scores, test_scores)

  plt.plot(neighbors, train_scores, label="Train score")
  plt.plot(neighbors, test_scores, label="Test score")
  plt.xticks(np.arange(1, 21, 1))
  plt.xlabel("Number of neighbors")
  plt.ylabel("Model score")
  plt.legend()
  print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
  plt.show()
  


# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)


def lr_fit():
  rs_log_reg.fit(X_train, y_train)


# Fit random hyperparameter search model
def lr_fit_and_score():
  lr_fit()
  print(rs_log_reg.score(X_test, y_test))


# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model

# Find the best parameters

def rf_fit():
  rs_rf.fit(X_train, y_train);


def rf_fit_and_score():
  rf_fit()
  print(rs_rf.score(X_test, y_test))

# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model

def gs_fit():
  gs_log_reg.fit(X_train, y_train);


def gs_lr_fit_score():
  gs_fit()
  # Evaluate the model
  print(gs_log_reg.score(X_test, y_test))


