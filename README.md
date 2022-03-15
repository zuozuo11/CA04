# CA04


## Background

Based on the data in CA03, we evaluated the different model's performance. Our first task is finding Optimal Value of a key Hyper-parameter. We need the influence of different amount of N_ESTIMator_options on model accuracy. Here we discuss randomforest model, AdaBoost model, Gradient Boost model(classifier) and XGB Model. At last, We show the accuracy and AUC of these four models


---------------------------------------------------------------------------------------

## Install

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score



---------------------------------------------------------------------------------------
## Process
1 Data Source and Contents

2 Finding Optimal Value of a key Hyper-parameter

3 Building a Random Forest Model

4 Building AdaBoost, Gradient Boost (classifier) and XGB Model

5 Compare Performance
