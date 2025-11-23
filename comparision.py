import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import zipfile

from pycaret.classification import *
from pycaret.classification import ClassificationExperiment

zip_file_path = 'dataset-1.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall()

df = pd.read_csv('transaction_dataset.csv')
df = df.drop(columns=['Index', 'Address','Unnamed: 0'])
print(df.head())
common_elements_list =['FLAG','Time Diff between first and last (Mins)', 'Received Tnx', 'avg val received', ' Total ERC20 tnxs']

original_data = df
new_data = df
for i in df.columns:
  if i not in common_elements_list:
    new_data = new_data.drop(columns=[i], axis=1)
    
cat_data = df
cat_ele =  ['FLAG', 'Number of Created Contracts', 'max value received ', 'avg val received', 'min val sent', 'total Ether sent', 'total ether balance', ' ERC20 total Ether received', ' ERC20 total Ether sent contract', ' ERC20 uniq rec contract addr', ' ERC20 min val rec']
for i in df.columns:
  if i not in cat_ele:
    cat_data = cat_data.drop(columns=[i], axis=1)

lgbm_data = df
lgbm_ele = ['FLAG','Avg min between received tnx','Time Diff between first and last (Mins)','Sent tnx','Received Tnx','Unique Received From Addresses','avg val received', 'min val sent','total ether balance','Total ERC20 tnxs',' ERC20 total Ether received',' ERC20 total ether sent',' ERC20 min val rec']
for i in df.columns:
  if i not in lgbm_ele:
    lgbm_data = lgbm_data.drop(columns=[i], axis=1)
print(lgbm_data.shape)
          
#print("=============================  ORIGINAL DATASET  =============================")
#bin_set_og = setup(original_data, target = 'FLAG', session_id = 123, fix_imbalance = True, fix_imbalance_method = 'SMOTE')
#exp = ClassificationExperiment()
#exp.setup(original_data, target = 'FLAG', session_id = 123, fix_imbalance = True, fix_imbalance_method = 'SMOTE')
#best_og = compare_models()
#xg = create_model('xgboost')
# plot confusion matrix
#plot_model(best_og, plot = 'confusion_matrix')

#print("================================  NEW DATASET  ================================")
#bin_set_new = setup(new_data, target = 'FLAG', session_id = 123, fix_imbalance = True, fix_imbalance_method = 'SMOTE')
#exp = ClassificationExperiment()
#exp.setup(new_data, target = 'FLAG', session_id = 123, fix_imbalance = True, fix_imbalance_method = 'SMOTE')
#best_new = compare_models()
#xg = create_model('xgboost')
# plot confusion matrix
#plot_model(best_new, plot = 'confusion_matrix')

print("=============================  CHI & ANNOVA DATASET  =============================")
bin_set_cat = setup(cat_data, target = 'FLAG', session_id = 123,fix_imbalance = True, fix_imbalance_method = 'SMOTE')
exp = ClassificationExperiment()
exp.setup(cat_data, target = 'FLAG', session_id = 123,fix_imbalance = True, fix_imbalance_method = 'SMOTE')
best_cat = compare_models()
xg = create_model('xgboost')
plot_model(best_cat, plot = 'confusion_matrix')

#print("=============================  LightGBM DATASET  =============================")
#bin_set_lg = setup(lgbm_data, target = 'FLAG', session_id = 123, fix_imbalance = True, fix_imbalance_method = 'SMOTE')
#exp = ClassificationExperiment()
#exp.setup(lgbm_data, target = 'FLAG', session_id = 123, fix_imbalance = True, fix_imbalance_method = 'SMOTE')
#best_lg = compare_models()
#xg = create_model('xgboost')
# plot confusion matrix
#plot_model(best_lg, plot = 'confusion_matrix')
