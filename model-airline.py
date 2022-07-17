# %% [markdown]
# # Thêm các thư viện

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

# %%
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics

# %% [markdown]
# # I. Chuẩn bị dữ liệu

# %% [markdown]
# ## 1. Đọc dữ liệu

# %%
df = pd.read_csv("Invistico_Airline.csv")

# %%




# %% [markdown]
# Chúng ta có thể thấy tập dữ liệu này gồm các nhóm liên quan đến thuộc tính của khác hàng (Age, Gender,...), thông tin vè chuyến bay (Class, Inflight wifi service, ...), đánh giá của khách hàng (Seat comfort, Food and drink, ...). 

# %% [markdown]
# Loại dữ liệu của từng nhóm
# - Thuộc tính khách hàng: Object 
# - Thông tin chuyến bay: Interger and Object 
# - Đánh giá của khách hàng: Interger (1->5) 

# %% [markdown]
# ## 2. Tổng quan về dữ liệu

# %%
def rename_column(df,from_c,to_c = '_'):
    display(Markdown('**RESULT:**'))
    print('Columns changed from {} to _'.format(from_c))
    print(df.columns[df.columns.str.contains(from_c)])
    df.columns = [label.replace(from_c, to_c ) for label in df.columns]

# %%
rename_column(df,' ')
rename_column(df,'-')

df["Age_cat"] = pd.cut(df["Age"], bins= [0, 27, 57, 77, np.inf]
        , labels= ["7-27", "27-57", "57-77", "77-85"])


en = LabelEncoder()
df['satisfaction'] = en.fit_transform(df['satisfaction'])


# %%
# stratified sampling by Age_cat
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=101)
for train_index, group_index in split.split(df, df["Age_cat"]):
    strat_train_set = df.loc[train_index]
    strat_group_set = df.loc[group_index]
# test size = 0.2
split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=101)
for dev_index, test_index in split_2.split(strat_group_set, strat_group_set["Age_cat"]):
    strat_dev_set = strat_group_set.iloc[dev_index]
    strat_test_set = strat_group_set.iloc[test_index]

# %%
strat_train_set.drop("Age_cat", axis= 1, inplace= True)
strat_dev_set.drop("Age_cat", axis= 1, inplace= True)
strat_test_set.drop("Age_cat", axis= 1, inplace= True)

# %% [markdown]
# #### Đổi tên các tập dữ liệu

# %%
# Tập huấn luyện
y_train = strat_train_set['satisfaction']
X_train = strat_train_set.drop("satisfaction",axis =1 )

# %%
# Tập phát triển
y_dev = strat_dev_set['satisfaction']
X_dev = strat_dev_set.drop("satisfaction",axis = 1)

# %%
# Tập kiểm thử
y_test = strat_test_set['satisfaction']
X_test = strat_test_set.drop("satisfaction",axis = 1)

# %% [markdown]
# ### 2. Xây dựng Pipeline xử lý dữ liệu cho mô hình

# %%
# Rút trích tên cột các thuộc tính phân loại và số liệu
col_cat = X_train.columns[X_train.dtypes==object]
col_num = X_train.columns[X_train.dtypes!=object]

# %%
onehot_col = ['Gender', 'Customer_Type', 'Type_of_Travel']
ordi_col = col_cat.drop(['Gender', 'Customer_Type', 'Type_of_Travel'])

# %% [markdown]
# #####  Pipeline cho số liệu

# %%
#pipeline handle missing values and scaling feature
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="mean")),
 ('std_scaler', MinMaxScaler()),
 ])

# %% [markdown]
# ##### Pipeline tổng thể

# %%
# Pipeline tổng thể xử lý dữ liệu phân loạivà số học
full_pipeline = ColumnTransformer([
         ("num", num_pipeline, col_num),
         ("ordi",OrdinalEncoder(), ordi_col),
         ("onehot",OneHotEncoder(), onehot_col)
 ])

# %%
full_pipeline.fit(X_train)

# %% [markdown]
# Xử lý tên cột cho các thuộc tính khi đi qua Pipeline

# %%
col_name = list(col_num) + list(ordi_col)
full_col_name = col_name + list(full_pipeline.named_transformers_.onehot.get_feature_names())

# %%
# Đưa các tập dữ liệu qua Pipeline
X_train= full_pipeline.transform(X_train)
X_train = pd.DataFrame(X_train,columns=full_col_name)


X_test = full_pipeline.transform(X_test)
X_test = pd.DataFrame(X_test,columns=full_col_name)

X_dev = full_pipeline.transform(X_dev)
X_dev = pd.DataFrame(X_dev,columns=full_col_name)


# %%

from sklearn.ensemble import RandomForestClassifier

rdf_model = RandomForestClassifier(n_jobs= -1, random_state= 101,
                                 bootstrap= True, 
                                 max_features= 5,
                                 max_depth= 30,
                                 min_samples_split= 2,
                                 n_estimators= 120,
                                  verbose= 0)

# %%
rdf_model.fit(X_train, y_train)

# %%
rdf_model

# %%

y_proba_rdf = rdf_model.predict(X_test)
print(classification_report(y_proba_rdf, y_test))

import pickle
pickle.dump(rdf_model,open('model-airline.pkl','wb'))

# %%
pickle.dump(full_pipeline,open('Prepare_data.pkl','wb'))
# %%
