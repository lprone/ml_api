import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
filtered_df = df[['Age', 'Sex', 'Embarked', 'Survived']]

categoricals = []
for col, col_type in filtered_df.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        filtered_df[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(filtered_df, columns=categoricals, dummy_na=True)

dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
logistic_regression = LogisticRegression()
logistic_regression.fit(x, y)

joblib.dump(logistic_regression, 'model.pkl')
print("Model dumped!")

logistic_regression = joblib.load('model.pkl')

model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
