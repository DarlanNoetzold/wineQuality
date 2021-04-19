import requests
import pandas as pd
import numpy as np
import pickle
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df_raw = pd.read_csv( csv_url, sep=';' )

df1 = df_raw.copy()

mms = MinMaxScaler()
df1['free sulfur dioxide'] = mms.fit_transform(df1[['free sulfur dioxide']].values)
pickle.dump(mms, open('free_sulfur_scaler.pkl', 'wb'))

df1['total sulfur dioxide'] = np.log1p(df1['total sulfur dioxide'])
df1['total sulfur dioxide'] = mms.fit_transform(df1[['total sulfur dioxide']].values)
pickle.dump(mms, open('total_sulfur_scaler.pkl', 'wb'))


train, test = train_test_split( df1 ) #75% e 25% test

# train and test dataset
x_train = train.drop( 'quality', axis=1 )
y_train = train['quality']

x_test = test.drop( 'quality', axis=1 )
y_test = test['quality']

# model definition
model = ElasticNet( alpha=0.5, l1_ratio=0.5, random_state=42 )

# training
model.fit(x_train, y_train)

# prediction
pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)


print('RMSE: {}'.format(rmse))
print('MAE: {}'.format(mae))
print('R2: {}'.format(r2))


# save trained model
pickle.dump(model, open('model_wine_quality.pkl', 'wb'))

df = df_raw.drop('quality', axis=1).sample()
df_json = df.to_json(orient='records')

url = 'https://wine-quality-model.herokuapp.com/predict'
data = df_json
header = {'Content-type': 'application/json'}

r = requests.post(url=url, data=data, headers=header)

print(r)
print(r.json())

