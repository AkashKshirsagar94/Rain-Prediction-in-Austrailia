from src.components.data_transformation import DataTransformation
import pandas as pd
import numpy as np


df=pd.read_csv('notebook/raw_data.csv')

preprocessing_obj = DataTransformation.get_data_transformer_object(self=df)

target_column_name = "RainTomorrow"
numeric_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 
                                'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 
                                'Humidity9am','Humidity3pm', 'Cloud9am', 
                                'Cloud3pm', 'Temp3pm']
categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm',
                                    'RainToday']
x = df.drop(columns=[target_column_name], axis=1)
y = df[target_column_name]
y = y.replace({'Yes':1, 'No':0})
#print(y.dtype)

y_arr = [np.array(y).T]

input_feature_arr=preprocessing_obj.fit_transform(x)

#print(input_feature_arr[0:5])
#print(np.array(y)[0:5])

arr = np.c_[input_feature_arr, y_arr]
#print(input_feature_arr.shape)
print(y-y_arr.shape,'\n', input_feature_arr.shape,'\n',arr.shape)


