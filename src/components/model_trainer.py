import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, array):
        try:
            logging.info('splitting training & test input data')
            x_train, x_test, y_train, y_test = train_test_split(array[:,:-1], array[:,-1], 
                                                                test_size=0.2, random_state=10)

            models = {
                'Log regression': LogisticRegression(C=0.01, solver='liblinear'),
                'KNN': KNeighborsClassifier(n_neighbors=4),
                'Tree': DecisionTreeClassifier(criterion='entropy', max_depth=4),
                'SVM': svm.SVC(kernel='rbf'),
            }

            model_report: dict=evaluate_models(x_train=x_train, y_train=y_train, 
                                              x_test=x_test, y_test=y_test, models=models)
            
            #To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name fom dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info('Best model found on testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            acc_score = accuracy_score(y_test, predicted)
            print(acc_score)

            return acc_score, best_model_name

        except Exception as e:
            raise CustomException(e,sys)