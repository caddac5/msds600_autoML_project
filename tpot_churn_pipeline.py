import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

def make_predictions(df):

    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    tpot_data = pd.read_csv('../prepped_churn_data.csv', index_col='customerID')
    features = tpot_data.drop(['Churn','zscore_tenure','zscore_monthlyCharge','zscore_totalCharge','chargeDiff','avgCharges'], axis=1)
    training_features, testing_features, training_target, testing_target = \
                train_test_split(features, tpot_data['Churn'], random_state=42)

    # Average CV score on the training set was: 0.7983701728734842
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=0.75, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=100.0)),
        ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=5, min_samples_split=12, n_estimators=100)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)


    predictions = exported_pipeline.predict(df)
    return predictions


if __name__ == '__main__':
    df = pd.read_csv('new_churn_data_unmodified.csv', index_col='customerID')
    df.PhoneService = df.PhoneService.replace({'No':0,'Yes':1})
    df.Contract = df.Contract.replace({'Month-to-month':0,'One year':1,'Two year':2})
    df.PaymentMethod = df.PaymentMethod.replace({'Mailed check':0,'Electronic check':1,'Bank transfer (automatic)':2,'Credit card (automatic)':3})
    predictions = make_predictions(df)
    print(predictions)