import json

from joblib import dump, load
import glob
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import scipy
from optuna import pruners
from optuna.samplers import TPESampler
from scipy.stats import randint as sp_randint
import numpy as np

# from imblearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline as skpipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from GeoDS import utilities
from GeoDS.supervised import mapclass
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from catboost import Pool, CatBoostClassifier
from sklearn.svm import SVC
import xgboost
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
import lightgbm
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.ensemble import GradientBoostingClassifier
import joblib
# metric to optimize for the competition
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate

# to optimize the hyperparameters we import the randomized search class
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
import json


def loadJobsFromFolder(input_folder, kind='optuna_best_pipelines'):
    """
    Utility function to load previously saved job as .joblib files in a specific folder. It is a one-liner instead of using glob.
    Kind argument will be removed soon. Will
    Parameters
    ----------
    input_folder : str
        Folder where the job (.joblib files) are saved.
    kind : str
        Type of jobs we want to load. Not implemented for now. By default = optuna_best_pipelines. Other possible value is 'RandomizedGridSearchCV' .
    Returns
    -------
    dict
        list containing tuples of pipelines names / reloaded joblib file.Can be sent directly to the estimator argument of a stacking classifier.
    """

    jobs = glob.glob(input_folder + '*.joblib')
    if kind == 'optuna_best_pipelines':
        estimators = {}
        for j in jobs:
            name, extension, directory = utilities.Path_Info(j)
            estimators[name] = joblib.load(j)
        return estimators

    elif kind == 'RandomizedGridSearchCV':
        warnings.warn('RandomizedGridSearch wont be supported soon, kind argument will also be removed',
                      DeprecationWarning)
        pipes = {}
        for e in jobs:
            name, extension, directory = utilities.Path_Info(e)
            pipes[name] = joblib.load(e)
        return pipes


def _getAllScoresDict(cvresults, metrics):
    """
    PRIVATE Helper utility function to extract all splits from a RandomizedSearchCV  dict and convert them to a DataFrame.
    Now it only supports RandomizedSearchCV objects using multimetrics. Functionality could be extended for GridSearchCV, HavlinggridSearchCV and for single metric.
    :param cvresults: input RandomizedSearchCV  dictionary
    :type cvresults: dict
    :param metrics: names of the different metric used.
    :type metrics: list
    :return: DataFrame with metrics as the column headers and values will be those of the different splits from the cross-validation.
    :rtype: pandas.DataFrame
    """
    # If we use multiple metric in the randomized grid search -->
    # Get all the keys from CV result property, search for those name splitXX_test_score

    # This first block of code will extract all the scores from the various splits and will end with a dictionary such as
    # {'precision': ['split0_test_precision', 'split1_test_precision'....], 'recall' : ['split0_test_recall', 'split1_test_recall'....]}
    all_keys = cvresults.keys()
    scoresbymetric = {}
    for metric in metrics:
        temp_keys = []
        for key in all_keys:
            # this regex won't work if the Randomized Grid Search is mono metric : PH TO ADD
            # if mono metric, the key would be split_test_score instead
            string = "split[0-9]_test_" + metric
            match = bool(re.match(string, key))
            if (match):
                temp_keys.append(key)
        scoresbymetric[metric] = temp_keys

    # Now that we have all the scores, lets make a continous array of scores for each metric (merge folds together)
    # Resulting dict is {'{'precision': [0.903125, 0.8780487804878049, ...], 'recall': [0.9537953795379538, 0.9504950495049505, ...] '}
    data = {}
    for metric in metrics:

        data[metric] = []
        splits_list = scoresbymetric[metric]
        temp = list()
        for e in splits_list:
            for i in cvresults[e]:
                temp.append(i)
            data[metric] = temp

    # we now return a dataframe with a column for each metric. A dataframe per model
    df = pd.DataFrame.from_dict(data)
    return df


def _addClassifierNameCol(df, classifier_name):
    """
    PRIVATE Function to add classifier name to a DataFrame containing metrics scores from RandomizedGridSearch.
    :param df: A DataFrame containing metrics scores from RandomizedGridSearch.
    :type df: pandas.DataFrame
    :param classifier_name: Name of the classifier
    :type classifier_name: str
    :return: A DataFrame with a new column with the name of the Classifier in each row.
    :rtype: pandas.DataFrame
    """
    df['classifier'] = classifier_name
    return df


def _getClassifierName(clf):
    """
    PRIVATE Function to extract classifier name from a classifier object.
    :param clf: Input classifier object (CatBoostClassifier, per example)
    :type clf: Any classifier object that implements .__class__.__name__ attribute
    :return: Class name
    :rtype: str
    """
    name = clf.__class__.__name__
    to_remove = '()'
    # name variable will be CatBoostClassifier(). So parenthesis should be removed
    for i in to_remove:
        name.replace(i, '')
    return name

def compare_metrics(metrics, output_folder=None, save_csv=True, save_json=True, **kwargs):
    """
    Compare metrics and create a plot of metrics provided in the nested dictionary
    metrics dict is the output of hyperparameterstuning.get_results_from_studies written by Manit!
    In order for this function to work, the metrics format must be like the example below.
    
    Parameters
    ----------
    metrics : dictionary or path to the json file
        Dictionary with all models and related metrics
    set_size : OPTIONAL, tuple
        Set the figure size in inches -> Example: (15, 11)
    output_folder : OPTIONAL, str
        The path of the plot of metrics
    save_csv : True (By default)
        Saves the metrics dataframe if output_folder is provided by the user 
    save_json : True (By default)
        Saves the metrics dictionary as a JSON file if output_folder is provided by the user
    Example:
    
    metrics = {'CatBoost': {'fit_time': 7.048,
      'test_f1_macro': 0.5468,
      'train_f1_macro': 0.5767,
      'test_stdf1_macro': 0.0101,
      'train_stdf1_macro': 0.0032,
      'test_precision_macro': 0.5553,
      'train_precision_macro': 0.5827,
      'test_stdprecision_macro': 0.0086,
      'train_stdprecision_macro': 0.0058,
      'test_accuracy': 0.5782,
      'train_accuracy': 0.6034,
      'test_stdaccuracy': 0.0093,
      'train_stdaccuracy': 0.0031},
         'LGBM': {'fit_time': 1.0172,
      'test_f1_macro': 0.7446,
      'train_f1_macro': 0.8498,
      'test_stdf1_macro': 0.0142,
      'train_stdf1_macro': 0.0056,
      'test_precision_macro': 0.7223,
      'train_precision_macro': 0.8207,
      'test_stdprecision_macro': 0.0148,
      'train_stdprecision_macro': 0.0056,
      'test_accuracy': 0.7711,
      'train_accuracy': 0.8595,
      'test_stdaccuracy': 0.0148,
      'train_stdaccuracy': 0.0056},
         'RandomForest': {'fit_time': 52.1184,
      'test_f1_macro': 0.7428,
      'train_f1_macro': 0.8484,
      'test_stdf1_macro': 0.02,
      'train_stdf1_macro': 0.0026,
      'test_precision_macro': 0.7207,
      'train_precision_macro': 0.8199,
      'test_stdprecision_macro': 0.0199,
      'train_stdprecision_macro': 0.0028,
      'test_accuracy': 0.7687,
      'train_accuracy': 0.8555,
      'test_stdaccuracy': 0.0164,
      'train_stdaccuracy': 0.0029},
        'XGBoost': {'fit_time': 21.6675,
      'test_f1_macro': 0.7397,
      'train_f1_macro': 0.8486,
      'test_stdf1_macro': 0.0156,
      'train_stdf1_macro': 0.0063,
      'test_precision_macro': 0.7166,
      'train_precision_macro': 0.8196,
      'test_stdprecision_macro': 0.0153,
      'train_stdprecision_macro': 0.006,
      'test_accuracy': 0.7667,
      'train_accuracy': 0.8579,
      'test_stdaccuracy': 0.0139,
      'train_stdaccuracy': 0.007}}
    experiment_name = 'Update'
    output_folder = os.path.join(experiment_name, 'outputs/')
    if not os.path.exists(output_folder): os.makedirs(output_folder)  
   
    utilities.compare_metrics(metrics, output_folder=output_folder)
    
    """

    if type(metrics) == dict:
        # if metrics is a nested dictionary
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.reindex(metrics_df.mean().sort_values().index, axis=1)

        if output_folder != None and save_json == True:
            output_json_name = os.path.join(output_folder, 'metrics.json')
            with open(os.path.join(output_json_name), 'w') as file_json: 
                json.dump(metrics, file_json)

    elif type(metrics) == str:
        # If metrics is the path to the json file
        file, extension, directory = utilities.Path_Info(metrics)
        if (extension == '.json'):
            with open(metrics, 'r') as json_file: json_data = json.load(json_file)
            metrics_df = pd.DataFrame(json_data)
            metrics_df = metrics_df.reindex(metrics_df.mean().sort_values().index, axis=1)
    
    std_list = [] # List for standard deviations
    exclusion_list = [] # List used to eliminate fit time and standard deviations in the metrics dictionary
    
    # Let's find the indexes which contains 'std' and fit_time 
    for index in metrics_df.index.to_list():
        if 'std' in index or 'fit_time' in index:
            exclusion_list.append(index)
        if 'std' in index:
            std_list.append(index)
    
    # Let's define just_metric_df containing only metrics (No std and no fit_time!)
    just_metrics_df = metrics_df[~metrics_df.index.isin(exclusion_list)]
                
    if "set_size" in kwargs:
        ax = just_metrics_df.transpose().plot(figsize=(kwargs['set_size'][0], kwargs['set_size'][1]), zorder=10, kind='barh', xerr=np.array(metrics_df[metrics_df.index.isin(std_list)]), width=0.8)
    else:
        ax = just_metrics_df.transpose().plot(figsize=(12,8), zorder=10., kind='barh', xerr=np.array(metrics_df[metrics_df.index.isin(std_list)]), width=0.8)
        
    ax.set_title('Model Comparison', fontsize=20)
    ax.set_xlabel("Score", size = 20)
    ax.set_yticklabels(just_metrics_df.transpose().index.values.tolist(), rotation=45, 
                   rotation_mode="anchor", 
                   size = 15)
    ax.set_xlim([0.0, 1.0])
    ax.set_xticks(np.linspace(0, 1, 11, endpoint = True))
    ax.legend()
    ax.grid(zorder=5)
    leg = ax.legend(loc='upper left', fontsize=10, ncol=1, framealpha=.9)
    leg.set_zorder(100)
       
    if output_folder != None:
        if "title" in kwargs:
            plt.savefig(os.path.join(output_folder, kwargs['title']), dpi=100)
        else:
            plt.savefig(os.path.join(output_folder, 'model_comparison.png'), dpi=100)
      
        if save_csv == True: 
            metrics_df.to_csv(os.path.join(output_folder, 'metrics_df.csv'))
          
    return metrics_df

def CompareMetrics(jobs, metrics=['precision', 'recall', 'f1'], kind='violin'):
    """
    Function that produce either a violin or boxplot of the different models trained through RandomizedGridSearchCV with metrics as hue.
    Parameters
    ----------
    jobs : list
        A list containing each loaded joblib object (RandomizedGridSearch.cv)
    metrics : list
        A list of string of the various metrics. They need to be calculated in the RandomizedSearch
    kind : str
        type of plot. Supported : 'violin' or 'box' (Seaborn violinplot and boxplot)
    Returns
    -------
    Matplotlib.Axe
        Matplotlib Axe object of the plot
    """

    warnings.warn("Deprecated. Use Optuna methods instead.", DeprecationWarning)

    all_df = []
    for key in jobs:
        # 'clf' is hardcoded here. Be sure to use all the time clf
        classifier_name = _getClassifierName(jobs[key].best_estimator_['clf'])
        df = _getAllScoresDict(jobs[key].cv_results_, metrics)

        # This call switch the format of the dataframe so it suits violin or boxplot call
        df = df.melt(var_name='metrics', value_name='scores')
        df = _addClassifierNameCol(df, classifier_name)

        all_df.append(df)
    all_data = pd.concat(all_df)

    # NOW PLOTTING TIME : Use seaborn capabilities
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    fig, ax = plt.subplots()

    if (kind == 'violin'):
        g = sns.violinplot(x="classifier", y="scores", hue='metrics', data=all_data)
    elif (kind == 'box'):
        g = sns.boxplot(x="classifier", y="scores", hue='metrics', data=all_data)
    else:
        print("ERROR KIND ARGUMENT NOT SUPPORTED")

    ax.set_title('Scoring results for selected RandomizedGridSearchCV objects')
    return ax


def runTuning(pipelines, pipelines_names, X, y, target_layer, output_folder, train_size=0.8, average='binary'):
    """
    Function to fit + train many RandomizedGridSearchCV and saved the output into a joblib file.
    MANY IMPROVEMENTS CAN BE DONE FOR MORE FLEXIBILITY.
    -> Split for training and testing on X and y could be done prior to be sent to this function.
    -> use scores from inside randomized search cv object instead of using recall_score() or precision_score() functions
    -> reporting : Each trained joblib should record the following parameters
    - random state
    - train/test split : stratify, train_size
    - metrics calculated and their mean score
    - best params
    - target layer for training
    Parameters
    ----------
    pipelines : list
        A list containing a RandomizedGridSearchCV object, with all pre-processing steps and a final classifier.
    pipelines_names : list
        list of string containing the pipeline_names
    X : pandas.DataFrame
        input data (X)
    y : pandas.DataFrame
        labels (Y)
    target_layer : str
        target layer for training
    output_folder : str
        Output folder for saving joblib files.
    Returns
    -------
    """

    warnings.warn("Deprecated. Use Optuna methods instead. Use Optimizer() function", DeprecationWarning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=train_size,
        random_state=42)

    # pipeline_list = []

    print('Hyperparameter tuning in progress...')
    # best_precision = 0.0
    # best_clf = 0
    # best_gs = ''

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, rs in enumerate(pipelines):
        # estimator_name = self.get_estimator_name(rs.param_grid['clf'])
        print('\nEstimator: %s' % pipelines_names[idx])
        # Fit grid search
        rs.fit(X_train, y_train.values.ravel())

        # Predict on test data with best params (best estimator is automatically called by randomizedgridsearchcv)
        y_pred = rs.predict(X_test)
        # Test data precision of model with best params
        pre = precision_score(y_test, y_pred, average=average)
        rec = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)

        print(f'Test set precision score for best params: {pre}')
        print(f'Associated recall: {rec}')
        print(f'Associated f1: {f1}')

        if (isinstance(rs, imbpipeline) or isinstance(rs, skpipeline)):
            print("This iteration, a imbpipeline or sklearnpipeline was sent rather than a RandomizedGridSearchCV")
            print(
                "For example Gaussian Naive Bayesian do not require hyper-parameter tuning, therefore best params and best scores are not calculated. Job is saved anyway.")
            prefix = 'pipeline_only_'
        else:
            # Best params
            print('Best params: %s' % rs.best_params_)
            # Best training data precision
            print('Best training precision: %.3f' % rs.best_score_)
            print('----------------------------------------------------------------')
            prefix = 'best_rgs_'
        # Save the whole random search to file in the current working directory
        dump(rs, output_folder + prefix + pipelines_names[idx] + '_' + target_layer + '.joblib')

        del rs
    print("TUNING COMPLETED")

    return None


def default_rsg_param_distribution(classifier):
    """
    Get our default parameter distribution to be used in RandomizedGridSearchCV param_distributions argument.
    Note that every argument is preceded by clf _ _ . It is required to function into pipelines. Therefore, if you tweak the default pipelines,
    make sure that your new classifier step is still named 'clf'
    Parameters
    ----------
    classifier : str
        Name of the classifier
    Returns
    -------
    list
        a list containing a dictionary of the parameters, as required by RandomizedGridSearchCV param_distributions argument.
    """

    warnings.warn("Deprecated. Use Optuna methods instead.", DeprecationWarning)
    # Set ramdonmized search paramameters and distributions
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_log_dist = scipy.stats.loguniform(1e-4, 1e0)
    param_log_10 = scipy.stats.loguniform(1e-4, 1e1)
    param_log_100 = scipy.stats.loguniform(1e0, 1e3)
    param_uniary_dist = scipy.stats.uniform(0, 1)

    # Classifiers hyperparameters and their hyperparmeter search range
    rand_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                       'clf__min_samples_leaf': param_range,
                       'clf__max_depth': param_range,
                       'clf__min_samples_split': param_range[1:]
                       }]

    rand_params_lgbm = [{'clf__boosting_type': ['gbdt', 'dart'],
                         'clf__num_leaves': sp_randint(10, 300),
                         'clf__min_data': sp_randint(10, 100),
                         'clf__max_depth': sp_randint(5, 200),
                         'clf__reg_alpha': scipy.stats.uniform(0, 1),
                         'clf__reg_lambda': scipy.stats.uniform(0, 1),
                         'clf__learning_rate': scipy.stats.uniform(0, 1)
                         }]

    rand_params_catboost = [{"clf__learning_rate": np.linspace(0.01, 0.2, 5),
                             "clf__max_depth": sp_randint(3, 10)}]

    adab_base_estimators = []
    for i in range(1, 11):
        adab_base_estimators.append(DecisionTreeClassifier(max_depth=i))

    # ben oui toé https://stackoverflow.com/questions/28178763/sklearn-use-pipeline-in-a-randomizedsearchcv
    rand_params_adab = [{'clf__base_estimator': adab_base_estimators,

                         'clf__n_estimators': [10, 50, 100, 500, 1000],
                         'clf__learning_rate': scipy.stats.uniform(0, 1)}
                        ]

    classifiers = {
        'AdaBoost': rand_params_adab,
        'CatBoost': rand_params_catboost,
        'LGBM': rand_params_lgbm,
        'RandomForest': rand_params_rf
    }

    if (classifier in classifiers.keys()):
        return classifiers[classifier]
    else:
        print("Available classifiers")
        for i in classifiers.keys():
            print(i)
        raise Exception("The classifier you entered is not implemented yet.")


def make_RandomizedSearchGrids(pipeline_dict, scoring=['precision', 'recall', 'f1'], refit='precision', cv=5,
                               n_iter=10):
    """
    Prepare a list of RandomizedSearchGrids objects for a given set of estimator.
    Parameters
    ----------
    pipeline_dict : dict
        A dictionary where the key are the classifiers names and the value a pipeline. Typically returned from GeoDS.pipelineator.default_pipelines function
    scoring : list, default = ['precision', 'recall', 'f1']
        scoring metrics
    refit : str, default = 'precision'
        which metric to use to refit (see refit argument in sklearn docs for RandomizedSearchGridCV)
    cv : int
        number of cross-validations
    n_iter : int
        number of iterations
    Returns
    -------
    list
        list containing the various RandomizedSearchCV objects
    list
        list containing the name of the classifier
    """

    warnings.warn("Deprecated. Use Optuna methods instead.", DeprecationWarning)

    if ('GaussianNB' in pipeline_dict):
        del pipeline_dict['GaussianNB']
    names = []
    rscv = []

    for key in pipeline_dict:
        names.append(key)
        rscv.append(
            RandomizedSearchCV(estimator=pipeline_dict[key], param_distributions=default_rsg_param_distribution(key),
                               scoring=scoring, refit=refit, cv=cv, n_iter=n_iter))
    # Remove GaussianNB here

    return rscv, names


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="pipeline", value=trial.user_attrs["pipeline"])


def default_params(trial, classifier_name, random_state=42):
    param = {}
    if classifier_name == 'CatBoost':
        param = {
            "random_state": random_state,
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.5),
            "random_strength": trial.suggest_int("random_strength", 0, 100),
            "bagging_temperature": trial.suggest_loguniform(
                "bagging_temperature", 0.01, 100.00
            ),
            "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            "od_wait": trial.suggest_int("od_wait", 10, 50),
        }

    if classifier_name == 'RandomForest':
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt"]),
            "max_depth": trial.suggest_int("max_depth", 10, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": random_state
        }

    if classifier_name == 'LGBM':
        # min_data_in_leaf_min = 3
        # num_leaves * min_data_in_leaf <= nrows
        # num_leaves_max = min(2 ** param["max_depth"], int(len(X) / min_data_in_leaf_min))
        # param["num_leaves"] = trial.suggest_int("num_leaves", 10, num_leaves_max)
        # num_leaves * min_data_in_leaf <= nrows
        # min_data_in_leaf_max = int(len(X) / param["num_leaves"])
        # param["min_data_in_leaf"] = trial.suggest_int(
        #    "min_data_in_leaf", min_data_in_leaf_min, min_data_in_leaf_max

        param = {"boosting_type": trial.suggest_categorical("clf__bosting_type", ['gbdt', 'dart', 'goss']),
                 "num_leaves": trial.suggest_int('clf__num_leaves', 10, 300),
                 "min_data": trial.suggest_int('clf__min_data', 50, 200),
                 "max_depth": trial.suggest_int('clf__max_depth', 4, 200),
                 "reg_alpha": trial.suggest_uniform('clf__reg_alpha', 0, 1),
                 "reg_lambda": trial.suggest_uniform('clf__reg_lambda', 0, 1),
                 "learning_rate": trial.suggest_loguniform('clf__learning_rate', 1e-8, 1),
                 "random_state": random_state}

        if param["boosting_type"] == "dart":
            param["drop_rate"] = trial.suggest_loguniform("drop_rate", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
        if param["boosting_type"] == "goss":
            param["top_rate"] = trial.suggest_uniform("top_rate", 0.0, 1.0)
            param["other_rate"] = trial.suggest_uniform(
                "other_rate", 0.0, 1.0 - param["top_rate"])

    if classifier_name == 'XGBoost':
        param = {
            "random_state": random_state,
            'use_label_encoder': False,
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-9, 1),
            "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
            "subsample": trial.suggest_uniform("subsample", 0, 1),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0, 1),
        }
    if classifier_name == 'SVM':
        param = {
            "random_state": random_state,
            "probability": True,
            "C": trial.suggest_loguniform("C", 2 ** -10, 2 ** 10),
            "gamma": trial.suggest_loguniform("gamma", 2 ** -10, 2 ** 10),
            "kernel": trial.suggest_categorical("kernel", ['rbf', 'poly', 'linear', 'sigmoid'])
        }

    return param


def default_preprocessing(trial, classifier_name, numerical_features_names, categorical_features_indices, smart_map,
                          random_state=42):
    scaler = StandardScaler()

    classifiers_1 = ['LGBM', 'RandomForest', 'XGBoost', 'SVM']
    if classifier_name in classifiers_1:
        # Numerical transformer for rdf and lgbm
        numerical_transformer = Pipeline(steps=[
            ('scaler', scaler)])

        # categorical transformer is OHE. Ideally, we would have also generate frequency encoder and target encoder
        # or simply use Catboost for their native implentation, but time was limited
        if categorical_features_indices is None:
            pre = ColumnTransformer(transformers=[
                ('num', numerical_transformer, make_column_selector(dtype_include=np.number))])
        else:

            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            pre = ColumnTransformer(transformers=[
                ('num', numerical_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include='object'))])
        if smart_map:
            imb = mapclass.MapClassCustomMitigator()
        else:
            imb = SMOTEENN(random_state=random_state)


    elif classifier_name == 'CatBoost':
        # numerical_transformer = Pipeline(steps=[
        #     ('scaler', scaler)])

        # pre = ColumnTransformer(transformers=[('num', numerical_transformer, make_column_selector(dtype_include=np.number))])
        pre = ColumnTransformer(transformers=[
            ('scaler', scaler, numerical_features_names)], remainder='passthrough')
        if smart_map:
            imb = mapclass.MapClassCustomMitigator()
        else:
            if categorical_features_indices is None:
                imb = SMOTEENN(random_state=random_state)
            else:
                imb = SMOTENC(categorical_features=categorical_features_indices, random_state=random_state)

    return pre, imb


def default_objective(trial, classifier_name, numerical_features_names, categorical_features_indices, X_train, y_train,
                      cv, groups, scoring, smart_map, random_state=42):
    params = default_params(trial, classifier_name)

    print(params)
    if (classifier_name == 'CatBoost') and (
            categorical_features_indices is not None
    ):
        params['cat_features'] = categorical_features_indices

    pre, imb = default_preprocessing(trial, classifier_name, numerical_features_names, categorical_features_indices,
                                     smart_map)

    if classifier_name == 'RandomForest':
        clf = RandomForestClassifier(**params)
    elif classifier_name == 'LGBM':
        clf = lightgbm.LGBMClassifier(objective='binary', **params)
    elif classifier_name == 'XGBoost':
        clf = xgboost.XGBClassifier(**params)
    elif classifier_name == 'CatBoost':
        clf = CatBoostClassifier(**params, silent=True)
    elif classifier_name == 'SVM':
        clf = SVC(**params)

    pipeline = Pipeline(steps=[('pre', pre), ('imb', imb), ('clf', clf)])

    score = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, groups=groups, n_jobs=-1,
                           return_train_score=True)
    print(score)

    roc = score['test_f1_macro'].mean()
    trial.report(roc, trial.number)

    trial.set_user_attr(key="pipeline", value=pipeline)

    if trial.should_prune():
        raise optuna.TrialPruned()

    for i in scoring:
        trial.set_user_attr(key='fit_time', value=score['fit_time'].mean())
        trial.set_user_attr(key='test_{}'.format(i), value=score['test_{}'.format(i)].mean())
        trial.set_user_attr(key='train_{}'.format(i), value=score['train_{}'.format(i)].mean())
        trial.set_user_attr(key='test_std_{}'.format(i), value=score['test_{}'.format(i)].std())
        trial.set_user_attr(key='train_std_{}'.format(i), value=score['train_{}'.format(i)].std())
    return roc


def Optimizer(n_trials, models, output_folder, numeric_cols, categorical_features_indices, X_train, y_train, cv, groups,
              scoring=[],
              random_state=42, smart_map=False):
    """
    Function to launch studies and get best parameters for many classifiers.
    If cat_indexs are not present then explicitly assign cat_index parameter with None (i.e categorical_features_indices = None)
    Parameters
    ----------
    n_trials : int
        Number of trials
    models : list
        list of estimators (pipelines)
    output_folder : str
        where to save joblib files of each estimators with best hyperparameters set to the classfiier
    numeric_cols : list
        list of the numeric columns name
    categorical_features_indices : list
        list of columns indexes
    X_train : DataFrame or ndarray
        Training data
    y_train : DataFrame or ndarray
        Training labels
    cv : str
        sklearn based cross validation tecchnique
    groups : Pandas series or ndarray
        Define group for the sklearn based group cross validation techniques
    scoring : str
        name of the scoring metric https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
    random_state : int
        Random state
    Examples
    --------
    Eg 1:
    # Define sklearn based cv technique
    cross_validation = KFold(n_splits=5)
    models = ['CatBoost', 'RandomForest', 'SVM', 'LGBM', 'XGBoost']
    output_folder = 'output/'
    Optimizer(20, models, output_folder, numeric_cols, cat_indexes, X_train, y_train,cv =  cross_validation, groups = None, scoring="f1_macro", random_state=42)
    Eg 2:
    # For illustrative purposes we are using only 5 trials but in actual case we should use a high number like 50 or 100.
    # Define the groups that you want to use for stratified group k folds happening for model optimization
    from sklearn.model_selection import KFold
    groups =  Y_train['Cat_kmeans_geophys_stack']
    cross_validation = StratifiedGroupKFold(n_splits=5)
    models = ['CatBoost', 'RandomForest', 'SVM', 'LGBM', 'XGBoost']
    output_folder = 'output/'
    Optimizer(20, models, output_folder, numeric_cols, cat_indexes, X_train, y_train,cv=cross_validation,groups=groups, scoring="f1_macro", random_state=42)
    """
    #TODO: Test pruning of trials
    for m in models:
        name = m + "_study"
        name2 = m + "_bestpipeline_parameters"
        name3 = m + "_bestpipeline_fitted_model"
        study = optuna.study.create_study(study_name=name, sampler=TPESampler(seed=random_state),
                                          pruner=pruners.HyperbandPruner(min_resource=1, reduction_factor=3),
                                          direction="maximize")
        study.optimize(
            lambda trial: default_objective(trial, m, numeric_cols, categorical_features_indices, X_train, y_train, cv,
                                            groups, scoring, smart_map),
            n_trials=n_trials,
            callbacks=[callback])
        pipeline = study.user_attrs["pipeline"]

        if not os.path.exists(output_folder + 'studies/'):
            os.makedirs(output_folder + 'studies/')
        if not os.path.exists(output_folder + 'pipelines/'):
            os.makedirs(output_folder + 'pipelines/')
        if not os.path.exists(output_folder + 'models/'):
            os.makedirs(output_folder + 'models/')
        # Fit the pipeline with entire x_train and y_train. Previously pipeline object was traing using split part of x_train, y_train.
        fitted_pipeline = pipeline.fit(X_train, y_train)

        joblib.dump(study, output_folder + 'studies/' + name + '.joblib')
        joblib.dump(pipeline, output_folder + 'pipelines/' + name2 + '.joblib')
        joblib.dump(fitted_pipeline, output_folder + 'models/' + name3 + '.joblib')
        print("Trained models saved in {0} ".format(output_folder + 'models/'))


def get_results_from_studies(path_to_studies):
    """
    Helper function to fetch training and validation metrics from optuna study object
    Examples:
        studies_path = 'output/studies/'
        metrics = hyperparameterstuning.get_results_from_studies(studies_path)
    """
    try:
        studies = loadJobsFromFolder(path_to_studies)
    except:
        studies = path_to_studies

    final_dict = {}
    for j in studies:
        metrics = studies[j].best_trial.user_attrs
        del metrics['pipeline']
        K = 4

        # loop to iterate for values
        res = dict()
        for key in metrics:
            # rounding to K using round()
            res[key] = round(metrics[key], K)
        final_dict[j.split("_")[0]] = res
    return final_dict


def compare_models(metrics, output_folder=None, save_csv=True, save_json=True, **kwargs):
    """
    Compare models and create a plot of metrics provided in the nested dictionary
    Parameters
    ----------
    metrics : dictionary or path to the json file
        Dictionary with all models and related metrics
    set_size : OPTIONAL, tuple
        Set the figure size in inches -> Example: (15, 11)
    output_folder : OPTIONAL, str
        The path of the plot of metrics
    save_csv : True (By default)
        Saves the metrics dataframe if output_folder is provided by the user
    save_json : True (By default)
        Saves the metrics dictionary as a JSON file if output_folder is provided by the user
    Example:
    metrics = {'CatBoost': {'fit_time': 7.048,
      'test_f1_macro': 0.5468,
      'train_f1_macro': 0.5767,
      'test_stdf1_macro': 0.0101,
      'train_stdf1_macro': 0.0032,
      'test_precision_macro': 0.5553,
      'train_precision_macro': 0.5827,
      'test_stdprecision_macro': 0.0086,
      'train_stdprecision_macro': 0.0058,
      'test_accuracy': 0.5782,
      'train_accuracy': 0.6034,
      'test_stdaccuracy': 0.0093,
      'train_stdaccuracy': 0.0031},
         'LGBM': {'fit_time': 1.0172,
      'test_f1_macro': 0.7446,
      'train_f1_macro': 0.8498,
      'test_stdf1_macro': 0.0142,
      'train_stdf1_macro': 0.0056,
      'test_precision_macro': 0.7223,
      'train_precision_macro': 0.8207,
      'test_stdprecision_macro': 0.0148,
      'train_stdprecision_macro': 0.0056,
      'test_accuracy': 0.7711,
      'train_accuracy': 0.8595,
      'test_stdaccuracy': 0.0148,
      'train_stdaccuracy': 0.0056},
         'RandomForest': {'fit_time': 52.1184,
      'test_f1_macro': 0.7428,
      'train_f1_macro': 0.8484,
      'test_stdf1_macro': 0.02,
      'train_stdf1_macro': 0.0026,
      'test_precision_macro': 0.7207,
      'train_precision_macro': 0.8199,
      'test_stdprecision_macro': 0.0199,
      'train_stdprecision_macro': 0.0028,
      'test_accuracy': 0.7687,
      'train_accuracy': 0.8555,
      'test_stdaccuracy': 0.0164,
      'train_stdaccuracy': 0.0029},
        'XGBoost': {'fit_time': 21.6675,
      'test_f1_macro': 0.7397,
      'train_f1_macro': 0.8486,
      'test_stdf1_macro': 0.0156,
      'train_stdf1_macro': 0.0063,
      'test_precision_macro': 0.7166,
      'train_precision_macro': 0.8196,
      'test_stdprecision_macro': 0.0153,
      'train_stdprecision_macro': 0.006,
      'test_accuracy': 0.7667,
      'train_accuracy': 0.8579,
      'test_stdaccuracy': 0.0139,
      'train_stdaccuracy': 0.007}}
    experiment_name = 'Update'
    output_folder = os.path.join(experiment_name, 'outputs/')
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    utilities.compare_models(metrics, output_folder=output_folder)
    """
    if type(metrics) == dict:
        # if metrics is a nested dictionary
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.reindex(metrics_df.mean().sort_values().index, axis=1)
        if output_folder != None and save_json == True:
            output_json_name = os.path.join(output_folder, 'metrics.json')
            with open(os.path.join(output_json_name), 'w') as file_json:
                json.dump(metrics, file_json)
    elif type(metrics) == str:
        # If metrics is the path to the json file
        file, extension, directory = utilities.Path_Info(metrics)
        if (extension == '.json'):
            with open(metrics, 'r') as json_file: json_data = json.load(json_file)
            metrics_df = pd.DataFrame(json_data)
            metrics_df = metrics_df.reindex(metrics_df.mean().sort_values().index, axis=1)
    std_list = []  # List for standard deviations
    exclusion_list = []  # List used to eliminate fit time and standard deviations in the metrics dictionary
    # Let's find the indexes which contains 'std' and fit_time
    for i in metrics_df.index.to_list():
        if 'std' in i or 'fit_time' in i:
            exclusion_list.append(i)
        if 'std' in i:
            std_list.append(i)
    # Let's define just_metric_df containing only metrics (No std and no fit_time!)
    just_metrics_df = metrics_df[~metrics_df.index.isin(exclusion_list)]
    if "set_size" in kwargs:
        ax = just_metrics_df.transpose().plot(figsize=(kwargs['set_size'][0], kwargs['set_size'][1]), zorder=10,
                                              kind='barh', xerr=np.array(metrics_df[metrics_df.index.isin(std_list)]),
                                              width=0.8)
    else:
        ax = just_metrics_df.transpose().plot(figsize=(12, 8), zorder=10., kind='barh',
                                              xerr=np.array(metrics_df[metrics_df.index.isin(std_list)]), width=0.8)
    ax.set_title('Model Comparison', fontsize=20)
    ax.set_xlabel("Score", size=20)
    ax.set_yticklabels(just_metrics_df.transpose().index.values.tolist(), rotation=45,
                       rotation_mode="anchor",
                       size=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_xticks(np.linspace(0, 1, 11, endpoint=True))
    ax.legend()
    ax.grid(zorder=5)
    leg = ax.legend(loc='upper left', fontsize=10, ncol=1, framealpha=.9)
    leg.set_zorder(100)
    if output_folder != None:
        if "title" in kwargs:
            plt.savefig(os.path.join(output_folder, kwargs['title']), dpi=100)
        else:
            plt.savefig(os.path.join(output_folder, 'model_comparison.png'), dpi=100)
        if save_csv == True:
            metrics_df.to_csv(os.path.join(output_folder, 'metrics_df.csv'))
    return metrics_df