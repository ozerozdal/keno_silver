import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np
import seaborn as sns
import pandas as pd
import os
import imblearn
import catboost
import fasttreeshap
import warnings
from sklearn.svm import SVC

def plot_permutation_importance(estimator, X_train, y_train, scoring=None, 
                                     n_repeats=5, random_state=42, n_jobs=-1, **kwargs):
    """
    Create a permutation importance plot.
    Reference : https://www.kaggle.com/dansbecker/permutation-importance
    Parameters
    ----------
    estimator : object
        a pre-fitted estimator (can be a pipeline)
    X_train : ndarray or DataFrame
        Data on which permutation importance will be computed.
    y_train : array-like or None
        Targets for supervised or None for unsupervised.
    scoring : str, callable, list, tuple, or dict, default=None
        Scorer to use. Refer to sklearn original documentation
    n_repeats : int, default=5
        Number of times to permute a feature
    random_state : int
        Random seed.
    n_jobs : int or None, default=-1
        Number of jobs to run in parallel.
    output_directory : str
        The path of the plot
    title : str
        Title of the plot
    set_size : tuple
        Set the figure size in inches -> Example: (15, 11)
        
    Returns
    -------
    Examples
    --------
    Eg 1:
    from GeoDS.prospectivity import featureimportance as fe
    trial_name = 'RawBaseline_Amaruq_May26'
    reporting_folder = os.path.join(trial_name, 'reporting_folder/')
    fe.plot_permutation_importance(pipeline, X_test, y_test, output_directory=reporting_folder, 
                            title= 'Permutation_Importances_Test_Set',
                            set_size=(15,15)) 
    """

    result = permutation_importance(estimator=estimator, X=X_train, y=y_train, scoring=scoring, n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=n_jobs)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_train.columns[sorted_idx])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    else:
        ax.set_title("Permutation Importances (Train Set)")

    if "set_size" in kwargs:
        fig.set_size_inches(kwargs['set_size'][0], kwargs['set_size'][1])
    else:
        fig.set_size_inches(15, 11)

    fig.tight_layout()
    plt.show()

    if "output_directory" in kwargs and "title" in kwargs:
        fig.savefig(kwargs['output_directory'] + kwargs['title'], dpi=100)

    return fig, ax


def plot_shap_values(model, X_test, combine_cat_class_importances = True, **kwargs):
    """
    Create a plot of shap values for a given pre-fitted estimator. SHAP VALUES DO NOT SUPPORT STACKING CLASSIFIERS
    Parameters
    ----------
    model : imblearn.pipeline.Pipeline model or sklearn classifier
    X_test : pandas.DataFrame
        The test portion of a dataset
    combine_cat_class_importances : OPTIONAL, boolean, default TRUE
        For a non-CatBoost model (Random Forest, LGBM, XGBoost) with some categorical variables,
        It sum up the categorical shap values after One Hot Encoding is applied.
        If you want to see the shap values for each OneHotEncoded feature separately, set it to False.            
    set_size : OPTIONAL, tuple
        Set the figure size in inches -> Example: (15, 11)
    output_directory : OPTIONAL, str
        The path of the plot
    """
    # REFERENCES :
    # https://github.com/slundberg/shap/issues/406
    # PROBLEM #1 : plot shap_values was plotting categorical variables in gray (see ref above)
    # FIXED #1 : PROBLEM #1 has been resolved. 'features' object of shap.summary_plot must be a numpy array, not a dataframe.

    # PROBLEM #2 : Plot shap values of OneHotEncoded columns in terms of original categorical feature (Fixed!)
    
    # PROBLEM #3 : We will update the code for more than one raw categorical features (Next update!)
    
    if type(model) == imblearn.pipeline.Pipeline:
        # If the user is using a pipeline model, 
        # the shap values are calculated in this if block!

        pre_model = model['pre'] # Pre step of the pipeline
        classifier = model['clf'] # Classifier of the pipeline
        ct = model.named_steps['pre'] # Define the column transform for the given pipeline model          
        
        if type(classifier) == catboost.core.CatBoostClassifier:
            explainer = fasttreeshap.TreeExplainer(classifier, feature_names=X_test.columns.tolist())
            (n, d) = X_test.shape
            
        elif type(classifier) == SVC:   
            svm_explainer = shap.KernelExplainer(classifier.predict, X_test)
            (n, d) = X_test.shape
       
        elif type(classifier) != catboost.core.CatBoostClassifier and type(classifier) != SVC:
            # When the user uses a non-CatBoost model (Random Forest, LGBM, XGBoost) with some categorical variables, 
            # the "pre" step of the pipeline applies OneHotEncoder for categorical variables. 
            # In other words, the dimension of X_train and X_test change at the "pre" stage of the pipeline. 
            # For this reason, the columns of X_test are not recognized by the bestpipeline_fitted_model, 
            # but the columns of the column-transformed version are. 
            # Therefore, we need to apply a column transform first and then check the shap values of individual OneHotEncoder columns.
                                    
            X_test_transform = pd.DataFrame(ct.transform(X_test), columns= pre_model.get_feature_names_out().tolist())    
            explainer = fasttreeshap.TreeExplainer(classifier, feature_names=X_test_transform.columns.tolist())
            (n, d) = X_test_transform.shape
                
    else:
        explainer = fasttreeshap.TreeExplainer(model, feature_names=X_test.columns.tolist())
        (n, d) = X_test.shape    

    # Plotting part starts here!
    # The plotting part can be written in a cleaner way.
    # However, shap.summary_plot calls 'plt.show()' to ensure the plot displays, which makes it
    # a bit difficult to save two consecutive summary plots.
    # Therefore, we have to set 'show=False' for the first shap.summary_plot.
    # If an output_directory is set, only the second plot will be displayed, but both plots will be saved separately.
    # If an output_directory is NOT set, then both plots will be displayed. However, none of them will be saved.
    
    if "output_directory" in kwargs:
        show=False
        if not os.path.exists(kwargs['output_directory']):
            os.makedirs(kwargs['output_directory'])
    else:
        show=True    
    
    # Let's compute shap values and plot the first summary_plot
    if type(model) == imblearn.pipeline.Pipeline:
        if type(classifier) == catboost.core.CatBoostClassifier:
            shap_vals = explainer.shap_values(X_test)
            fasttreeshap.summary_plot(shap_vals, features=np.array(X_test), feature_names=X_test.columns, max_display = d, show=show)
        elif type(classifier) == SVC:
            shap_vals = svm_explainer.shap_values(X_test)
            shap.summary_plot(shap_vals, features=np.array(X_test), feature_names=X_test.columns)
        elif combine_cat_class_importances and type(classifier) != catboost.core.CatBoostClassifier and type(classifier) != SVC:
            # The shap values of the OneHotEncoder columns will be written 
            # in terms of the original categorical feature.    
            warnings.warn("The current version of plot_shap_values sum the shap values of OneHotEncoded columns by default." + os.linesep +
                "However, it assumes that the data cube has only one categorical property." + os.linesep +
                "If you have more than one categorical features, please combine_cat_class_importances = False." + os.linesep +
                "In the next update the function will work with more than one original categorical features.")
                        
            shap_vals = explainer.shap_values(X_test_transform)
            num_features_indices = []
            cat_features_indices = []
            arr_combined_list_list = []

            for count, element in enumerate(explainer.feature_names):
                if element.startswith('cat'):
                    cat_features_indices.append(count)
                else:
                    num_features_indices.append(count)

            for i in range(len(shap_vals)):
                shap_arr = shap_vals[i]
                j, k = shap_arr.shape
    
                arr_combined_list = []
    
                for j in range(j):
                    num_shap_arr = shap_arr[j, num_features_indices]
                    cat_shap_arr = shap_arr[j, cat_features_indices].sum()
                    arr_combined = np.append(num_shap_arr, cat_shap_arr)
                    arr_combined_list.append(arr_combined)
    
                arr_combined_arr = np.array(arr_combined_list)
                arr_combined_list_list.append(arr_combined_arr) # arr_combined_list_list is the modified version of shap_vals
            
            fasttreeshap.summary_plot(arr_combined_list_list, 
                          features=np.array(X_test), 
                          feature_names=X_test.columns, 
                          max_display = d, 
                          show=show)            
        
        elif combine_cat_class_importances == False and type(classifier) != catboost.core.CatBoostClassifier and type(classifier) != SVC:
            # If the user want to see the shap values for each OneHotEncoded feature separately, combine_cat_class_importances will be set to False.
            # In this case, the shap values will be calculated as follows:
            shap_vals = explainer.shap_values(X_test_transform)
            fasttreeshap.summary_plot(shap_vals, 
                                      features=np.array(X_test_transform), 
                                      feature_names=X_test_transform.columns, 
                                      max_display = d, 
                                      show=show)            
    else:
        shap_vals = explainer.shap_values(X_test)
        fasttreeshap.summary_plot(shap_vals, features=np.array(X_test), feature_names=X_test.columns, max_display = d, show=show)
    
    # "set_size" is OPTIONAL to change the size of the figures
    if "set_size" in kwargs:
        fig1 = plt.gcf()
        fig1.set_size_inches(kwargs['set_size'][0], kwargs['set_size'][1])
        fig1.tight_layout()
    else:
        fig1 = plt.gcf()
        fig1.set_size_inches(12, 12)
        fig1.tight_layout()

    if "output_directory" in kwargs:
        fig1.savefig(os.path.join(kwargs['output_directory'], 'shap_summary.png'))
        fig1.clf()

    # Let's plot the second summary_plot with a plot_type="bar"   
    if type(model) == imblearn.pipeline.Pipeline:
        if type(classifier) == catboost.core.CatBoostClassifier:
            fasttreeshap.summary_plot(shap_vals, 
                                      features=np.array(X_test), 
                                      feature_names=X_test.columns, 
                                      plot_type="bar", 
                                      max_display = d, 
                                      show=show)
        elif type(classifier) == SVC:
            shap.summary_plot(shap_vals, 
                              features=np.array(X_test), 
                              feature_names=X_test.columns,
                              plot_type="bar", 
                              max_display = d, 
                              show=show)
                                   
        elif combine_cat_class_importances and type(classifier) != catboost.core.CatBoostClassifier and type(classifier) != SVC:
            # The shap values of the OneHotEncoder columns will be written 
            # in terms of the original categorical feature.
            fasttreeshap.summary_plot(arr_combined_list_list, 
                          features=np.array(X_test), 
                          feature_names=X_test.columns,
                          plot_type="bar",
                          max_display = d, 
                          show=show)
        elif combine_cat_class_importances == False and type(classifier) != catboost.core.CatBoostClassifier and type(classifier) != SVC:
            # If the user want to see the shap values for each OneHotEncoded feature separately, combine_cat_class_importances will be set to False.
            # In this case, the shap values will be calculated as follows:
            shap_vals = explainer.shap_values(X_test_transform)
            fasttreeshap.summary_plot(shap_vals, 
                                      features=np.array(X_test_transform),
                                      feature_names=X_test_transform.columns,
                                      plot_type="bar", 
                                      max_display = d,
                                      show=show)
    else:
        fasttreeshap.summary_plot(shap_vals, features=np.array(X_test), feature_names=X_test.columns, plot_type="bar", max_display = d, show=show)
    
    # "set_size" is OPTIONAL to change the size of the figures
    if "set_size" in kwargs:
        fig2 = plt.gcf()
        fig2.set_size_inches(kwargs['set_size'][0], kwargs['set_size'][1])
        fig2.tight_layout()
    else:
        fig2 = plt.gcf()
        fig2.set_size_inches(12, 12)
        fig2.tight_layout()    
    
    if "output_directory" in kwargs:
        fig2.savefig(os.path.join(kwargs['output_directory'], 'shap_summary_barplot.png'))
        
    return


def plot_feature_importance(model, feature_names = None, combine_cat_class_importances = True, ntop=None, **kwargs):
    """
    Create feature importance using sklearn's ensemble models model.feature_importances_ property.
    Parameters
    ----------
    model : class
        any class from sklearn ensemble or tree familly, see https://scikit-learn.org/stable/search.html?q=feature_importances_
    feature_names : list, not required for pipeline models. It's required if the user wants to use a ensemble sklearn classifier.
        name of the variables
    set_size : OPTIONAL, tuple
        Set the figure size in inches -> Example: (15, 11)
    output_directory : OPTIONAL, str
        The path of the plot
    title : OPTIONAL, str
        The name of the plot
    combine_cat_class_importances : OPTIONAL, boolean, default TRUE
        For a non-CatBoost model (Random Forest, LGBM, XGBoost) with some categorical variables,
        It sum up the categorical features importance values after One Hot Encoding is applied.
        If you want to see the feature importance for each OneHotEncoded feature separately, set it to False.
    Returns
    -------
    fig : Matplotlib.Figure
        Figure from matplotlib
    Example
    -------
    from GeoDS.prospectivity import featureimportance as fe
    from joblib import dump, load
    trial_name = 'RawBaseline_Amaruq_May26'
    output_folder = os.path.join(trial_name, 'outputs/')
    RF_predictions_folder = os.path.join(trial_name, 'RF_predictions/')
    best_fitted_models = hyperparameterstuning.loadJobsFromFolder(output_folder+"/models/")
    
    for model in best_fitted_models.keys():
        model_path = output_folder + 'models/' + f'{model}.joblib
        if model == 'RandomForest_bestpipeline_fitted_model':
            RandomForest_bestpipeline_fitted_model = load(model_path)
            fe.plot_feature_importance_update(RandomForest_bestpipeline_fitted_model,
                          output_directory=RF_predictions_folder, 
                          title= 'RF_Feature_Importances.png')
    """
    if type(model) == imblearn.pipeline.Pipeline:
        # If the user is using a pipeline model, 
        # the importance of the feature is calculated in this if block!
                
        pre_model = model['pre'] # Pre step of the pipeline
        classifier = model['clf'] # Classifier of the pipeline
        ct = model.named_steps['pre'] # Define the column transform for the given pipeline model 
        
        # It's not required to set a list of feature_names if the user wants to use a pipeline model.
        # The following line will get the feature names.
        feature_names = pre_model.get_feature_names_out()
        feature_importance = np.array(classifier.feature_importances_)
        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)
        
        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
                        
        if combine_cat_class_importances and type(classifier) != catboost.core.CatBoostClassifier and 'cat' in ct.named_transformers_.keys():
            # When the user uses a non-CatBoost model (Random Forest, LGBM, XGBoost) with some categorical variables, 
            # the "pre" step of the pipeline applies OneHotEncoder for categorical variables. 
            # In other words, the dimension of X_train and X_test expands at the "pre" stage of the pipeline. 
            # For this reason, the columns of X_test are not recognized by the bestpipeline_fitted_model, 
            # but the columns of the column-transformed version are. 
            # Therefore, we first have to apply a column transform and then sum up the feature importance values of individual OneHotEncoder columns.
            
            # Original categorical features list. Categorical features before applying OneHotEncoder
            original_cat_features = ct.named_transformers_['cat'].feature_names_in_.tolist()
            
            # Categorical features list after applying OneHotEncoder
            all_cat_list = ct.named_transformers_['cat'].get_feature_names_out(original_cat_features).tolist()
                 
            # A for loop for original_cat_features to find the one hot encoded features corresponding to each original categorical feature   
            for original_cat_feature in original_cat_features:
                # List of one hot encoded features corresponding to each original categorical feature
                cat_list = [i for i in all_cat_list if i.startswith(original_cat_feature)]
                
                # OneHotEncoded columns must be renamed.
                # ct.named transformers['cat'].get_feature_names_out(original cat_features) returns column names missing "cat__" in front.
                # Let's fix it easily!
                for i, element in enumerate(cat_list):
                    cat_list[i] = 'cat__' + element
                
                # Slice fi_df dataframe to return the only rows for the associated OneHotEncoded features names (cat_list) 
                # and then sum the feature importance values
                cat_sum = fi_df[fi_df['feature_names'].isin(cat_list)]['feature_importance'].sum()
                
                # Slice fi_df dataframe to return the only rows other than categorical features.
                # In other words, dataframe with numerical features
                fi_df = fi_df[~fi_df['feature_names'].isin(cat_list)]
                # Create a temporary dictionary to return the originial categorical feature 
                # and the summation of OneHotEncoded features importances
                temp_dict = {'feature_names': original_cat_feature, 'feature_importance': cat_sum}
                # Append the temporary_dict to the dataframe
                fi_df = fi_df.append(temp_dict, ignore_index=True)
                
            # Sort the DataFrame in order decreasing feature importance
            fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
                    
    else:
        # If the user wants to use an ensemble sklearn classifier instead of using the pipeline, 
        # feature importance is calculated by running the else block!
        # It is required to set a list of feature_names if the user wants to use a ensemble sklearn classifier.
        # In this case, however, OneHotEncoded column contributions will be listed separately, 
        # regardless of the combine_cat_class_importances preference.
        # Example model types are as follows:
        #    catboost.core.CatBoostClassifier
        #    sklearn.ensemble._forest.RandomForestClassifier
        #    lightgbm.sklearn.LGBMClassifier
        # Example:
        # --------
        # plot_feature_importance_update(RandomForest_bestpipeline_fitted_model['clf'], 
        #        RandomForest_bestpipeline_fitted_model['pre'].get_feature_names_out(),
        #        output_directory=RF_predictions_folder, 
        #        title= 'RF_Feature_Importances.png')
                
        # Create arrays from feature importance and feature names
        feature_importance = np.array(model.feature_importances_)
        feature_names = np.array(feature_names)
        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)
        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        
    # Define size of bar plot
    fig, ax = plt.subplots()
            
    # Plot Searborn bar chart
    if ntop != None: ax = sns.barplot(x=fi_df['feature_importance'][:ntop], y=fi_df['feature_names'][:ntop])
    else: ax = sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    
    # Add chart labels
    if type(model) == imblearn.pipeline.Pipeline:
        model_name = str(type(classifier)).split(".")[-1][:-2]
    else:
        model_name = str(type(model)).split(".")[-1][:-2]
    plt.title(model_name + ' Feature Importances')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    if "set_size" in kwargs:
        fig.set_size_inches(kwargs['set_size'][0], kwargs['set_size'][1])
    else:
        fig.set_size_inches(10, 10)
        
    fig.tight_layout()
    
    if "output_directory" in kwargs and "title" in kwargs:
        fig.savefig(kwargs['output_directory'] + kwargs['title'])
    return