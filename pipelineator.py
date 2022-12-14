"""
So far most of the models we train do follow combination of Standard Scaling + Imbalance Mitigation + Classifier, this applies to both prospectivity and smart map (supervised maps).
        The idea of this class is to allow our team to obtain basic, default pipelines with the usual ensemble methods that we are using in a simple command line.
        Also, it is imperative to have the flexibility to test different techniques within the same project. Therefore, each pipeline are following Imblearn Pipelines structure and additional steps in the pipelines can easily be added.
        Those pipelines will work great either with binary or multiclass classifications problems.
Examples
----------
            Get default pipelines prepared for a multiclass training.
            If you are accessing individual pipelines through the class attributes such as dp.catboost_pipeline, be sure to assign it a to a variable like in the below example.
            The class attributes are only returning a shallow copy so calling .fit directly on them won't work.
            >>> from sklearn.datasets import load_wine
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>> from GeoDS.supervised import pipelineator
            >>> X, y = load_wine(return_X_y = True, as_frame=True)
            >>> Z = X.dtypes.astype(str).sort_index().sort_values()
            >>> X = X.reindex(columns=Z.index)
            >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            >>> n_classes = len(np.unique(y))
            >>> cat_indices = X_train.columns.get_indexer(X_train.select_dtypes('object').columns)
            >>> dp = pipelineator.DefaultSupervisedPipeline(categorical_features_indices=cat_indices, objective='multiclass', lgbm_num_classes=n_classes)
            >>> catboost_pipeline = dp.catboost_pipeline
            >>> catboost_pipeline.fit(X_train, y_train)
            >>> y_pred = catboost_pipeline.predict(X_test)
            Get default pipelines but add a principal component with column dropper step. Let's say you visually assessed each individual PCs before and you want to drop pc 5, 6 and 7.
            You will ask to drop the columns with indices 5, 6 and 7.
            >>> from GeoDS.supervised import pipelineator
            >>> from sklearn.preprocessing import StandardScaler
            >>> from sklearn.decomposition import PCA
            >>> columns_to_drop = [5,6,7]
            >>> dropper = pipelineator.get_column_dropper(cols_indexes=columns_to_drop)
            >>> numerical_transformer = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA()), ('dropper', dropper)])
            >>> dp = pipelineator.DefaultSupervisedPipeline(numerical_transformer=numerical_transformer)
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as imbpipeline
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN

import xgboost
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from sklearn.ensemble import AdaBoostClassifier

#mapclass module could be merged at some point with Pipelineator
from GeoDS.supervised import mapclass

class DefaultSupervisedPipeline:

    def __init__(self,
                 numerical_transformer=None, categorical_transformer=None, imbalance_mitigator=None,
                 categorical_features_indices=None, objective='binary',
                 lgbm_num_classes=None, random_state=42):
        """
        DefaultSupervisedPipeline Object instanciator.
        Available models : CatBoost, LightGBM, XGBoost, AdaBoost, Random Forest
        Parameters
        ----------
        numerical_transformer : sklearn.pipeline.Pipeline, default=None
            By default, it is only a Standard Scaler that is applied on numerical columns. You can customize it with any Pipeline object.
        categorical_transformer : sklearn.pipeline.Pipeline, default=None
            By default, we apply OneHotEncoding on categorical columns (except for CatBoost which handle the categories by itself), You can customize it with any Pipeline object.
        imbalance_mitigator : sklearn.base.BaseEstimator, default=None
            By default, we apply SMOTENN if there are no categorical columns, otherwise SMOTENC. For multiclass, it is our custom MapClassCustomMitigator class. You can cutomize it with any object that derived from sklearn.base.BaseEstimator and implements .fit_resample() and .resample()..
            Further reading : https://towardsdatascience.com/enrich-your-train-fold-with-a-custom-sampler-inside-an-imblearn-pipeline-68f6dff964bf
            Be sure to add only a sklearn.base.BaseEstimator, NOT A PIPELINE as you can't add a mitigator alone in a pipeline since they don't implement .fit
        categorical_features_indices : list, default=None
            Required for CatBoost only; indices of the categorical features such as [5,6,7]
        objective : str, default='binary'
            Required for LightGBM only; possible values are 'binary' or 'multiclass'.
        lgbm_num_classes : int, default=None
            Required for LightGBM only; for objective='multiclass', you need to specify the number of classes that are in your training set.
        random_state : int, default=42
            Random Seed
        """
        if categorical_features_indices is None:
            categorical_features_indices = []
        self.categorical_features_indices = categorical_features_indices

        # This is the oonly classifier that requires explicit user argument to tell between a binary/
        self.lgbm_num_classes = lgbm_num_classes
        self.objective = objective

        self.random_state = random_state

        self.numerical_transformer = numerical_transformer

        self.categorical_transformer = categorical_transformer

        # Create the preprocessing pipe of the numerical features and categorical features
        self.preprocessing = ColumnTransformer(transformers=[
            ('numerical_transfo', self.numerical_transformer, make_column_selector(dtype_include=np.number)),
            ('categorical_transfo', self.categorical_transformer, make_column_selector(dtype_include='object'))])

        self.imbalance_mitigator = imbalance_mitigator

        print('Important consideration if you are using the CatBoost Pipeline : '
              '1 - The input DataFrame (X) must be sorted out so that columns with numerical float/int dtypes are first then categorical columns with object dtypes are last. Use X = pipelineator.sort_columns_from_dtype(X)'
              )
        print('Given X, your DataFrame with your data, you will need to apply the two following lines : '
              'Z = X.dtypes.astype(str).sort_index().sort_values()'
              'X = X.reindex(columns=Z.index)')
        # https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        if value == 'binary':
            assert self.lgbm_num_classes == 2 or self.lgbm_num_classes is None
        elif value == 'multiclass':
            assert self.lgbm_num_classes > 2
        else:
            raise ValueError(
                "The objective argument must be specified in order to use LightGBM. Use 'binary' or 'multiclass'.")

        self._objective = value

    def summary(self):
        """
        Prints a summary of how the pipelines are built.
        """
        print('Your pipelines for Ensemble Learning - Supervised Classification')
        print('(1a) Pre-processing  for numerical columns :')
        print(self.numerical_transformer)
        print('(1b) Pre-processing  for categorical columns :')
        print(self.categorical_transformer)

        print('(2) Imbalance mitigation technique')
        print(self.imbalance_mitigator)

        print('(3) Classfiers available : ')
        print('CatBoost, LGBM, XGBoost, Random Forest, AdaBoost')

    @property
    def catboost_pipeline(self):
        pre = ColumnTransformer(
            transformers=[('num', self.numerical_transformer, make_column_selector(dtype_include=np.number))], remainder="passthrough")
        clf = CatBoostClassifier(cat_features=self.categorical_features_indices, verbose=False, allow_writing_files=False)


        #TODO
        #PH NOTE from 29 september; for the CatBoost pipeline,
        # we will bypass  for now the user-defined imbalance mitigator to SMOTENN / SMOTENC by default to CatBoost Multiclass because our custom imbalance mitigator do not work yet for multiclass with CatBoost.
        if(self.objective == 'binary'):
            imb = self.imbalance_mitigator
        elif(self.objective == 'multiclass'):
            if(self.categorical_features_indices is None or len(self.categorical_features_indices) == 0):
                imb = SMOTEENN(random_state=self.random_state)
            else:
                imb = SMOTENC(categorical_features=self.categorical_features_indices, random_state=self.random_state)
        else:
            raise ValueError("objective must be binary or multiclass. We need to know it for CatBoost imbalance pipeline choice.")


        pipe = imbpipeline([('pre', pre), ('imb', imb), ('clf', clf)])
        return pipe

    @catboost_pipeline.setter
    def catboost_pipeline(self, value):
        pass

    @property
    def random_forest_pipeline(self):
        clf = RandomForestClassifier(random_state=self.random_state)
        return imbpipeline([('pre', self.preprocessing), ('imb', self.imbalance_mitigator), ('clf', clf)])

    @property
    def lgbm_pipeline(self):
        if self.objective == 'multiclass':
            clf = lightgbm.LGBMClassifier(objective=self.objective, random_state=self.random_state,
                                          num_class=self.lgbm_num_classes)
        else:
            clf = lightgbm.LGBMClassifier(objective=self.objective, random_state=self.random_state)
        return imbpipeline([('pre', self.preprocessing), ('imb', self.imbalance_mitigator), ('clf', clf)])

    @property
    def xgboost_pipeline(self):
        clf = xgboost.XGBClassifier(random_state=self.random_state, use_label_encoder=True)
        return imbpipeline([('pre', self.preprocessing), ('imb', self.imbalance_mitigator), ('clf', clf)])

    @property
    def adaboost_pipeline(self):
        clf = AdaBoostClassifier(random_state=self.random_state)
        return imbpipeline([('pre', self.preprocessing), ('imb', self.imbalance_mitigator), ('clf', clf)])

    @property
    def numerical_transformer(self):
        return self._numerical_transformer

    @numerical_transformer.setter
    def numerical_transformer(self, value):
        if value is None:
            self._numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        else:
            self._numerical_transformer = value

    @property
    def categorical_transformer(self):
        return self._categorical_transformer

    @categorical_transformer.setter
    def categorical_transformer(self, value):
        if value is None:
            # transformer en pipeline sklearn
            self._categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
        else:
            self._categorical_transformer = value

    @property
    def imbalance_mitigator(self):
        return self._imbalance_mitigator

    @imbalance_mitigator.setter
    def imbalance_mitigator(self, value):
        if value is None:
            if(self.objective == 'binary'):
                if len(self.categorical_features_indices) == 0:
                    self._imbalance_mitigator = SMOTEENN(random_state=self.random_state)
                else:
                    self._imbalance_mitigator = SMOTENC(categorical_features=self.categorical_features_indices,
                                                        random_state=self.random_state)
            elif(self.objective == 'multiclass'):
                self._imbalance_mitigator = mapclass.MapClassCustomMitigator()
            else:
                raise ValueError("Cannot set the default imbalance mitigator properly as the objective is not 'binary' or 'multiclass'")
        else:
            self._imbalance_mitigator = value

    def get_all_pipelines(self):
        """
        Obtain all the constructed pipelines, one for each of our usual Ensemble Classifiers.
        Returns
        -------
        list
            A list where each element is a imblearn.Pipeline object.
        """
        return [self.catboost_pipeline, self.random_forest_pipeline, self.lgbm_pipeline, self.xgboost_pipeline,
                self.adaboost_pipeline]


def get_column_dropper(cols_indexes):
    """
    Static method to get a classic column dropper sklearn.compose.ColumnTransformer.
    For PCA and ICA evaluation workflow, if you need to drop certain columns, you can use this function to get this step taht can be added to the pipelines.
    Parameters
    ----------
    cols_indexes : list
        The indices of the columns which need to be dropped.
    Returns
    -------
    sklearn.compose.ColumnTransformer
        The column transformer that will drop certain columns and let the other pass through.
    """
    dropper = ColumnTransformer([("dropper", "drop", cols_indexes)], remainder="passthrough")
    return dropper


if __name__ == "__main__":
    import doctest
    doctest.run_docstring_examples('pipelineator.py')