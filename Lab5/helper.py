
import numpy as np

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as ctb
import lightgbm as lgb

from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def get_models(use_dummy):
    
    models = [('dummy', DummyClassifier(strategy='stratified'))] if use_dummy else []

    models += [
        ('svm', SVC()),
        ('naive bayes', MultinomialNB()),
        ('decision tree', DecisionTreeClassifier(max_depth=5)),
        ('random forest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ('xgboost', xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ('lightgbm', lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=0)),
        ('catboost', ctb.CatBoostClassifier(n_estimators=50, max_depth=5, random_state=0, verbose=False))
    ]
    
    return models


def plot_results(result, scoring):
    result = sorted(result, key=lambda x: x[1])

    ys = [i[1] for i in result]
    ys_std = [i[2] for i in result]
    xs_labels = [i[0] for i in result]
    xs = range(len(xs_labels))
    
    plt.figure(figsize=(15, 5))
    plt.title('best model={}, {}=mean: {}, std: {}'.format(xs_labels[-1], scoring, ys[-1], ys_std[-1] ), fontsize=14)
    plt.xlabel('models')
    plt.ylabel(scoring)
    plt.bar(xs, ys, yerr=ys_std)
    plt.xticks(xs, xs_labels, rotation=90)
    
    
def run_models(X, y, scoring, cv=3, plot_result=True, show_confusion_matrix=True, use_dummy=True):
    result = []
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    
    num_cols = 4
    if show_confusion_matrix:
        fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(15,10))


    for it, (model_name, model) in enumerate(get_models(use_dummy)):
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            mean = np.around( np.mean(scores), 2)
            std = np.around( np.std(scores), 2)

            print("model={}, {}: mean={}, std={}".format(model_name, scoring, mean, std))

            result.append((model_name, mean, std))

            if show_confusion_matrix:
                y_pred = cross_val_predict(model, X, y, cv=cv)
                ax = axes[it // num_cols, it % num_cols]
                plot_confusion_matrix(y, y_pred, ax=ax, title='model: {}'.format(model_name))

        except Exception as e:
            print(f"!! Model={model_name}, failed with error={e}")
            
    if show_confusion_matrix:
        plt.tight_layout()  
        plt.show()

        
        
    if plot_result:
        plot_results(result, scoring)
        
        
def use_vectorizer_and_run_models(text, y, vectorizer, vectorizer_kwargs, kwargs, print_use_tokens=True):
    vec = vectorizer(**vectorizer_kwargs)
    X = vec.fit_transform(text).toarray()
    
    if print_use_tokens:
        tokens = vec.get_feature_names_out()
        print("tokens #{}: {}".format(len(tokens), tokens) )
        
    
    return run_models(X, y, **kwargs)
