from sklearn.model_selection import cross_val_score, KFold

def cross_validate_model(model, X, y):
    kfold = KFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=kfold)
    return scores.mean(), scores.std()
