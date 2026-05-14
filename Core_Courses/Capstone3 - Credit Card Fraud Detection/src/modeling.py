from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def build_final_catboost():
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        verbose=0,
        random_state=42
    )

    return model


def evaluate_binary_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    return y_prob, metrics