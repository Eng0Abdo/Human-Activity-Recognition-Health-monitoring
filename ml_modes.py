# ml_models_final_ready.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

train = pd.read_csv("train_features.csv")
test = pd.read_csv("test_features.csv")

target_col = "Activity_ID" 

X_train = train.drop(columns=[target_col])
y_train = train[target_col]

X_test = test.drop(columns=[target_col])
y_test = test[target_col]

X_train = X_train.select_dtypes(include=['float64', 'int64'])
X_test = X_test[X_train.columns] 

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    
    if len(le.classes_) == 2:
        y_prob = model.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test_enc, y_prob)
    else:
        roc = "N/A (multi-class)"
    
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='weighted')
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC Score": roc
    })

results_df = pd.DataFrame(results)
print(results_df)