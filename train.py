print('Importing libraries...')
import joblib
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
print('Done!')

DATASET_PATH = "src/winequality.csv"
print('Importing Dataset...')
df = pd.read_csv(DATASET_PATH)
print(df.head(5))
print('Done...!')


X = df.drop(["quality", "type"], axis=1)
y = df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                        ("classifier", RandomForestClassifier())])

model.fit(X_train, y_train)

print("Accuracy: {:.2f}".format(model.score(X_test, y_test)))

# Create the folder "models" if not exists
if not os.path.exists("models"):
    os.makedirs("models")


filename = "model.joblib"
PATH = ('D:/00_PROYECTOS_PERSONALES/DATA SCIENCE/01 - JEDHA FULL-STACK/00-Projects/05 - Bloc n°5 Industrialisation d un algorithme d apprentissage automatique et automatisation des processus de décision/ML in Production/models')



if filename in os.listdir(PATH):
    pass
else:
    joblib.dump(model, "models/model.joblib")