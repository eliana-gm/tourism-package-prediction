import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    data.drop(columns='CustomerID', inplace=True, errors='ignore')
    return data

def preprocess_data(data):
    y = data['ProdTaken']
    X = data.drop(columns=['ProdTaken', 'DurationOfPitch', 'NumberOfFollowups', 'ProductPitched', 'PitchSatisfactionScore'])

    # Handling missing values
    imputer_median = SimpleImputer(strategy='median')
    imputer_mode = SimpleImputer(strategy='most_frequent')

    median_cols = ['Age', 'MonthlyIncome', 'NumberOfTrips']
    mode_cols = ['TypeofContact', 'PreferredPropertyStar', 'NumberOfChildrenVisiting']

    X[median_cols] = imputer_median.fit_transform(X[median_cols])
    X[mode_cols] = imputer_mode.fit_transform(X[mode_cols])

    # Encoding and scaling
    X = pd.get_dummies(X, drop_first=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=1, stratify=y)

def build_model(X_train, y_train):
    model = SVC(kernel='rbf', probability=True, random_state=1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return classification_report(y_test, preds, output_dict=True), confusion_matrix(y_test, preds)

def save_model(model, filepath='scripts/final_model.joblib'):
    joblib.dump(model, filepath)

if __name__ == "__main__":
    data = load_data('data/tourism.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = build_model(X_train, y_train)
    report, cmatrix = evaluate_model(model, X_test, y_test)
    save_model(model)
    print("Model evaluation complete. Saved to disk.")
