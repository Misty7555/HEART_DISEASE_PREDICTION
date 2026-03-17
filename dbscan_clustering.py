from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_dbscan(df):

    X = df.drop("target", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=3, min_samples=5)

    clusters = model.fit_predict(X_scaled)

    print("\nDBSCAN Output")
    print(pd.Series(clusters).value_counts())

    return X_scaled, clusters