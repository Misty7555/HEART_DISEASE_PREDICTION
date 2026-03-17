from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_kmeans(df):

    X = df.drop("target", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=4, random_state=42)

    clusters = model.fit_predict(X_scaled)

    print("\nKMeans Output")
    print(pd.Series(clusters).value_counts())

    return X_scaled, clusters