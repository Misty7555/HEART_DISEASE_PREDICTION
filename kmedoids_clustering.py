from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings

def run_kmedoids(df):

    warnings.filterwarnings("ignore")

    X = df.drop("target", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMedoids(
        n_clusters=4,
        method='pam',
        init='k-medoids++',
        max_iter=500,
        random_state=42
    )

    clusters = model.fit_predict(X_scaled)

    print("\nKMedoids Output")
    print(pd.Series(clusters).value_counts())

    return X_scaled, clusters