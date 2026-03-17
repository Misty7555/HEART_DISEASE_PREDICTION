from preprocessing import load_data
from linear_regression import run_linear_regression
from naive_bayes import run_naive_bayes
from kmean_clustering import run_kmeans
from kmedoids_clustering import run_kmedoids
from dbscan_clustering import run_dbscan
from visualization_dashboard import show_dashboard

def main():

    df = load_data()

    y_test_lr, pred_lr = run_linear_regression(df)
    y_test_nb, pred_nb = run_naive_bayes(df)
    X_kmeans, c_kmeans = run_kmeans(df)
    X_kmedoids, c_kmedoids = run_kmedoids(df)
    X_dbscan, c_dbscan = run_dbscan(df)

    show_dashboard(
        y_test_lr, pred_lr,
        y_test_nb, pred_nb,
        X_kmeans, c_kmeans,
        X_kmedoids, c_kmedoids,
        X_dbscan, c_dbscan
    )

if __name__ == "__main__":
    main()