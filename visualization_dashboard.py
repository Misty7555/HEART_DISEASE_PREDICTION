import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np

def show_dashboard(y_test_lr, pred_lr, y_test_nb, pred_nb,
                   X_kmeans, c_kmeans,
                   X_kmedoids, c_kmedoids,
                   X_dbscan, c_dbscan):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ------------------ Linear Regression ------------------
    axes[0, 0].scatter(y_test_lr, pred_lr, alpha=0.6)
    axes[0, 0].plot([min(y_test_lr), max(y_test_lr)],
                    [min(y_test_lr), max(y_test_lr)], linestyle='--')
    axes[0, 0].set_title("Linear Regression")
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")

    # ------------------ Residual Plot ------------------
    residuals = y_test_lr - pred_lr
    axes[0, 1].scatter(pred_lr, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, linestyle='--')
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residuals")

    # ------------------ Confusion Matrix ------------------
    cm = confusion_matrix(y_test_nb, pred_nb)
    im = axes[0, 2].imshow(cm)
    axes[0, 2].set_title("Naive Bayes CM")

    # Add values inside confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 2].text(j, i, cm[i, j], ha='center', va='center')

    # ------------------ PCA Function ------------------
    def plot_pca(ax, X, labels, title):

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        labels = np.array(labels)

        # Handle DBSCAN noise (-1)
        unique_labels = np.unique(labels)

        for lab in unique_labels:
            if lab == -1:
                # Noise points in black
                ax.scatter(X_pca[labels == lab, 0],
                           X_pca[labels == lab, 1],
                           color='black', label='Noise', alpha=0.5)
            else:
                ax.scatter(X_pca[labels == lab, 0],
                           X_pca[labels == lab, 1],
                           label=f'Cluster {lab}', alpha=0.6)

        ax.set_title(title)
        ax.legend(fontsize=8)

    # ------------------ Clustering Plots ------------------
    plot_pca(axes[1, 0], X_kmeans, c_kmeans, "KMeans")
    plot_pca(axes[1, 1], X_kmedoids, c_kmedoids, "KMedoids")
    plot_pca(axes[1, 2], X_dbscan, c_dbscan, "DBSCAN")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()