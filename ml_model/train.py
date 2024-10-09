#training model using mall data on kmeans clustering
import pandas as pd
from sklearn.cluster import KMeans
import joblib

def train_model():
    loaded_data = pd.read_csv('D:\mall-segmentation\ml_model\data\mall.csv')
    X = loaded_data.iloc[:, [3, 4]].values
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    joblib.dump(kmeans, 'ml_model/kmeans_model.pkl')
    return kmeans

if __name__ == '__main__':
    train_model()
    print('Model trained successfully')
