import numpy as np
from PIL import ImageFile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ImageFile.LOAD_TRUNCATED_IMAGES = True
__all__ = ['Kmeans']

def preprocess_features(npdata, pca_dim=256, whitening=False, L2norm=False):
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32') # [3609, 512]
    pca = PCA(pca_dim, whiten=whitening)
    npdata = pca.fit_transform(npdata) # [3609, 256]
    if L2norm:
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]
    return npdata

class Clustering:
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        self.k = k
        self.pca_dim = pca_dim
        self.whitening = whitening
        self.L2norm = L2norm
        
    def cluster(self, data, verbose=False):
        xb = preprocess_features(data, self.pca_dim, self.whitening, self.L2norm)
        I = self.run_method(xb, self.k)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
        return None
    
    def run_method(self):
        print('Define each method')
    
class Kmeans(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters): # [3609, 256], 4
        kmeans = KMeans(n_clusters=n_clusters)
        I = kmeans.fit_predict(x) # [3609]
        return I