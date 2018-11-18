from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

def main(train_path, test_path):
    im_train = imread(train_path)
    im_test = imread(test_path)

    k = 6
    km = KMeans()
    km.train(k, im_train)
    im = km.compress(im_test)

    plt.imshow(im)
    plt.show()


class KMeans:
    def __init__(self, max_iter=100):
        self.max_iter = 100

    def initialize(self, k, im):
        randi = np.random.randint(im.shape[0], size=k)
        randj = np.random.randint(im.shape[1], size=k)
        self.centroids = np.vstack(im[i, j, :] for i, j in zip(randi, randj))

    def cost(self, v, axis=None):
        return np.sum(np.square(v), axis=axis)

    def train(self, k, im):
        im.astype(float)
        self.initialize(k, im)

        it = 0
        e = float('inf')
        while it < self.max_iter and e > 0:
            m = np.stack(self.cost(im - mu, axis=2) for mu in self.centroids)
            c = np.argmin(m, axis=0)
            centroids = np.array([np.sum(im[c==i], axis=0)/np.sum(c==i) for i in range(self.centroids.shape[0])])
            e =  self.cost(self.centroids - centroids)
            self.centroids = centroids
            it += 1

    def compress(self, im):
        im.astype(float)
        m = np.stack(self.cost(im - mu, axis=2) for mu in self.centroids)
        c = np.argmin(m, axis=0)
        im = self.centroids[c]
        return im.astype(np.uint8)

if __name__ == "__main__":
    main(train_path='../data/peppers-small.tiff',
         test_path='../data/peppers-large.tiff')
