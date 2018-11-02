import math
import matplotlib.pyplot as plt
import numpy as np
import util

def dot_kernel(a, b):
    """An implementation of a dot product kernel.

    Args:
        a: A vector
        b: A vector
    """
    return np.dot(a, b)

def rbf_kernel(a, b, sigma=1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)

def train(pct, x_train, y_train, x_eval, y_eval):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/output/p05_{kernel_name}_predictions.txt.
    The output plots are saved to src/output_{kernel_name}_output.pdf.

    Args:
        kernel_name: The name of the kernel.
        kernel: The kernel function.
        learning_rate: The learning rate for training.
    """
    for x, y in zip(x_train, y_train):
        pct.update(x, y)

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda x: pct.predict(x))
    util.plot_points(x_eval, y_eval)
    plt.savefig('./output/p05_{}_output.pdf'.format(pct.kernel_name))

    y_pred = [pct.predict(x_eval[i, :]) for i in range(y_eval.shape[0])]
    np.savetxt('./output/p05_{}_predictions'.format(pct.kernel_name), y_pred)

def main(train_path, eval_path):
    x_train, y_train = util.load_csv(train_path)
    x_eval, y_eval = util.load_csv(eval_path)

    pct = Perceptron('dot', dot_kernel)
    train(pct, x_train, y_train, x_eval, y_eval)

    pct = Perceptron('rbf', rbf_kernel)
    train(pct, x_train, y_train, x_eval, y_eval)


class Perceptron():
    def __init__(self, kernel_name, kernel, step_size=0.5):
        self.kernel_name = kernel_name
        self.kernel = kernel
        self.step_size = step_size
        self.x = []
        self.a = []

    def sign(self, a):
        """Gets the sign of a scalar input."""
        return int(a >= 0)

    def update(self, x, y):
        """Updates the state of the perceptron.

        Args:
            x: A vector containing the features for a single instance
            y: A 0 or 1 indicating the label for a single instance
        """
        theta_phi_x = self.step_size * sum(self.a[j]*(self.kernel(self.x[j], x)+1) for j in range(len(self.a)))
        a = y - self.sign(theta_phi_x)
        self.x.append(x)
        self.a.append(a)

    def predict(self, x):
        """Peform a prediction on a given instance x given the current state
        and the kernel.

        Args:
            x: A vector containing the features for a single instance

        Returns:
            Returns the prediction (i.e 0 or 1)
        """
        theta_phi_x = sum(self.a[j]*(self.kernel(self.x[j], x)+1) for j in range(len(self.a)))
        return self.sign(theta_phi_x)


if __name__ == "__main__":
    main(train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_train.csv')

