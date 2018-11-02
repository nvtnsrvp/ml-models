import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    fn, ext = pred_path.split('.')
    plot_path = '.'.join([fn, 'png'])

    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    gda = GDA()
    gda.fit(x_train, y_train)
    y_pred = gda.predict(x_eval)

    util.plot(x_eval, y_pred, gda.theta, plot_path, correction=1.0)


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        def sum_outer(x):
            m, n = x.shape
            return sum(np.outer(x[i, :], x[i, :]) for i in range(m))

        m, n = x.shape
        sum_y1 = np.sum(y)
        self.phi = sum_y1 / m
        self.mu_0 = np.sum(x[y==0], axis=0) / (m-sum_y1)
        self.mu_1 = np.sum(x[y==1], axis=0) / sum_y1

        mu_y = np.tile(self.mu_0, (m, 1))
        mu_y[y==1] = self.mu_1
        self.sigma = sum_outer(x - mu_y) / m
        sigma_i = np.linalg.inv(self.sigma)

        theta = np.dot(self.mu_0 - self.mu_1, sigma_i)
        theta_0 = 0.5*(-self.mu_0.T.dot(sigma_i).dot(self.mu_0) + self.mu_1.T.dot(sigma_i).dot(self.mu_1)) + np.log(1-self.phi) - np.log(self.phi)
        self.theta = np.insert(theta, 0, theta_0)

        if self.verbose:
            print('mu_0', self.mu_0)
            print('mu_1', self.mu_1)
            print('sigma\n', self.sigma)
            print('theta\n', self.theta)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        x_mu0 = x - self.mu_0
        x_mu1 = x - self.mu_1

        sigma_i = np.linalg.inv(self.sigma)
        z0 = np.sum(np.dot(np.dot(x_mu0, sigma_i), x_mu0.T), axis=1)
        z1 = np.sum(np.dot(np.dot(x_mu1, sigma_i), x_mu1.T), axis=1)

        p_y0 = (1.0 - self.phi) * np.exp(-0.5 * z0)
        p_y1 = self.phi * np.exp(-0.5 * z1)

        return (p_y1 > p_y0).astype(int)
