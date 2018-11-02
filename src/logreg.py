import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    fn, ext = pred_path.split('.')
    plot_path = '.'.join([fn, 'png'])

    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_eval)

    util.plot(x_eval, y_eval, lr.theta, plot_path, correction=1.0)


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        def hessian(x, g):
            m, n = x.shape
            hh = np.multiply(g, 1-g)
            return 1.0/m * sum(hh[i] * np.outer(x[i, :], x[i, :]) for i in range(m))
        def error(e):
            return np.sum(np.abs(e))

        m, n = x.shape
        self.theta = np.zeros(n, dtype=x.dtype)
        e = np.ones(n)
        i = 0
        while i < self.max_iter and error(e) > self.eps:
            g = 1.0 / (1 + np.exp(-np.dot(x, self.theta)))
            gradJ = -1.0/m * np.dot(y - g, x)
            H = hessian(x, g)
            e = np.dot(np.linalg.inv(H), gradJ)
            self.theta -= e
            i += 1

        if self.verbose:
            print(i, 'theta', self.theta)


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return np.round(1.0 / (1 + np.exp(-np.dot(x, self.theta))))
