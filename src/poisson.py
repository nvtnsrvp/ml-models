import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    fn, ext = pred_path.split('.')
    plot_path = '.'.join([fn, 'png'])

    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    pr = PoissonRegression(max_iter=3000, step_size=lr)
    pr.fit(x_train, y_train)
    y_pred = pr.predict(x_eval)


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        def error(e):
            return np.sum(np.abs(e))

        m, n = x.shape
        self.theta = np.zeros(n)
        e = np.ones(n)
        i = 0
        while i < self.max_iter and error(e) > self.eps:
            grad = 1.0/m * np.dot(y - np.exp(x.dot(self.theta)), x)
            e = self.step_size * grad
            self.theta += e
            i += 1

        if self.verbose:
            print(i, self.theta)

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return np.exp(x.dot(self.theta))
