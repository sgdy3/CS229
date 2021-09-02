import numpy as np
import util
import matplotlib.pyplot as plt
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
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    model=PoissonRegression(step_size=1e-7)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_valid)
    plt.figure()
    plt.plot(y_valid, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        theta=self.theta if self.theta else np.zeros(x.shape[1])
        i=0
        while True:
            h=np.exp(np.dot(x,theta))
            gradient=np.dot(y-h,x)/x.shape[0]
            theta0=theta
            theta=theta+self.step_size*gradient
            if(self.verbose):
                y_pred=np.exp(np.dot(x,theta))
                error=np.mean(np.sqrt((y_pred-y)**2))
                print('{}th iteraion error:{}'.format(i,error))
            i=i+1
            if(np.linalg.norm(theta-theta0,ord=1)<self.eps):
                self.theta=theta
                break
        return self
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        theta=self.theta
        h = np.exp(np.dot(x, theta))
        return h
        # *** END CODE HERE ***

train_path=r'E:\python_program\cs229\data\ds4_train.csv'
valid_path=r'E:\python_program\cs229\data\ds4_valid.csv'
main(0.01,train_path,valid_path,'b')