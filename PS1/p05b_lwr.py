import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_vaild,y_valid=util.load_dataset(eval_path, add_intercept=True)
    model=LocallyWeightedLinearRegression(tau)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_vaild)
    mse = np.mean((y_pred - y_valid)**2)
    print('MSE={}'.format(mse))

    plt.figure()
    plt.title('tau={}'.format(tau))
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_vaild, y_pred, 'ro', linewidth=2)
    plt.plot(x_vaild, y_valid, 'go', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x=x
        self.y=y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,). eval_path
        """
        # *** START CODE HERE ***
        y_pred=[]
        for i in range(x.shape[0]):
            W=np.exp(-np.sum((self.x-x[i])**2,axis=1)/2/self.tau**2)
            W=np.diag(W)
            theta=np.dot(np.linalg.inv(np.dot(np.dot(self.x.T,W),self.x)),np.dot(np.dot(self.x.T,W),self.y))
            y_pred.append(np.dot(theta.T,x[i]))
        y_pred=np.array(y_pred)
        return y_pred
        # *** END CODE HERE ***
train_path=r'E:\python_program\cs229\data\ds5_train.csv'
valid_path=r'E:\python_program\cs229\data\ds5_valid.csv'
main(0.5,train_path,valid_path)