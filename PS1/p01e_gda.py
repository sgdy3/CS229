import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, pred_path,*vaild_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(pred_path, add_intercept=False)
    model=GDA()
    theta_T,theta_0=model.fit(x_train,y_train)
    model.theta=[theta_T,theta_0]
    model.predict(x_test)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        x1=x[y==1]
        x0=x[y==0]
        phi=np.sum(y==1)/y.shape[0]
        mu0=np.sum(x0,axis=0)/x0.shape[0]
        mu1=np.sum(x1,axis=0)/x1.shape[0]
        sigma=(np.dot((x0-mu0).T,(x0-mu0))+np.dot((x1-mu1).T,(x1-mu1)))/x.shape[0]
        sigma_i=np.linalg.inv(sigma)
        theta_T=np.dot((mu1-mu0).T,sigma_i)
        theta_0=1/2*(np.dot(np.dot(mu0.T,sigma_i),mu0)-np.dot(np.dot(mu1.T,sigma_i),mu1)-np.log((1-phi)/phi))
        g=1/(1+np.exp(-np.dot(x,theta_T.T)-theta_0))
        if (self.verbose):
            error = np.mean(np.sqrt((g - y) ** 2))
            print("training set error:{}".format(error))
        return theta_T,theta_0
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        theta_T=self.theta[0]
        theta_0=self.theta[1]
        g=1/(1+np.exp(-np.dot(x,theta_T.T)-theta_0))
        # *** END CODE HERE
def plot_bound(train_path,model):
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='blue')
    plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], marker='x', color='red')
    x=np.arange(0,7,0.1)
    y = -model.theta[0][0] / model.theta[0][1] * x - model.theta[1] / model.theta[0][1]
    plt.plot(x, y, color='orange')
    plt.show()

train_path=r'E:\python_program\cs229\data\ds1_train.csv'
test_path=r'E:\python_program\cs229\data\ds1_valid.csv'
pred=main(train_path,test_path)