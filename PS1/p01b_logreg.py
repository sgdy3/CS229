import numpy as np
import util
import  matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, pred_path,*vaild_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test,y_test=util.load_dataset(pred_path,add_intercept=True)
    model=LogisticRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    return y_pred
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        theta = self.theta if self.theta else np.zeros(x.shape[1])
        for i in range(self.max_iter):
            theat_x=np.dot(x,theta)
            g=1/(1+np.exp(-theat_x))
            if(self.verbose):
                error=np.mean(np.sqrt((g-y)**2))
                print("{}th iteration error:{}".format(i,error))
            diag=1/x.shape[0]*np.diag(g*(1-g))
            Hessian=np.dot(np.dot(x.T,diag),x)
            derivation=np.dot(1/x.shape[0]*x.T,(g-y))
            theta_0=theta
            theta=theta-self.step_size*np.dot(np.linalg.inv(Hessian),derivation)
            if(np.sum(abs(theta-theta_0))<self.eps):
                self.theta = theta
                break
        return self


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        theat_x = np.dot(x, self.theta)
        y_pred=1/(1+np.exp(-theat_x))
        return y_pred
        # *** END CODE HERE ***

def plot_bound(train_path,model):
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    plt.scatter(x_train[y_train == 1, 1], x_train[y_train == 1, 2], color='blue')
    plt.scatter(x_train[y_train == 0, 1], x_train[y_train == 0, 2], marker='x', color='red')
    x=np.arange(0,7,0.1)
    y=-model.theta[1]/model.theta[2]*x-model.theta[0]/model.theta[2]
    plt.plot(x, y, color='orange')
    plt.show()


train_path=r'E:\python_program\cs229\data\ds1_train.csv'
test_path=r'E:\python_program\cs229\data\ds1_valid.csv'
pred=main(train_path,test_path)