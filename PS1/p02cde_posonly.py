import numpy as np
import util
import  matplotlib.pyplot as plt
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, **pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    if pred_path:
        pred_path_c = pred_path.replace(WILDCARD, 'c')
        pred_path_d = pred_path.replace(WILDCARD, 'd')
        pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid,y_valid=util.load_dataset(valid_path,add_intercept=True)
    x_test,y_test=util.load_dataset(test_path,add_intercept=True)
    _,t_train=util.load_dataset(train_path,'t')
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model=LogisticRegression()
    model=model.fit(x_train,t_train)
    t_pred=model.predict(x_test)
    t_pred=np.array([0 if i<=0.5 else 1 for i in t_pred])
    util.plot(x_test,t_pred,model.theta)
    plt.show()
    # Part (d): Train on y-labels and test on true labels
    model=LogisticRegression()
    model=model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    y_pred=np.array([0 if i<=0.5 else 1 for i in y_pred])
    util.plot(x_test,y_pred,model.theta)
    plt.show()
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    model=LogisticRegression()
    model=model.fit(x_train,y_train)
    x_valid=x_valid[y_valid==1]
    y_valid_pred=model.predict(x_valid)
    alpha=np.mean(y_valid_pred)
    y_pred = model.predict(x_test)
    y_pred/=alpha
    y_pred =np.array([0 if i<=0.5 else 1 for i in y_pred])
    model.theta[0]+=np.log(2/alpha-1) #with differing alpha,the boundary's interception also need adjust
    util.plot(x_test,y_pred,model.theta)
    plt.show()
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE


def plot_bound(x_train,y_train,model):
    plt.scatter(x_train[y_train == 1, 1], x_train[y_train == 1, 2], color='blue')
    plt.scatter(x_train[y_train == 0, 1], x_train[y_train == 0, 2], marker='x', color='red')
    x=np.arange(0,7,0.1)
    y=-model.theta[1]/model.theta[2]*x-model.theta[0]/model.theta[2]
    plt.plot(x, y, color='orange')
    margin1 = (max(x_train[:, -2]) - min(x_train[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    plt.show()

train_path=r'E:\python_program\cs229\data\ds3_train.csv'
valid_path=r'E:\python_program\cs229\data\ds3_valid.csv'
test_path=r'E:\python_program\cs229\data\ds3_test.csv'
main(train_path,valid_path,test_path)

