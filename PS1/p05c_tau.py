import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_vaild,y_valid=util.load_dataset(valid_path, add_intercept=True)
    model=LocallyWeightedLinearRegression(tau)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_vaild)
    mse = np.mean((y_pred - y_valid)**2)
    print('tau={},MSE={}'.format(tau,mse))

    plt.figure()
    plt.title('tau={}'.format(tau))
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_vaild, y_pred, 'ro', linewidth=2)
    plt.plot(x_vaild, y_valid, 'go', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # *** END CODE HERE ***

train_path = r'E:\python_program\cs229\data\ds5_train.csv'
valid_path = r'E:\python_program\cs229\data\ds5_valid.csv'
for tau in [0.1,0.2,0.5,0.7,0.9,1]:
    main(tau, train_path, valid_path,'a','b')