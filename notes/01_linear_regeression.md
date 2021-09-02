Linear Regression model
=======================

>Info:  Linear Regression model is used for solving regression problem. Main idea is use a polynomial of input features($x_j$) to fit the relationship between X and Y. 

# 1. solutions 
In geneal, LR model can be expressed like:

<center> 

$\mathop{\arg\min}\limits_{\theta}:J(\theta)=||X\theta-Y||_2^2$ 

$X\in R^{m*n},Y\in R^{m},\theta \in R^{n}$
</center>
Based on this primal problem, we can figure out different soluions below.

## i.MSE Alg
Minimize Squared error (MSE) is the word description of formular above. So the main idea of MSE Alg is obvious, finding the **gradient** of loss funtion (squared error func),then gradually descending to the global minimum.

    This thought contains two strong supports.
    a). gradient represent the fastest rising dirction of loss funcion.
    b). once we found the local minimum, it must be the global minmum, cause the loss funcion is conve.

To caculate the gradient funciont, we definitely need to compute derivative with respect to input samples. However, we also have two different ways to do the caculation.
### a). batch gradient descent
looks at every example in the traing-set on every step;
### b). stochastic gradient discnet
respect to single training example in traing-set only.May cause parameters oscillation, but much faster.

After we get the gradient, we need to update the parameters:
<center>

$\theta_j =\theta_j-\alpha\frac{\partial J(\theta)}{\partial \theta_j}$

$\alpha ~is~the ~learing ~rate$
</center>

## ii. Norm equations
Without gradually approxmating global minimum, soving norm euation can help us directly find the minimum.There are two interpretation for this method.
### a). algebra interpretation
<center>

$J(\theta)=\frac{1}{2}(X\theta-Y)^T(X\theta-Y)$

</center>

To find the golbal minimum of $J(\theta)$, we just need to caulute the deriavative with respect to $\theta$, then let it be 0 to get the optimal paramters we need. So
<center>

$\nabla_\theta J(\theta) =X^TX\theta -X^TY$
$\theta=(X^TX)^{-1}X^TY$
</center>

### b). geometrical interpretion 
we caculate $X\theta=\hat{Y}, while~~X\in R^{m*n},\hat{Y}\in R^{m},\theta \in R^{n}$.<br>
This a projection between input space $\theta$ and output space $\hat{Y}$ via transforming funcion $X$. Cause we always have $m>n$, only row space in space $\theta$ and column space in $Y$ can have bijection. By doing prediction, out goal is to achieve $\hat{Y}=Y$, but because of the transforming matrix is not full rank, $Y$ may out of the column space, we can't achieve $\hat{Y}=Y$. We can only minimize the distance between the $\hat{Y}$ which is a coordinate in column space, and the $Y$ which is the real value out of column space. Obviously, when $\hat{Y}$ is the projection of $Y$ in column space, the two coordinates has the minimum distance.
<center>

$proj(Y,X)=X(X^TX)^{-1}X^TY$<br>
$X\theta=X(X^TX)^{-1}X^TY$<br>
$\theta=(X^TX)^{-1}X^TY$
</center>

## iii. Locally weighted regeression
>Linear regression model need the raw data lies in a straight line. Otherwise, the fiting effect won't be satisfying. One way to solve this problem is adding the extra higher order features like $x^2,x^3...$ But this solution may cause overfitting, so we create a new method called Locally weighted linear regression to make the choice of features less critical.

In linear regression model:
<center>

$\mathop{\arg\min}\limits_{\theta}:J(\theta)=\Sigma_{i} (y^i-\theta^Tx^i)$ 
</center>
In LWR:
<center>

$\mathop{\arg\min}\limits_{\theta}:J(\theta)=\sum_i \omega_i(y^i-\theta^Tx^i)$<br>
$\omega_i=exp(-\frac{(x^i-x)^2}{(2\tau^2)})$ 
</center>

Intruducing $\omega^i$ means the model is constructed based on particular x at which we're trying to evaluate. Closer $x^i$ will have higher weight in the model.<br>
LWR is a non-parametric model, which means the model doesn't have fixed parameters, but depending on the traning-set and the predicting sample. In contrast, linear regression model is a parameter model.After we complete the traning stage, we can get fixed parameters which can be used in any predicting samples.

# 2. probalistic intepretation 
 >Although the squared error funcion seems to be a very intuitively resonable measurement of prediction. We actually are caculating the Euclidean distance between $\hat{Y}$ and $Y$. But why should we choose qudratic function? We can find a explanation from the view of probablity. 

 To elicit the conclusion, we need to give 2 assumptions.
 ### a). 
 <center>

$y^i=\theta^Tx^i+\epsilon^i, \epsilon^i\sim N(0,\sigma^2)$<br>
</center>

$\epsilon^i$ is the error that we can't capture from model, like random noise.<br>
So the assumption actually make sense. It figuers out the relationship between groud-truth and prediction. It also explicitly indicates our captured $Y$ is sampled from a Gaussian distribution, instead of a constant value.
### b).
<center>

$\epsilon^i~is~IID(independently~indentically ~distributed)$
</center>
This assumpution is also resonable. Every sample is IID.

After giving the 2 assumption, we can imply that:
<center>

$p(y^i|x^i;\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^i-\theta^Tx^i)^2}{2\sigma^2})$
</center>
As we already observed x^i, unknown \theta. Intiuively,we'd like to maxmize the probablity of what alread happened. This maxmization is called ML (maximize likelyhood function).
<center>

$\mathop{\arg\max}\limits_{\theta}:L(\theta)=L(\theta;x,y)=p(y|x;\theta)=\prod\limits_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^i-\theta^Tx^i)^2}{2\sigma^2})$ 
</center>

To simplify the caculation, we'll take the logarithm on equation.
<center>

$\mathop{\arg\max}\limits_{\theta}:logL(\theta)=mlog\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}\sum\limits_{i=1}^m(y^i-\theta^Tx^i)^2$ <br>
$\Rightarrow\mathop{\arg\min}\limits_{\theta}:\frac{1}{2}\sum_{i=1}{m}(y^i-\theta^Tx^i)^2$
</center>
Till now, we can ensure that qudractic funcion is a resonable measurement of distance between predtion and true value.

