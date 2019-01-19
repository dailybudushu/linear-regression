# linear regression


## 摘要

    使用了winequality-red.csv数据集，通过岭回归，获得最优模型。分别进行了手写编程和运用scikit-learn包

## 关键字

    winequality-red.csv数据集，岭回归，scikit-learn，手写实现

## 引言

    该文介绍了线性回归中岭回归的实现，并在手写编程中分析讨论了初始w和b的值的影响

## 相关工作

- 从csv中读取相应数据
- 理解线性回归中的岭回归
- 学习使用scikit-learn包

## 本文工作

### 一.分配训练集和测试集

红酒数据集中共包括12列1600行数据，其中包含一个表头，因此共1599行可用数据，我默认将160个数据作为测试集，而1439个数据作为训练集。

### 二.确定模型

#### 模型

每个数据由11个特征以及一个quality。因此我将模型定为

<img src="https://latex.codecogs.com/gif.latex?$$&space;\hat{y}=x_{n*11}\times{W_{11*1}}&plus;b&space;$$" title="$$ \hat{y}=x_{n*11}\times{W_{11*1}}+b $$" />

其中n为样本个数

#### 损失函数

这里用了最小二乘法,损失函数为

<img src="https://latex.codecogs.com/gif.latex?$$&space;J(\theta)=\frac{1}{2\times{n}}\sum_{i=1}^{n}(h(x^i)-y^i)^2&space;$$" title="$$ J(\theta)=\frac{1}{2\times{n}}\sum_{i=1}^{n}(h(x^i)-y^i)^2 $$" />

可以在后面加上正则惩罚项L2范式

<img src="https://latex.codecogs.com/gif.latex?$$&space;J(\theta)=\frac{1}{2\times{n}}\sum_{i=1}^{n}(h(x^i)-y^i)^2&plus;\frac{\lambda}{2n}\sum_n|w|^2&space;$$" title="$$ J(\theta)=\frac{1}{2\times{n}}\sum_{i=1}^{n}(h(x^i)-y^i)^2+\frac{\lambda}{2n}\sum_n|w|^2 $$" />

#### 三.小批量梯度下降

<img src="https://latex.codecogs.com/gif.latex?$$&space;X=\left[&space;\begin{matrix}&space;x_1^1&space;&&space;x_2^1&space;&&space;...&space;&&space;x_m^1&space;\\&space;x_1^2&space;&&space;x_2^2&space;&&space;...&space;&&space;x_m^2\\&space;|&space;&&space;|&space;&&space;...&&space;|\\&space;x_1^n&space;&&space;x_2^n&space;&&space;...&&space;x_m^n&space;\end{matrix}&space;\right]&space;$$" title="$$ X=\left[ \begin{matrix} x_1^1 & x_2^1 & ... & x_m^1 \\ x_1^2 & x_2^2 & ... & x_m^2\\ | & | & ...& |\\ x_1^n & x_2^n & ...& x_m^n \end{matrix} \right] $$" />

<img src="https://latex.codecogs.com/gif.latex?$$&space;W=&space;\left[&space;\begin{matrix}&space;w_1&space;\\&space;w_2&space;\\&space;|&space;\\&space;w_m&space;\end{matrix}&space;\right]&space;$$" title="$$ W= \left[ \begin{matrix} w_1 \\ w_2 \\ | \\ w_m \end{matrix} \right] $$" />

<img src="https://latex.codecogs.com/gif.latex?$$&space;y=&space;\left[&space;\begin{matrix}&space;y^1\\&space;y^2\\&space;|&space;\\&space;y^n&space;\end{matrix}&space;\right]&space;$$" title="$$ y= \left[ \begin{matrix} y^1\\ y^2\\ | \\ y^n \end{matrix} \right] $$" />

<img src="https://latex.codecogs.com/gif.latex?$$&space;E=&space;\left[&space;\begin{matrix}&space;h(x^1)-y^1&space;\\&space;h(x^2)-y^2\\&space;|&space;\\&space;h(x^n)-y^n&space;\end{matrix}&space;\right]&space;$$" title="$$ E= \left[ \begin{matrix} h(x^1)-y^1 \\ h(x^2)-y^2\\ | \\ h(x^n)-y^n \end{matrix} \right] $$" />

其中m为特征个数，n为样本个数

<img src="https://latex.codecogs.com/gif.latex?$$&space;\frac{\partial{J(\theta)}}{\partial{w}}=\frac{1}{M}\times{X^TE}&space;$$" title="$$ \frac{\partial{J(\theta)}}{\partial{w}}=\frac{1}{M}\times{X^TE} $$" />

<img src="https://latex.codecogs.com/gif.latex?$$&space;w=w-step\times\frac{1}{n}\times{X^TE}&space;$$" title="$$ w=w-step\times\frac{1}{n}\times{X^TE} $$" />

<img src="https://latex.codecogs.com/gif.latex?$$&space;b=b-step\times{\frac{1}{n}}\sum_n(h(w^i)-y^i)&space;$$" title="$$ b=b-step\times{\frac{1}{n}}\sum_n(h(w^i)-y^i) $$" />

### 四.scikit-learn实现

这个就是调包然后利用其中的linearRegression()实现，看了一下源码。。但是还没看太懂。预测模型的误差rmse为0.6

## 实验结果分析

### 数据集

    winequality-red.csv

### 编程语言

    python

### 工具包

    numpy,scikit-learn

### w初始化值对结果的影响

#### 不同w初始值下的梯度下降图和rmse

开始时我初始化w是利用np.random.rand(11, 1),这样造成的梯度下降图为
![image1](梯度下降1.png)

看起来十分满意，但是实际上最终的rmse较大,为3左右

后来将np.random.rand(11, 1)改为np.random.rand(11, 1)*0.1，并加入正则惩罚项图片则变为：

![image2](梯度下降2.png)

看起来效果不好但是实际上，修改后的rmse为1左右，变为了原来的1/3

#### 分析

在多次调试后，我发现主要原因是在w调整后，step同时要跟着一起更改，在最初的w里面，可以看到最初的损失函数在700左右，这也就意味着这要用更大的step，只有这样才能正常下降到底部否则将无法降到底端，而更大的step也代表这当其到达底端时，会导致来回变化更大，换句话说就是误差更大。而在初始的w乘0.1时，导致开始时损失函数较小，在40左右，这也就使得我们可以调整更小的w，使其正常下降到底端时，变化更小，也就是使误差更小。

#### plus

以上内容，纯属猜测

## 总结

- 数学很重要
- 要多看源码

## 参考文献

- [机器学习-----线性回归浅谈(Linear Regression)](https://www.cnblogs.com/GuoJiaSheng/p/3928160.html)(有很多错误但是思路还行)
