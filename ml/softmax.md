# <center> 基于Numpy实现softmax </center>

## 一、 softmax基本概念
    在多分类问题上会遇到两个问题
    1. 输出层的输出范围不确定，难以直观上判断这些值得意义；
    2. 真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。
 1. softmax通过下式将输出值变换为正值且概率的和为1的概率分布：
$$
    \ {y_1},\ \hat{y_2},\ \hat{y_3} = softmax(o_{1},\  o_{2},\ o_{3})
$$
其中
$$
    \hat{y_1} = \frac{exp(o_1)}{\sum^{3}_{i=1}{exp(o_i)}}, \quad
    \hat{y_2} = \frac{exp(o_2)}{\sum^{3}_{i=1}{exp(o_i)}},\quad
    \hat{y_3} = \frac{exp(o_3)}{\sum^{3}_{i=1}{exp(o_i)}}
$$

## 一、模型初始化
### 1、 W, b初始化
    W为权重矩阵，是一个二维矩阵，是由输出参数与最终的分类个数决定的，大小为（输入参数，分类类型）
    b为偏差，是一个向量，长度为分类类型个数
## 二、训练

### 1、单样本分类的⽮量计算表达式
    为了方便计算一般将W进行转置，（分类类型，输入参数），将b转置为（1，分类类型）
    O = X * Wt + tbt

权重与偏差
$$
W = \begin{bmatrix}
    w_{11} & w_{12} & \dots &w_{1n} \\
    w_{21} & w_{22} & \dots &w_{2n} \\
    \vdots & \vdots & \ddots &\vdots \\
    w_{m1} & w_{m2} & \dots &w_{mn} \\
    \end{bmatrix}

,\quad

b = \begin{bmatrix}
    b_1 &
    b_2 &
    \dots &
    b_{n}
    \end{bmatrix}
$$
<br/>

输入层，样本特征
$$
x^{(i)} = \begin{bmatrix}
        x^{i}_{1} \quad
        x^{i}_{2} \quad
        \dots \quad
        x^{i}_{m}
        \end{bmatrix}

$$
<br/>

输出层
$$
o^{(i)} = \begin{bmatrix}
        o^{i}_{1} \quad
        o^{i}_{2} \quad
        \dots \quad
        o^{i}_{m}
        \end{bmatrix}

$$
<br/>

概率分布
$$
y^{(i)} = \begin{bmatrix}
        y^{i}_{1} \quad
        y^{i}_{2} \quad
        \dots \quad
        y^{i}_{m}
        \end{bmatrix}

$$
<br/>


softmax回归对样本i分类的⽮量计算表达式为
$$
    o^{(i)} = x^{i}W + b, \\
    \hat{y}^{(i)} = softmax(o^{(i)}).
$$
<br/>


计算过程
$$
    \begin{bmatrix}
    o_1 \\
    o_2 \\
    \vdots \\
    o_{n}
    \end{bmatrix}
    =
    \begin{bmatrix}
    x_1 & x_2 & \dots &x_{m}
    \end{bmatrix}
    *
    \begin{bmatrix}
    w_{11} & w_{12} & \dots &w_{1n} \\
    w_{21} & w_{22} & \dots &w_{2n} \\
    \vdots & \vdots & \ddots &\vdots \\
    w_{m1} & w_{m2} & \dots &w_{mn} \\
    \end{bmatrix}
    +
    \begin{bmatrix}
    b_1 &
    b_2 &
    \dots &
    b_{n}
    \end{bmatrix} \tag{1}
$$

<br/>

$$
    o_1 = x_1w_{11} + x_2w_{21} + x_3w_{31} + \dots + x_nw_{n1} + b_1\\
    o_2 = x_1w_{12} + x_2w_{22} + x_3w_{32} + \dots + x_nw_{n2}  + b_2\\
    \vdots \\
    o_m = x_1w_{1m} + x_2w_{2m} + x_3w_{3m} + \dots + x_nw_{nm} + b_m\tag{2}
$$

### 2. 误差公式
 
交叉熵函数
$$
    H(y^{(i)}, \hat y^{(i)}) = -\sum^{q}_{j=1} y^{(i)}_j log \hat y^{(i)}_j \tag{3}
$$

交叉熵损失函数
$$
    \ell(\Theta)  = \frac{1}{n} \sum ^n _{i=1} H(y^{(i)}, \hat y^{(i)}) \tag{4}
$$

### 4. 反向传播

    最小化误差 ->  利用梯度下降法求出误差最小时 w 与 b 的值。
    
    求梯度
$$
    \frac{\partial y}{\partial o} * \frac{\partial o}{\partial w} 
    ---> \frac{\partial y}{\partial o} \\
    ---> \frac{\partial o}{\partial w} \\
$$     
$$   
    \frac{\partial y}{\partial o} * \frac{\partial o}{\partial b} 
    ---> \frac{\partial y}{\partial o} \\
    ---> \frac{\partial o}{\partial b} \\
$$
## 三、预测


