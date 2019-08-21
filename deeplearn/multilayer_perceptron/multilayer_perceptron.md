<!--
 * @Descripttion: 
 * @version: 
 * @Author: SunZewen
 * @Date: 2019-08-01 17:20:22
 * @LastEditors: SunZewen
 * @LastEditTime: 2019-08-10 12:55:33
 -->
# 多层感知机 （multilayer perception，MLP）


## 一、 隐藏层

## 二、神经网络模型
示例为一个三层网络模型，包含两层隐藏层。

横向流程图源码格式：



### 2.1 前向传播

第一层
$$
h_{11} = w_{111}*x_{1} + w_{112} * x_{2} + b_{1} \\
h_{12} = w_{121}*x_{1} + w_{122} * x_{2} + b_{1} \\
h_{13} = w_{131}*x_{1} + w_{132} * x_{2} + b_{1}  \\

ho_{11} = relu(h_{11}) \\
ho_{12} = relu(h_{12}) \\
ho_{13} = relu(h_{13})
$$

第二层
$$
h_{21} = w_{211}*ho_{11} + w_{212} * ho_{12} + w_{213} * ho_{13} + b_{2} \\
h_{22} = w_{221}*ho_{11} + w_{222} * ho_{12} + w_{223} * ho_{13} + b_{2} \\
h_{23} = w_{231}*ho_{11} + w_{232} * ho_{12} + w_{233} * ho_{13} + b_{2} \\

ho_{21} = relu(h_{21}) \\
ho_{22} = relu(h_{22}) \\
ho_{23} = relu(h_{23})
$$

第三层
$$
o_{1} = h_{31} = w_{311}*ho_{21} + w_{312} * ho_{22} + w_{313} * ho_{23} \\
o_{2} = h_{32} = w_{321}*ho_{21} + w_{322} * ho_{22} + w_{323} * ho_{23}
$$

$$
a_{1} = softmax(o_1) \\
a_{2} = softmax(o_2)
$$

误差计算 

$$
E = H(p, q) = - \sum_{x}
            p(x) \;
            \rm {log}\;
            q(x)
$$


### 2.2 反向传播

第3层 

根据单层soft max推导结果可得


$$
\frac{\partial E}{\partial w_{311}} = \sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1} * \frac{\partial o_1}{\partial w_{311}} \\
\qquad \qquad \qquad \qquad \qquad \quad \ = \frac{\partial E}{\partial a_1} * \frac{\partial a_1}{\partial o_1} * \frac{\partial o_1}{\partial w_{311}}  + \frac{\partial E}{\partial a_2} * \frac{\partial a_2}{\partial o_1} * \frac{\partial o_1}{\partial w_{311}}  \\
\quad = (a_1 - y_1) * ho_{21} \quad \\
$$

$$
\frac{\partial E}{\partial b_{3}} = \sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1} * \frac{\partial o_1}{\partial b_{3}} \\
\qquad \qquad \qquad \qquad \qquad \quad \ = \frac{\partial E}{\partial a_1} * \frac{\partial a_1}{\partial o_1} * \frac{\partial o_1}{\partial b_{3}}  + \frac{\partial E}{\partial a_2} * \frac{\partial a_2}{\partial o_1} * \frac{\partial o_1}{\partial b_{3}}  \\
\quad = a_1 - y_1 \quad \\
$$

第2层
$$
\frac{\partial E}{\partial w_{211}} = (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1} )* \frac{\partial o_1}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {w_{211}}}  + (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_2} )* \frac{\partial o_2}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {w_{211}}} \\
\qquad \qquad = (a_1 - y_1) * w_{311} * relugd(h_{21}) * ho_{11} + (a_2 - y_2) * w_{321} * relugd(h_{21}) * ho_{11} \\
\qquad \qquad = ((a_1 - y_1) * w_{311}  + (a_2 - y_2) * w_{321}) * relugd(h_{21}) * ho_{11}
$$

$$
\frac{\partial E}{\partial b_{2}} = (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1} )* \frac{\partial o_1}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {b_{2}}} + (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_2} )* \frac{\partial o_2}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {b_{2}}}\\
\qquad \qquad = ((a_1 - y_1) * w_{311}  + (a_2 - y_2) * w_{321}) * relugd(h_{21})
$$

第1层
$$
\frac{\partial E}{\partial w_{111}} = (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1}) * \frac{\partial o_1}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {ho_{11}}} * \frac {\partial ho_{11}} {\partial {h_{11}}} * \frac {\partial h_{11}} {\partial {w_{111}}}\\
\qquad \qquad = (a_1 - y_1) * w_{311} * relugd(h_{21}) * w_{211} * relugd(h_{11}) * x_{1}
$$

$$
\frac{\partial E}{\partial b_{1}} = (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1}) * \frac{\partial o_1}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {ho_{11}}} * \frac {\partial ho_{11}} {\partial {h_{11}}} * \frac {\partial h_{11}} {\partial {b_{1}}}\\
\qquad \qquad = (a_1 - y_1) * w_{311} * relugd(h_{21}) * w_{211} * relugd(h_{11})
$$

通用求导公式
$$
\frac{\partial E}{\partial w_{kij}} = (\sum^{2}_{i = 1} \frac{\partial E}{\partial a_i} * \frac{\partial a_i}{\partial o_1}) * \frac{\partial o_1}{\partial ho_{21}} * \frac {\partial ho_{21}} {\partial {h_{21}}}  * \frac {\partial h_{21}} {\partial {ho_{11}}} * \frac {\partial ho_{11}} {\partial {h_{11}}} * \frac {\partial h_{11}} {\partial {w_{111}}}\\
\qquad \qquad = (a_1 - y_1) * w_{311} * relugd(h_{21}) * w_{211} * relugd(h_{11}) * x_{1}
$$
反向传播计算过程



## 三、激活函数

### 3.1 ReLU函数

### 3.2 sigmoid函数

### 3.3 tanh函数

## 四、梯度计算

1、