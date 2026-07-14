# 第二十六章：角动量耦合、Clebsch-Gordan 系数与选择定则

两个自旋各自沿 $z$ 轴有确定分量，并不意味着复合态有确定的总
$J^2$。乘积基
$|j_1,m_1\rangle\otimes|j_2,m_2\rangle$ 适合描述局部分量，耦合基
$|j,m\rangle$ 则适合旋转不变 Hamiltonian；Clebsch--Gordan 系数正是
两组正交基之间的变换矩阵。对两个自旋 $1/2$，这次换基把四维空间分成
三维对称 triplet 与一维反对称 singlet，所有系数都能由一次降算符计算
得到。

本章先把有限维 $\mathfrak{su}(2)$ 张量积分解登记为外部输入，再在
$1/2\otimes1/2$ 情形书内构造全部四个归一态。总 $J_z$ 立即给出
$m=m_1+m_2$ 的系数选择条件。最后，Wigner--Eckart 定理把球张量矩阵元
分成几何 Clebsch--Gordan 系数与动力学约化矩阵元，从而说明选择定则只
给必要的零元条件，并不保证所有允许矩阵元都非零。

## 26.1 总角动量与共同本征基

**定义 26.1.** 给定两个角动量表示 $J^{(1)}$ 与 $J^{(2)}$，总角动量为
$$
J=J^{(1)}\otimes I+I\otimes J^{(2)}.
$$
通常取共同本征基
$$
J^2|j,m\rangle=j(j+1)|j,m\rangle,\qquad
J_z|j,m\rangle=m|j,m\rangle.
$$

**外部输入定理 26.2（有限维 $\mathfrak{su}(2)$ 表示分解，QM-EXT-12）.** 不可约表示 $j_1$ 与 $j_2$ 的张量积分解为
$$
j_1\otimes j_2\cong
\bigoplus_{j=|j_1-j_2|}^{j_1+j_2}j,
$$
其中 $j$ 每次增加 $1$。

一般分解由表示论定理保证；两个自旋 $1/2$ 的情形只有四个维度，可以从
最高权态开始逐次降阶，直接看到 $3+1$ 分解。

## 26.2 两个自旋 $1/2$

**命题 26.3.** 两个自旋 $1/2$ 的张量积分解为三重态和单态：
$$
\frac12\otimes\frac12\cong 1\oplus0.
$$
可取
$$
|1,1\rangle=|\uparrow\uparrow\rangle,
$$
$$
|1,0\rangle=\frac{|\uparrow\downarrow\rangle+|\downarrow\uparrow\rangle}{\sqrt2},
$$
$$
|1,-1\rangle=|\downarrow\downarrow\rangle,
$$
以及
$$
|0,0\rangle=\frac{|\uparrow\downarrow\rangle-|\downarrow\uparrow\rangle}{\sqrt2}.
$$

**证明.** 总 $J_z$ 的本征值由两个自旋 $z$ 分量相加得到。最高权向量 $|\uparrow\uparrow\rangle$ 满足 $m=1$，由降算符
$$
J_-=S_-^{(1)}\otimes I+I\otimes S_-^{(2)}
$$
得到
$$
J_-|\uparrow\uparrow\rangle=|\downarrow\uparrow\rangle+|\uparrow\downarrow\rangle.
$$
归一化后得到 $|1,0\rangle$，再次下降得到 $|1,-1\rangle$。$m=0$ 子空间剩余的归一正交向量为反对称组合，它被 $J_+$ 与 $J_-$ 消去，因此对应 $j=0$。$\square$

这四个态同时给出了乘积基到耦合基的第一个完整变换矩阵。一般
Clebsch--Gordan 系数就是同一换基问题的矩阵元，其最先可见的零元条件
来自总 $J_z$ 本征值必须匹配。

## 26.3 Clebsch-Gordan 系数

**定义 26.4.** Clebsch-Gordan 系数定义为基变换系数
$$
|j,m\rangle
=\sum_{m_1+m_2=m}
\langle j_1m_1,j_2m_2|jm\rangle
|j_1,m_1\rangle\otimes|j_2,m_2\rangle.
$$

**命题 26.5.** Clebsch-Gordan 系数满足选择条件 $m=m_1+m_2$。

**证明.** 张量积基向量是 $J_z$ 的本征向量：
$$
J_z(|j_1,m_1\rangle\otimes|j_2,m_2\rangle)
=(m_1+m_2)|j_1,m_1\rangle\otimes|j_2,m_2\rangle.
$$
若它在 $|j,m\rangle$ 的展开中系数非零，则必须属于同一 $J_z$ 本征值空间，因此 $m=m_1+m_2$。$\square$

基变换系数控制态空间分解。实验跃迁还包含一个算子；若该算子在旋转下
按球张量变换，其矩阵元可再次化为一个 Clebsch--Gordan 系数乘约化数据。

## 26.4 选择定则

**定义 26.6.** 设算子 $T_q^{(k)}$ 在旋转下像秩 $k$ 球张量。其矩阵元
$$
\langle j',m'|T_q^{(k)}|j,m\rangle
$$
的非零条件称为选择定则。

**外部输入定理 26.7（Wigner-Eckart 定理，QM-EXT-13）.** 球张量矩阵元分解为
$$
\langle j',m'|T_q^{(k)}|j,m\rangle
=\langle j,m;k,q|j',m'\rangle
\langle j'\|T^{(k)}\|j\rangle
$$
的 Clebsch-Gordan 系数与约化矩阵元之积。

**推论 26.8.** 若上述矩阵元非零，则
$$
m'=m+q,\qquad |j-k|\le j'\le j+k.
$$

**证明.** 这正是 Clebsch-Gordan 系数非零所需的 $m$ 加法条件和角动量三角条件。$\square$

角动量耦合把局部量子数基改写为总旋转表示的不可约通道。两个自旋
$1/2$ 的 triplet/singlet 构造给出了全部系数与交换对称性；一般
Clebsch--Gordan 系数满足磁量子数和三角条件，Wigner--Eckart 定理再把
这些条件传给球张量矩阵元。选择定则只排除必为零的跃迁，实际强度仍在
约化矩阵元中。最后一章将用三个完整模型把连续传播、自旋旋转和受驱
跃迁放在同一计算尺度上检验。

## 练习

**练习 26.1.** 验证命题 26.3 中四个态两两正交并归一。

**练习 26.2.** 证明 singlet 态 $|0,0\rangle$ 被总 $J_z,J_+,J_-$ 消去。

**练习 26.3.** 若偶极算子是秩 $1$ 球张量，写出由推论 26.8 得到的 $\Delta m$ 选择定则。
