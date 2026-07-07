# 第二十六章：角动量耦合、Clebsch-Gordan 系数与选择定则

## 本章目标

本章补齐角动量理论的教材核心：总角动量分解、两个自旋 $1/2$ 的 singlet/triplet、Clebsch-Gordan 系数的定义和选择定则的算子来源。

## 依赖前置知识

需要张量积、角动量代数、自旋 $1/2$、对称性和交换子。

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

## 本章小结

角动量耦合把张量积态空间分解为总角动量不可约部分。两个自旋 $1/2$ 给出三重态和单态，是多体自旋和交换对称性的基本模型。Clebsch-Gordan 系数控制基变换，Wigner-Eckart 定理把选择定则归结为表示论。

## 练习

**练习 26.1.** 验证命题 26.3 中四个态两两正交并归一。

**练习 26.2.** 证明 singlet 态 $|0,0\rangle$ 被总 $J_z,J_+,J_-$ 消去。

**练习 26.3.** 若偶极算子是秩 $1$ 球张量，写出由推论 26.8 得到的 $\Delta m$ 选择定则。

