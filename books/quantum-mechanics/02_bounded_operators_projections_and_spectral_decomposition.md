# 第二章：有界算子、投影与谱分解

一个二能级可观测量若写成 $A=\operatorname{diag}(2,-1)$，它并不是给每个态预先指定一个确定数值。对态 $(\sqrt3/2,1/2)^T$，实验结果只能是 $2$ 或 $-1$，概率分别由两个坐标分量在相应特征子空间上的投影决定。这个简单计算同时提出三个问题：怎样在不选基的情况下表示事件，为什么测量值必须为实数，以及概率加权平均为何等于内积 $\langle\psi,A\psi\rangle$。

有界算子是回答这些问题的第一层完整模型。伴随刻画内积下的反向作用，自伴性保证有限维谱为实数，正交投影表示互斥事件，谱分解则把一个可观测量拆成“数值乘事件”的和。这里先在有限维和处处定义的有界算子范围内完成全部代数；位置、动量等无界可观测量为什么不能照搬这些公式，将在下一章由定义域与谱测度处理。

## 2.1 有界算子与伴随

**定义 2.1.** Hilbert 空间 $\mathcal H$ 上的线性算子 $A:\mathcal H\to\mathcal H$ 称为有界，若存在 $C\ge0$ 使
$$
\|A\psi\|\le C\|\psi\|
$$
对所有 $\psi\in\mathcal H$ 成立。所有有界算子组成代数 $\mathcal B(\mathcal H)$。

**定义 2.2.** 有界算子 $A$ 的伴随 $A^*$ 是唯一满足
$$
\langle A^*\psi,\phi\rangle=\langle\psi,A\phi\rangle
$$
的有界算子。若 $A=A^*$，称 $A$ 为自伴有界算子。

**命题 2.3.** 若 $A,B\in\mathcal B(\mathcal H)$，则 $(AB)^*=B^*A^*$。

**证明.** 对任意 $\psi,\phi$，
$$
\langle (AB)^*\psi,\phi\rangle=\langle\psi,AB\phi\rangle
=\langle A^*\psi,B\phi\rangle
=\langle B^*A^*\psi,\phi\rangle.
$$
由内积非退化性得 $(AB)^*=B^*A^*$。$\square$

伴随允许我们表达“取出某个子空间分量”时不破坏内积结构。满足幂等性与自伴性的算子恰好给出这种正交分解，因此成为测量事件的算子模型。

## 2.2 投影

**定义 2.4.** 算子 $P\in\mathcal B(\mathcal H)$ 称为正交投影，若
$$
P^2=P,\qquad P^*=P.
$$

**命题 2.5.** 若 $P$ 是正交投影，则 $\operatorname{Ran}P$ 与 $\ker P$ 正交，且
$$
\mathcal H=\operatorname{Ran}P\oplus\ker P.
$$

**证明.** 对 $x=Pu\in\operatorname{Ran}P$ 和 $y\in\ker P$，
$$
\langle x,y\rangle=\langle Pu,y\rangle=\langle u,Py\rangle=0.
$$
任意 $\psi$ 可写为 $\psi=P\psi+(I-P)\psi$，第一项在 $\operatorname{Ran}P$ 中，第二项由 $P(I-P)=0$ 在 $\ker P$ 中。若交中有 $z$，则 $z=Pz$ 且 $Pz=0$，故 $z=0$。$\square$

单个投影只区分“落在子空间内”与“落在其正交补中”。一个具有多个可能数值的可观测量，需要一族两两正交且和为恒等的谱投影。

## 2.3 有限维谱分解

**定理 2.6（有限维谱定理）.** 设 $A$ 是有限维 Hilbert 空间上的自伴算子。则存在互异实数 $\lambda_1,\dots,\lambda_m$ 和两两正交的正交投影 $P_1,\dots,P_m$，使
$$
A=\sum_{r=1}^m\lambda_rP_r,\qquad \sum_{r=1}^mP_r=I.
$$

**证明.** 复矩阵的 Schur 分解给出正交归一基，使 $A$ 为上三角矩阵。因 $A=A^*$，该上三角矩阵同时等于其共轭转置，故非对角元为零且对角元为实数。按相同特征值分组，令 $P_r$ 为对应特征子空间的正交投影，即得分解。$\square$

**定义 2.7.** 在纯态 $\psi$ 中，自伴有界算子 $A$ 的期望值定义为
$$
\langle A\rangle_\psi=\langle\psi,A\psi\rangle.
$$

**命题 2.8.** 若 $A=\sum_r\lambda_rP_r$，则在态 $\psi$ 中测得 $\lambda_r$ 的概率为
$$
\|P_r\psi\|^2,
$$
且期望值为 $\sum_r\lambda_r\|P_r\psi\|^2$。

**证明.** 由投影正交性，
$$
\langle\psi,A\psi\rangle=\sum_r\lambda_r\langle\psi,P_r\psi\rangle
=\sum_r\lambda_r\|P_r\psi\|^2.
$$
因为 $\sum_rP_r=I$，概率和为 $\sum_r\|P_r\psi\|^2=\|\psi\|^2=1$。$\square$

**例子 2.8A（两结果可观测量）.** 在 $\mathbb C^2$ 中令
$$
A=\begin{pmatrix}2&0\\0&-1\end{pmatrix},\qquad
\psi=\begin{pmatrix}\sqrt3/2\\1/2\end{pmatrix}.
$$
谱投影为 $P_2=\operatorname{diag}(1,0)$ 与
$P_{-1}=\operatorname{diag}(0,1)$，故
$$
\Pr(A=2)=\|P_2\psi\|^2=\frac34,\qquad
\Pr(A=-1)=\|P_{-1}\psi\|^2=\frac14.
$$
由谱概率计算的平均值为
$$
2\cdot\frac34-1\cdot\frac14=\frac54,
$$
直接矩阵计算也给出 $\langle\psi,A\psi\rangle=5/4$。

谱投影是取值只有 $0,1$ 的特殊正算子。实际测量还会出现“部分响应”的事件，因此先把正性与 $0\le E\le I$ 的 effect 概念从投影中抽离出来。

## 2.4 正算子与 effect

**定义 2.9.** 有界自伴算子 $A$ 称为正算子，记作 $A\ge0$，若
$$
\langle\psi,A\psi\rangle\ge0
$$
对所有 $\psi\in\mathcal H$ 成立。若 $0\le E\le I$，称 $E$ 为 effect。

**命题 2.10.** 正交投影 $P$ 是 effect。

**证明.** 对任意 $\psi$，
$$
\langle\psi,P\psi\rangle=\langle P\psi,P\psi\rangle=\|P\psi\|^2\ge0,
$$
故 $P\ge0$。同理
$$
\langle\psi,(I-P)\psi\rangle=\|(I-P)\psi\|^2\ge0,
$$
故 $P\le I$。$\square$

**说明 2.11.** 投影测量只使用特殊 effect，即满足 $P^2=P$ 的 effect。第十八章的 POVM 允许一般 effect，因此能描述非理想测量和带辅助系统的测量。

有限维谱分解已经把可观测量、事件、概率与期望值连成一条可计算链：$A=\sum_r\lambda_rP_r$ 给出结果 $\lambda_r$，而 $\|P_r\psi\|^2$ 给出其概率。下一章保留这条链的含义，却把有限求和改为谱积分；真正的新困难不是积分符号本身，而是无界算子只能作用在 Hilbert 空间的一个稠密定义域上。

## 练习

**练习 2.1.** 证明若 $P$ 是正交投影，则 $I-P$ 也是正交投影。

**练习 2.2.** 对 Pauli 矩阵 $\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ 写出谱投影。
