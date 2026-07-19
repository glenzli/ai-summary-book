# 附录 A.2 感知机收敛定理

本附录证明[卷一 1.2 节](../../vol-01/ch01_early_ai_perceptron_connectionism.md#section-1-2)使用的错误次数界。定理只针对具有正间隔的线性可分数据；它不适用于 XOR，不保证最大间隔，也不说明非可分数据上的平均表现。

## A.2.1 设定与更新约定

令训练集

$$
D=\{(x_i,y_i)\}_{i=1}^N,
\qquad x_i\in\mathbb R^d,
\quad y_i\in\{-1,+1\}.
$$

若模型含偏置 $b$，可使用增广向量

$$
\widetilde x_i=(x_i,1),
\qquad
\widetilde w=(w,b).
$$

下文仍将增广后的向量记作 $x_i,w$。因此 $R$ 与 $\gamma$ 都在同一个增广欧氏空间中计算；它们不能一个在原输入空间、另一个在增广空间。

感知机从 $w_0=0$ 开始。每当呈现样本 $(x_i,y_i)$ 且

$$
y_iw_t^\mathsf Tx_i\le0,
\tag{A.2.1}
$$

就更新

$$
w_{t+1}=w_t+\eta y_ix_i,
\qquad \eta>0.
\tag{A.2.2}
$$

条件 (A.2.1) 明确规定落在决策边界上的样本也触发更新。下文用 $k$ 只计更新次数，而不是扫描过的样本数。

## A.2.2 Novikoff 错误次数界

**定理 A.2.1（感知机错误次数界）** 假设存在单位向量 $u$、常数 $\gamma>0$ 与 $R<\infty$，使对所有训练样本

$$
y_i u^\mathsf Tx_i\ge\gamma,
\qquad
\|x_i\|\le R.
\tag{A.2.3}
$$

则无论这些样本以何种顺序呈现，按 (A.2.1)--(A.2.2) 更新的总次数 $k$ 满足

$$
k\le\left(\frac R\gamma\right)^2.
\tag{A.2.4}
$$

特别地，若算法反复完整扫描有限训练集，直到某一轮没有更新，则它会在有限次更新后得到一个正确分类全部训练样本的参数。

**证明** 将第 $r$ 次触发更新的样本记作 $(x^{(r)},y^{(r)})$，更新后的参数记作 $w_r$。由 (A.2.3)，

$$
\begin{aligned}
u^\mathsf Tw_r
&=u^\mathsf Tw_{r-1}+\eta y^{(r)}u^\mathsf Tx^{(r)}\\
&\ge u^\mathsf Tw_{r-1}+\eta\gamma.
\end{aligned}
$$

从 $w_0=0$ 归纳得到

$$
u^\mathsf Tw_k\ge k\eta\gamma.
\tag{A.2.5}
$$

另一方面，触发更新意味着
$y^{(r)}w_{r-1}^\mathsf Tx^{(r)}\le0$，故

$$
\begin{aligned}
\|w_r\|^2
&=\|w_{r-1}+\eta y^{(r)}x^{(r)}\|^2\\
&=\|w_{r-1}\|^2
+2\eta y^{(r)}w_{r-1}^\mathsf Tx^{(r)}
+\eta^2\|x^{(r)}\|^2\\
&\le\|w_{r-1}\|^2+\eta^2R^2.
\end{aligned}
$$

因此

$$
\|w_k\|^2\le k\eta^2R^2.
\tag{A.2.6}
$$

由 Cauchy--Schwarz、$\|u\|=1$ 及 (A.2.5)--(A.2.6)，当 $k>0$ 时

$$
k^2\eta^2\gamma^2
\le(u^\mathsf Tw_k)^2
\le\|w_k\|^2
\le k\eta^2R^2.
$$

消去 $k\eta^2$ 即得 (A.2.4)。若反复扫描训练集，超过该上界后不可能再触发更新；下一轮完整扫描因而无更新并终止。$\square$

<img src="images/margin_gamma_comparison.png" width="100%" />

## A.2.3 这个界说明什么

1. **它是顺序无关的最坏情形界。** 样本顺序会影响实际更新数和最终分隔面，但不会破坏 (A.2.4)。
2. **它由相对间隔决定。** 统一缩放全部 $x_i$ 会同时缩放 $R$ 与 $\gamma$，比值不变。非均匀特征缩放会改变所用几何，可能增大也可能减小该比值。
3. **它不是样本复杂度界。** $N$ 没有显式出现，但增加样本可能缩小可达到的间隔。
4. **它不选择最大间隔解。** 定理只保证找到某个训练集分隔面；SVM 的优化目标不同。

<img src="images/convergence_speed_gamma.png" width="100%" />

## A.2.4 假设失败时的反例

取一维两个样本

$$
(x_1,y_1)=(1,+1),
\qquad
(x_2,y_2)=(1,-1).
$$

它们不可能被同一个齐次线性分类器正确分类。令 $\eta=1$ 并按 $x_1,x_2,x_1,x_2,\ldots$ 循环呈现：从 $w=0$ 出发，$x_1$ 使 $w$ 更新为 $1$，随后 $x_2$ 使 $w$ 更新回 $0$，过程永久循环。这里不是证明失效而算法仍应收敛，而是正间隔假设本身不成立。

有限数据若严格线性可分，则归一化某个分隔向量后，有限个正数 $y_i u^\mathsf Tx_i$ 的最小值自动给出 $\gamma>0$。对无限数据流，仅逐点严格为正并不保证存在统一正间隔。

## A.2.5 来源

- Novikoff, [*On Convergence Proofs for Perceptrons*](https://cs.uwaterloo.ca/~y328yu/classics/novikoff.pdf), 1962。
- Rosenblatt, [*The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*](https://doi.org/10.1037/h0042519), 1958。
