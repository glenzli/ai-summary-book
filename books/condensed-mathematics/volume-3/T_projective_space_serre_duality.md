# 附录 T：射影空间上线丛的 Serre 对偶

## T.0 目标

附录 S 计算了

$$
H^q(\mathbb P^n,\mathcal O(d)).
$$

本附录在该计算基础上，直接证明 $\mathbb P^n$ 上线丛情形的 Serre 对偶：

$$
H^q(\mathbb P^n,\mathcal O(d))^\vee
\cong
H^{n-q}(\mathbb P^n,\mathcal O(-d-n-1)).
$$

这不是一般 Serre duality 的替代证明；它是一个完全可计算模型，用于检查符号、次数和 trace。

## T.1 Canonical bundle

**命题 T.1.** 有

$$
\omega_{\mathbb P^n}\simeq\mathcal O_{\mathbb P^n}(-n-1).
$$

**证明.** Euler sequence 为

$$
0\to\mathcal O_{\mathbb P^n}
\to
\mathcal O_{\mathbb P^n}(1)^{\oplus(n+1)}
\to
T_{\mathbb P^n}
\to0.
$$

取 determinant 得

$$
\det T_{\mathbb P^n}\simeq\mathcal O(n+1).
$$

因此

$$
\omega_{\mathbb P^n}
=\det \Omega^1_{\mathbb P^n}
\simeq
(\det T_{\mathbb P^n})^\vee
\simeq
\mathcal O(-n-1).
$$

证毕。

## T.2 Čech residue functional

使用标准覆盖 $U_i=\{X_i\ne0\}$。附录 S 中 $H^n(\mathbb P^n,\mathcal O(-n-1))$ 由唯一 Laurent 单项式

$$
(X_0X_1\cdots X_n)^{-1}
$$

生成。

**定义 T.2.** 定义 trace

$$
\operatorname{Tr}:H^n(\mathbb P^n,\omega_{\mathbb P^n})\to\mathbb C
$$

为取 Čech 代表中单项式 $(X_0\cdots X_n)^{-1}$ 的系数。

**命题 T.3.** 该 trace 与 Čech coboundary 相容，因此定义在上同调上。

**证明.** 附录 S 的单项式分解表明 $H^n(\mathcal O(-n-1))$ 只由负指标集合为全体顶点的单项式贡献。Čech coboundary 的像来自次数 $n-1$ 的 cochains；对单项式分量而言，只有负指标集合为非空真子集时可能出现在次数 $n-1$，而这些分量在引理 S.1 中 acyclic，不产生全负单项式的 $H^n$ 类。因此取该系数在 coboundary 上为零。证毕。

## T.3 单项式配对

对整数 $d$，乘法给

$$
\mathcal O(d)\otimes\mathcal O(-d-n-1)
\to
\mathcal O(-n-1)=\omega_{\mathbb P^n}.
$$

复合 trace 得配对

$$
H^q(\mathcal O(d))
\times
H^{n-q}(\mathcal O(-d-n-1))
\to
\mathbb C.
$$

**定理 T.4.** 该配对是完美配对。

**证明.** 由附录 S，只有两种情形需要检查。

若 $q=0$，则 $H^0(\mathcal O(d))$ 非零当且仅当 $d\ge0$，基为普通单项式

$$
X^a=X_0^{a_0}\cdots X_n^{a_n},
\qquad a_i\ge0,\quad \sum a_i=d.
$$

对偶群

$$
H^n(\mathcal O(-d-n-1))
$$

的基由全负单项式 $X^b$ 给出，其中

$$
b_i<0,\qquad \sum b_i=-d-n-1.
$$

条件

$$
X^aX^b=(X_0\cdots X_n)^{-1}
$$

等价于

$$
b_i=-a_i-1.
$$

因此每个 $X^a$ 与唯一的 $X^{-a-\mathbf 1}$ 配对为 $1$，与其他基向量配对为 $0$。矩阵为置换矩阵，故完美。

若 $q=n$，同一论证交换两边。

若 $0<q<n$，附录 S 给出两边同为零。零向量空间与零向量空间之间的配对是完美配对。证毕。

## T.4 与一般 Serre duality 的关系

一般 Serre duality 断言对光滑 proper $n$ 维复流形或光滑射影簇上的相干层 $\mathcal F$，

$$
H^q(X,\mathcal F)^\vee
\cong
\operatorname{Ext}^{n-q}(\mathcal F,\omega_X).
$$

对 $\mathcal F=\mathcal O(d)$，因为 $\mathcal O(d)$ 是线丛，

$$
\operatorname{Ext}^{n-q}(\mathcal O(d),\omega)
\cong
H^{n-q}(\omega\otimes\mathcal O(-d)).
$$

命题 T.1 把右侧化为

$$
H^{n-q}(\mathcal O(-d-n-1)).
$$

本附录证明的是该公式在 $\mathbb P^n$ 线丛上的完整单项式模型。

## 练习

1. 对 $\mathbb P^2$ 和 $d=1$，写出 $H^0(\mathcal O(1))$ 与 $H^2(\mathcal O(-4))$ 的配对矩阵。
2. 对 $\mathbb P^1$ 和 $d=-2$，验证 $H^1(\mathcal O(-2))^\vee\cong H^0(\mathcal O(0))$。
3. 用 Euler sequence 直接计算 $\omega_{\mathbb P^1}$ 和 $\omega_{\mathbb P^2}$。
4. 解释为什么本附录不能推出一般相干层的 Serre duality。
