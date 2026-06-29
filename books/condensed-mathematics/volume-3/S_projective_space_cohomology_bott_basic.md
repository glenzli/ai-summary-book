# 附录 S：射影空间上线丛上同调的单项式计算

## S.0 目标

附录 H 完整计算了 $\mathbb P^1$ 上 $\mathcal O(d)$ 的 Čech 上同调。本附录把计算推广到 $\mathbb P^n$，得到基础 Bott 公式：

$$
H^q(\mathbb P^n,\mathcal O(d))
$$

只在 $q=0$ 或 $q=n$ 可能非零。证明使用标准仿射覆盖和 Laurent 单项式分解，不调用 Borel-Weil-Bott。

## S.1 标准覆盖与 Čech 复形

令

$$
\mathbb P^n=\operatorname{Proj}\mathbb C[X_0,\ldots,X_n].
$$

设

$$
U_i=\{X_i\ne0\}.
$$

对非空子集 $I\subset\{0,\ldots,n\}$，记

$$
U_I=\bigcap_{i\in I}U_i.
$$

则

$$
\Gamma(U_I,\mathcal O(d))
=
\mathbb C[X_0,\ldots,X_n,X_i^{-1}\ (i\in I)]_d,
$$

即总次数为 $d$ 的齐次 Laurent 多项式，其中只允许对 $I$ 中变量取负指数。

Čech $p$-cochains 为

$$
\check C^p(\mathfrak U,\mathcal O(d))
=
\prod_{|I|=p+1}\Gamma(U_I,\mathcal O(d)).
$$

由于所有 $U_I$ 都是仿射空间与 torus 的乘积，且为 Stein，Cartan B 给出该 Čech 复形计算 sheaf cohomology。

## S.2 单项式分解

对整数向量

$$
a=(a_0,\ldots,a_n)\in\mathbb Z^{n+1},
\qquad
|a|=\sum_i a_i=d,
$$

记

$$
X^a=X_0^{a_0}\cdots X_n^{a_n}.
$$

定义负指标集合

$$
N(a)=\{i\mid a_i<0\}.
$$

单项式 $X^a$ 出现在 $\Gamma(U_I,\mathcal O(d))$ 中，当且仅当

$$
N(a)\subset I.
$$

因此 Čech 复形按单项式直和分解：

$$
\check C^\bullet(\mathfrak U,\mathcal O(d))
=
\bigoplus_{|a|=d} C_a^\bullet,
$$

其中

$$
C_a^p=
\bigoplus_{\substack{|I|=p+1\\N(a)\subset I}}\mathbb C\cdot e_I.
$$

微分为交错限制：

$$
(\delta c)_I=\sum_{r=0}^{p+1}(-1)^r c_{I\setminus\{i_r\}},
\qquad I=\{i_0<\cdots<i_{p+1}\},
$$

不存在的项记为零。

**引理 S.1.** $C_a^\bullet$ 的同调如下：

1. 若 $N(a)=\varnothing$，则 $H^0(C_a^\bullet)\cong\mathbb C$，其余同调为零。
2. 若 $N(a)=\{0,\ldots,n\}$，则 $H^n(C_a^\bullet)\cong\mathbb C$，其余同调为零。
3. 若 $N(a)$ 是非空真子集，则 $C_a^\bullet$ acyclic。

**证明.** 令 $\Delta$ 为顶点集 $\{0,\ldots,n\}$ 上的 $n$-单形。$C_a^p$ 的基由包含 $N(a)$ 的 $p$-面给出。

若 $N(a)=\varnothing$，则 $C_a^\bullet$ 是 $\Delta$ 的普通增广前 cochain 复形。单形可收缩，所以只有 $H^0\cong\mathbb C$。

若 $N(a)$ 是非空真子集，则包含 $N(a)$ 的所有面构成 $\Delta$ 中以 $N(a)$ 为最小面的 star。这个 star 是可收缩的，并且它在 Čech 复形中的起始次数为 $|N(a)|-1$。交错微分给出的复形是该可收缩 star 的相对 cochain 复形，带有初始项的同伦由选择一个顶点 $j\notin N(a)$ 后的锥算子给出：

$$
h(e_I)=
\begin{cases}
(-1)^{\epsilon(I,j)}e_{I\cup\{j\}},&j\notin I,\\
0,&j\in I.
\end{cases}
$$

直接代入交错微分得

$$
\delta h+h\delta=\operatorname{id}.
$$

故同调为零。

若 $N(a)$ 是全顶点集，则只有 $I=\{0,\ldots,n\}$ 贡献，位于次数 $n$，微分前后都为零，所以 $H^n\cong\mathbb C$。证毕。

## S.3 Bott 公式的基础情形

**定理 S.2.** 对任意整数 $d$，

$$
H^q(\mathbb P^n,\mathcal O(d))=
\begin{cases}
\mathbb C[X_0,\ldots,X_n]_d,& q=0,\ d\ge0,\\
\mathbb C[X_0,\ldots,X_n]_{-d-n-1}^{\vee},& q=n,\ d\le -n-1,\\
0,& \text{其他情形}.
\end{cases}
$$

**证明.** 由 S.1，cohomology 是所有 $C_a^\bullet$ 同调的直和。

若 $N(a)=\varnothing$，则所有 $a_i\ge0$，并且 $\sum a_i=d$。这种 $a$ 存在当且仅当 $d\ge0$。它们给出次数 $d$ 的齐次多项式基，因此

$$
H^0(\mathbb P^n,\mathcal O(d))
\cong
\mathbb C[X_0,\ldots,X_n]_d.
$$

若 $N(a)=\{0,\ldots,n\}$，则所有 $a_i<0$。令

$$
b_i=-a_i-1\ge0.
$$

则

$$
\sum_i b_i=-\sum_i a_i-(n+1)=-d-n-1.
$$

这种 $a$ 存在当且仅当 $-d-n-1\ge0$，即 $d\le -n-1$。单项式 $X^a$ 与次数 $-d-n-1$ 的普通单项式 $X^b$ 一一对应。选择 Čech 上的 residue pairing 后，该空间自然识别为

$$
\mathbb C[X_0,\ldots,X_n]_{-d-n-1}^{\vee}.
$$

非空真负指标集合的贡献由引理 S.1 消失，因此中间上同调全为零。证毕。

**推论 S.3.** 对 $0<q<n$，

$$
H^q(\mathbb P^n,\mathcal O(d))=0
$$

对所有 $d\in\mathbb Z$ 成立。

**推论 S.4.** 当 $n=1$ 时，

$$
\dim H^0(\mathbb P^1,\mathcal O(d))=\max(d+1,0),
$$

且

$$
\dim H^1(\mathbb P^1,\mathcal O(d))=\max(-d-1,0).
$$

这与附录 H 的计算一致。

## S.4 Serre 对偶的单项式配对

$\mathbb P^n$ 的 canonical bundle 为

$$
\omega_{\mathbb P^n}\simeq\mathcal O(-n-1).
$$

因此 Serre duality 对 $\mathcal O(d)$ 给出

$$
H^q(\mathbb P^n,\mathcal O(d))^\vee
\cong
H^{n-q}(\mathbb P^n,\mathcal O(-d-n-1)).
$$

定理 S.2 中 $q=0$ 与 $q=n$ 的两个公式正互相匹配：次数 $d$ 的多项式与次数

$$
-(-d-n-1)-n-1=d
$$

的多项式对偶。

在 Čech 表示中，配对由 Laurent 单项式的 residue 给出：只有指数向量满足

$$
a_i+b_i=-1\qquad(0\le i\le n)
$$

时配对非零。这正是 $X^aX^b=(X_0\cdots X_n)^{-1}$ 的系数抽取。

## S.5 Euler characteristic

由定理 S.2，

$$
\chi(\mathbb P^n,\mathcal O(d))
=
\sum_q(-1)^q\dim H^q(\mathbb P^n,\mathcal O(d)).
$$

对 $d\ge0$，

$$
\chi=\binom{n+d}{n}.
$$

对 $d\le -n-1$，

$$
\chi=(-1)^n\binom{-d-1}{n}.
$$

这两个表达式是同一个多项式

$$
\binom{d+n}{n}
=
\frac{(d+1)(d+2)\cdots(d+n)}{n!}
$$

在整数上的取值，其中负整数处按多项式解释。中间区间 $-n\le d\le -1$ 中 cohomology 全部为零，该多项式也为零。

这给出 Hirzebruch-Riemann-Roch 在 $\mathbb P^n$ 线丛情形下的一个可计算检验。

## 练习

1. 对 $\mathbb P^2$，列出 $\mathcal O(-3)$ 的 $H^2$ 的一个 Čech 单项式基。
2. 用引理 S.1 的锥同伦证明 $N(a)=\{0\}$ 时 $C_a^\bullet$ acyclic。
3. 计算 $\chi(\mathbb P^3,\mathcal O(d))$，并验证它等于 $\binom{d+3}{3}$。
4. 写出 $\mathbb P^2$ 上 $\mathcal O(1)$ 与 $\mathcal O(-4)$ 的 Serre 对偶单项式配对。
