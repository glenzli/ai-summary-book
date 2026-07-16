# 附录 B：实分析、凸性与极限工具

本附录记录信息论和概率不等式反复使用的初等分析工具。一般 Lebesgue 极限交换由外部输入承担；有限和式中的凸性证明在此闭合。

## B.1 对数不等式

对 $u>0$，自然对数满足

$$
\ln u\le u-1,
$$

等号当且仅当 $u=1$。这可由函数 $f(u)=u-1-\ln u$ 证明：$f'(u)=1-1/u$，在 $u=1$ 取最小值 $0$。由于信息论中的 $\log$ 以 $2$ 为底，实际使用的是

$$
\log u\le\frac{u-1}{\ln2}.
$$

**命题 B.1（log-sum 不等式）。** 若 $a_i,b_i\ge0$，$a=\sum_i a_i$，$b=\sum_i b_i$，并采用 KL 边界约定，则

$$
\sum_i a_i\log\frac{a_i}{b_i}\ge a\log\frac ab.
$$

**证明.** 若 $a=0$，两边均为 $0$。若 $b=0<a$，两边均为 $+\infty$。以下设 $a,b>0$。若某个 $a_i>0,b_i=0$，左边为 $+\infty$，结论成立。否则令 $\alpha_i=a_i/a$、$\beta_i=b_i/b$，并令 $S=\{i:\alpha_i>0\}$。代数展开给出

$$
\sum_i a_i\log\frac{a_i}{b_i}
=a\log\frac ab+a\sum_{i\in S}\alpha_i\log\frac{\alpha_i}{\beta_i}.
$$

最后一项不依赖第 8 章的 Gibbs 不等式即可控制。由上一段的对数不等式，

$$
\sum_{i\in S}\alpha_i\log\frac{\beta_i}{\alpha_i}
\le\frac1{\ln2}\sum_{i\in S}(\beta_i-\alpha_i)
=\frac1{\ln2}\left(\sum_{i\in S}\beta_i-1\right)\le0.
$$

因此 $\sum_{i\in S}\alpha_i\log(\alpha_i/\beta_i)\ge0$，代回即得 log-sum 不等式。证毕。

## B.2 熵的凹性

**命题 B.2（有限熵凹性）。** 对有限集合上的分布 $p^{(1)},\ldots,p^{(m)}$ 和权重 $\lambda_i\ge0$、$\sum_i\lambda_i=1$，

$$
H\left(\sum_i\lambda_ip^{(i)}\right)\ge \sum_i\lambda_iH(p^{(i)}).
$$

**证明.** 函数 $\phi(u)=-u\log u$ 在 $[0,1]$ 上凹，因为在 $(0,1)$ 上二阶导数为 $-1/(u\ln 2)<0$，端点按连续延拓处理。对每个坐标使用有限 Jensen 不等式，再对坐标求和。证毕。

## B.3 极限交换边界

单调收敛、Fatou 与控制收敛定理是外部输入 EI-2。正文中凡把极限移入期望，必须满足以下至少一种情况：

- 有限和式，直接交换；
- 非负递增，使用单调收敛；
- 由可积函数控制，使用控制收敛；
- 独立的外部极限定理已经明确登记。

若只知道逐点收敛，没有控制或单调结构，不能交换期望与极限。
