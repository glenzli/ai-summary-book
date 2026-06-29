# 附录 AK：Deformation to the normal cone 与 GRR 证明模块

## AK.0 目标

附录 AE 把 GRR 作为输入。本附录拆解一种经典证明路线：

1. GRR 对 projective bundle。
2. GRR 对 regular closed immersion。
3. deformation to the normal cone 把 closed immersion 化到 normal bundle。
4. 一般 proper morphism 由 graph embedding 和 projection 分解。

完整证明仍依赖 Chow/cohomology、Chern class、excess intersection 和 specialization。

## AK.1 Projective bundle 情形

**输入定理 AK.1（projective bundle formula）.** 设 \(p:\mathbb P(E)\to X\)，\(\xi=c_1(\mathcal O_{\mathbb P(E)}(1))\)。则

$$
H^\ast(\mathbb P(E),\mathbb Q)
\cong
\bigoplus_{i=0}^{r-1}H^{\ast-2i}(X,\mathbb Q)\xi^i.
$$

**输入定理 AK.2（relative Euler sequence）.** 有 exact sequence

$$
0\to\mathcal O\to p^\ast E\otimes\mathcal O(1)\to T_{\mathbb P(E)/X}\to0.
$$

**命题 AK.3.** GRR 对 projective bundle projection \(p\) 可化为 fiber 上 \(\mathbb P^{r-1}\) 的积分恒等式。

**证明.** 由 AK.1，任意 cohomology class 可唯一写成 \(\sum a_i\xi^i\)。pushforward \(p_\ast\) 由 fiber integration 决定，满足 \(p_\ast(\xi^{r-1})=1\) 和 Segre class 公式。Todd class 由 AK.2 计算。代入 GRR 后化为关于 \(\xi\) 的通用多项式恒等式，可在 universal splitting 情形检查。证毕。

## AK.2 Regular immersion 情形

设 \(i:Z\hookrightarrow X\) 是 regular immersion，法丛为 \(N\)。

**输入定理 AK.4（self-intersection formula）.** 对 \(\alpha\in H^\ast(Z)\)，有

$$
i^\ast i_\ast\alpha=c_c(N)\alpha
$$

其中 \(c=\operatorname{rank}N\)。

**输入定理 AK.5（Koszul GRR for zero section）.** 对向量丛 \(N\to Z\) 的 zero section \(s:Z\to N\)，GRR 成立，并由 Koszul complex 和 Thom class 给出。

## AK.3 Deformation to the normal cone

**输入定理 AK.6（deformation to the normal cone）.** 对 closed immersion \(i:Z\hookrightarrow X\)，存在空间

$$
\mathcal D=\operatorname{Bl}_{Z\times\{0\}}(X\times\mathbb A^1)
$$

使一般 fiber 同构于 \(X\)，特殊 fiber 包含 normal cone \(C_{Z/X}\)。若 \(i\) regular，则

$$
C_{Z/X}\simeq N_{Z/X}.
$$

**命题 AK.7.** 若 GRR 对 zero section \(Z\to N_{Z/X}\) 成立，且 characteristic classes 与 specialization 相容，则 GRR 对 regular immersion \(i:Z\to X\) 成立。

**证明.** deformation family 把 \(i\) specialization 到 zero section。K-theory pushforward、cohomology pushforward、Chern character 和 Todd class 在 specialization 下相容。一般 fiber 上的 GRR 等式等价于特殊 fiber 上的 GRR 等式；特殊 fiber 情形由 AK.5。证毕。

## AK.4 一般 proper morphism

**输入定理 AK.8（Chow lemma / graph factorization）.** 对 projective morphism \(f:X\to Y\)，可分解为

$$
X\xrightarrow{\Gamma_f}X\times Y\xrightarrow{p}Y
$$

其中 \(\Gamma_f\) 是 closed immersion，\(p\) 是 projection；在嵌入到 projective bundle 后，可化为 regular immersion 与 projective bundle projection 的组合。

**定理 AK.9（GRR 证明模块）.** 若 GRR 对 projective bundle projection 和 regular immersion 成立，并且对复合相容，则 GRR 对 projective morphism 成立。

**证明.** 取 AK.8 的分解。对每个因子应用 GRR，再用附录 AE 的复合相容命题拼接，得到 \(f\) 的 GRR。证毕。

## AK.5 边界

完整 GRR 证明还需要：

1. K-theory/G-theory 的精确定义；
2. Chern character 与 localized Chern character；
3. deformation specialization 的 functoriality；
4. singular 情形的 perfect complex 或 \(G_0\) 版本；
5. excess intersection formula。

本附录只记录光滑/projective 情形的证明骨架。

## 练习

1. 对 \(p:\mathbb P^1_X\to X\)，写出 projective bundle formula。
2. 用 Koszul complex 计算 zero section 的 \(K\)-theory pushforward。
3. 解释 deformation to normal cone 如何把一般 closed immersion 化为 normal bundle 问题。
4. 说明 AK.9 为什么需要 GRR 对复合相容。
