# 第十九章：几何 Satake

## 本章目标

本章陈述几何 Satake 等价。第十八章说明 Hecke 修改的局部模型是 affine Grassmannian
$$
\operatorname{Gr}_G=G((t))/G[[t]].
$$
几何 Satake 定理说：$\operatorname{Gr}_G$ 上的 $G[[t]]$-等变 perverse sheaves 构成一个张量范畴，并且该范畴等价于对偶群 $\widehat G$ 的有限维表示范畴。这个定理是几何 Langlands 中对偶群出现的几何来源。

## 依赖前置知识

需要第十一章的对偶群和根资料，第十八章的 affine Grassmannian 与 Schubert varieties。需要 perverse sheaves、intersection cohomology、卷积、Tannakian category 和 equivariant derived category 的基础。本章把几何 Satake 等价作为外部输入定理，只证明若干形式后果。附录 Y 给出 Ran space、Beilinson-Drinfeld Grassmannian、factorization 和 fusion 对本章张量结构的技术支撑。

收口归一化回指：本章比较几何 Satake 与经典 Satake 时必须追踪 $q^{\langle\rho,\lambda\rangle}$、Tate twist 和 sheaf-function convention；见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、9 节。

## 19.1 Affine Grassmannian 的 Schubert 分层

设 $k$ 为代数闭域，$G/k$ 为 split connected reductive group。取 split maximal torus $T\subset G$ 和 Borel subgroup $B$，dominant coweights 集合记为
$$
X_*(T)^+.
$$

**定义 19.1.** 对 $\lambda\in X_*(T)^+$，令 $t^\lambda\in G(k((t)))$ 为对应 loop。Affine Grassmannian 中的 Schubert cell 定义为
$$
\operatorname{Gr}_G^\lambda=G(k[[t]])\cdot t^\lambda.
$$
其 Zariski 闭包记为
$$
\overline{\operatorname{Gr}}_G^\lambda.
$$

**外部输入定理 19.2（Schubert 分解）.** 有不交并
$$
\operatorname{Gr}_G=\bigsqcup_{\lambda\in X_*(T)^+}\operatorname{Gr}_G^\lambda.
$$
每个 $\overline{\operatorname{Gr}}_G^\lambda$ 是有限维 projective variety，且
$$
\overline{\operatorname{Gr}}_G^\lambda
=
\bigcup_{\mu\le\lambda}\operatorname{Gr}_G^\mu
$$
其中 $\le$ 为 dominant order。

**定义 19.3.** 记
$$
\operatorname{IC}_\lambda
$$
为 Schubert variety $\overline{\operatorname{Gr}}_G^\lambda$ 的 intersection cohomology perverse sheaf，延拓到 $\operatorname{Gr}_G$ 上。

## 19.2 卷积

几何 Satake 的张量结构来自 convolution。

**定义 19.4.** 卷积 Grassmannian $\operatorname{Gr}_G\tilde\times\operatorname{Gr}_G$ 是商
$$
G((t))\times^{G[[t]]}\operatorname{Gr}_G.
$$
它带有映射
$$
m:\operatorname{Gr}_G\tilde\times\operatorname{Gr}_G\to\operatorname{Gr}_G
$$
由 loop 乘法诱导。

**定义 19.5.** 对 $G[[t]]$-等变 perverse sheaves $\mathcal F,\mathcal G$，卷积定义为
$$
\mathcal F*\mathcal G
=
m_!(\mathcal F\tilde\boxtimes\mathcal G).
$$
其中 $\mathcal F\tilde\boxtimes\mathcal G$ 是卷积 Grassmannian 上由等变性下降得到的外积 sheaf。

**外部输入定理 19.6（卷积的 perversity）.** 若 $\mathcal F,\mathcal G$ 为 $G[[t]]$-等变 perverse sheaves，则
$$
\mathcal F*\mathcal G
$$
仍为 perverse sheaf。

**注 19.7.** 定理 19.6 是几何 Satake 的关键几何输入之一。普通 proper pushforward 不自动保持 perverse；这里依赖 affine Grassmannian Schubert 几何的半小性。

## 19.3 Satake 范畴

**定义 19.8.** Satake 范畴定义为
$$
\operatorname{Sat}_G=\operatorname{Perv}_{G[[t]]}(\operatorname{Gr}_G)
$$
即 affine Grassmannian 上的 $G[[t]]$-等变 perverse sheaves 范畴，系数取 $\overline{\mathbb Q}_\ell$ 或 $\mathbb C$，依 sheaf theory 而定。

卷积 $*$ 使 $\operatorname{Sat}_G$ 成为 monoidal category。

**定义 19.9.** 全上同调函子
$$
\mathsf H^\bullet:\operatorname{Sat}_G\to\operatorname{Vect}
$$
定义为
$$
\mathsf H^\bullet(\mathcal F)=H^\bullet(\operatorname{Gr}_G,\mathcal F).
$$

**外部输入定理 19.10（Tannakian 性）.** 范畴 $\operatorname{Sat}_G$ 配备卷积和全上同调纤维函子后，是 neutral Tannakian category。

## 19.4 几何 Satake 定理

**外部输入定理 19.11（几何 Satake）.** 有张量范畴等价
$$
\operatorname{Sat}_G
\simeq
\operatorname{Rep}(\widehat G),
$$
其中 $\widehat G$ 是第十一章定义的复对偶群或相应系数域上的 Langlands dual group。在该等价下，
$$
\operatorname{IC}_\lambda
$$
对应于 $\widehat G$ 的最高权为 $\lambda$ 的不可约表示 $V_\lambda$。

**注 19.12.** 这里 $\lambda\in X_*(T)^+$ 在对偶群中成为 dominant weight，因为
$$
X^*(\widehat T)=X_*(T).
$$
这正是根资料对偶在几何中的体现。

**命题 19.13.** 在几何 Satake 下，卷积对应张量积：
$$
\operatorname{IC}_\lambda*\operatorname{IC}_\mu
\quad\leftrightarrow\quad
V_\lambda\otimes V_\mu.
$$

**证明.** 几何 Satake 定理 19.11 是张量范畴等价。张量范畴等价按定义把源范畴的 monoidal product 送到目标范畴的 tensor product。因此卷积对应张量积。$\square$

## 19.5 Hecke 函子与表示范畴

第十八章定义了 Hecke 函子 $\mathsf H_\lambda$。几何 Satake 将所有 $\lambda$ 的 Hecke 函子组织成 $\operatorname{Rep}(\widehat G)$ 的作用。

**定义 19.14.** 对 $V\in\operatorname{Rep}(\widehat G)$，令 $\mathcal S_V$ 表示几何 Satake 下对应的 perverse sheaf。定义 Hecke 函子
$$
\mathsf H_V:\mathcal D(\operatorname{Bun}_G)\to\mathcal D(\operatorname{Bun}_G\times X)
$$
为由 Hecke correspondence 和 kernel $\mathcal S_V$ 给出的函子。

**收口精修 19.A（最小 Hecke 作用模型）.** 若 $V=V_\lambda$ 是最高权 $\lambda$ 的 $\widehat G$-表示，则 $\mathcal S_V$ 可看作 affine Grassmannian 中 Schubert 闭包 $\overline{\operatorname{Gr}}_\lambda$ 上的交叉上同调 sheaf。Hecke 函子 $\mathsf H_V$ 在点 $x\in X$ 处允许 $G$-bundle 发生相对位置不超过 $\lambda$ 的修改。对 $G=\mathbb G_m$，这只是把线丛张量 $\mathcal O_X(nx)$；取 Frobenius trace 后，对应的函数操作就是经典 Hecke 算子在该点的求和。一般 $G$ 的定义把这个线丛例子替换为由 $\operatorname{Rep}(\widehat G)$ 控制的修改类型。

**命题 19.15.** 对 $V,W\in\operatorname{Rep}(\widehat G)$，Hecke 函子满足形式相容
$$
\mathsf H_V\circ\mathsf H_W
\simeq
\mathsf H_{V\otimes W}
$$
在允许两个修改点合并并使用 factorization 结构的意义下成立。

**证明草图.** 两次 Hecke 修改的局部模型是卷积 Grassmannian。几何 Satake 把卷积 sheaf $\mathcal S_V*\mathcal S_W$ 识别为 $\mathcal S_{V\otimes W}$。将该局部识别沿 Hecke correspondence 全局化，得到 Hecke 函子的张量相容性。完整证明需要 Beilinson-Drinfeld Grassmannian 的 factorization 结构。$\square$

## 19.6 与经典 Satake 的关系

**命题 19.16.** 若 $k=\mathbb F_q$，对 $\operatorname{Sat}_G$ 中的 sheaf 取 Frobenius trace，可得到 spherical Hecke algebra 的函数；几何 Satake 的 Grothendieck ring 版本恢复经典 Satake 同构。

**证明草图.** Grothendieck sheaf-function dictionary 把 $G[[t]]$-等变 sheaves 送到 $G(\mathcal O)$-双不变紧支撑函数。卷积 sheaf 的 Frobenius trace 对应函数卷积。几何 Satake 把 Grothendieck ring 识别为 $\operatorname{Rep}(\widehat G)$ 的表示环；经典 Satake 同构也把球 Hecke 代数识别为该表示环的半 Tate twist，即 $q^{1/2}$ 归一化形式。$\square$

**注 19.17.** 因此几何 Satake 不只是几何类比；它给出经典 Satake 同构的范畴化。

## 19.7 本章小结

几何 Satake 定理建立了核心等价
$$
\operatorname{Perv}_{G[[t]]}(\operatorname{Gr}_G)\simeq\operatorname{Rep}(\widehat G).
$$
Affine Grassmannian 的 Schubert variety 对应 $\widehat G$ 的最高权表示，卷积对应张量积。由此，$\operatorname{Rep}(\widehat G)$ 作用在 $\operatorname{Bun}_G$ 上的 sheaf 范畴；这正是 Hecke eigensheaf 定义的基础。

## 练习

**练习 19.1.** 对 $G=\operatorname{GL}_n$，说明 dominant coweight 如何成为 $\widehat G=\operatorname{GL}_n(\mathbb C)$ 的 dominant weight。

**练习 19.2.** 写出卷积 Grassmannian 的定义，并解释其中 $G[[t]]$ 的作用。

**练习 19.3.** 说明为什么几何 Satake 中卷积应对应表示张量积。

**练习 19.4.** 对 $G=\mathbb G_m$，描述 $\operatorname{Gr}_G$ 的 connected components，并猜测 Satake 范畴。

**练习 19.5.** 解释 sheaf-function dictionary 如何把几何 Satake 降为经典 Satake。
