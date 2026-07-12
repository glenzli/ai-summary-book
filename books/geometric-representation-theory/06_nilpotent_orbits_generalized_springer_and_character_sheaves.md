# 第六章：Nilpotent orbits、generalized Springer correspondence 与 character sheaves

## 本章目标

本章把第五章的 Springer theory 推向两个方向：nilpotent orbits 的局部系统参数，以及 Lusztig 的 generalized Springer correspondence 和 character sheaves。正文重点是定义和依赖边界，完整分类定理作为外部输入。

## 依赖前置知识

需要第五章的 nilpotent cone、Springer fibers 和 Springer correspondence，以及第三章的 equivariant local systems 和 IC sheaves。

## 6.1 Nilpotent orbits 和 component groups

**定义 6.1.** 对 $x\in\mathcal N$，其 $G$-orbit 记为
$$
\mathcal O_x=G\cdot x.
$$
稳定子为
$$
G_x=\{g\in G\mid \operatorname{Ad}_g x=x\}.
$$
component group 定义为
$$
A_G(x)=G_x/G_x^\circ.
$$

**命题 6.2.** $\mathcal O_x\simeq G/G_x$ 是 smooth locally closed variety，且
$$
\dim\mathcal O_x=\dim G-\dim G_x.
$$

**证明.** orbit map
$$
G\to\mathcal N,\qquad g\mapsto\operatorname{Ad}_g x
$$
的 stabilizer 为 $G_x$，因此它分解为单射 immersion $G/G_x\to\mathcal N$，其像为 $\mathcal O_x$。代数群作用的一般 orbit theorem 给出 orbit 是 locally closed smooth subvariety，并与 $G/G_x$ 同构。维数公式来自齐性空间维数。该证明使用 orbit theorem 作为标准代数群外部输入。$\square$

**命题 6.3.** Irreducible $G$-equivariant local systems on $\mathcal O_x$ 与 irreducible representations of $A_G(x)$ 对应。

**证明.** 由 $\mathcal O_x\simeq G/G_x$ 和命题 A.8，
$$
[\mathcal O_x/G]\simeq BG_x.
$$
在第三章的 Betti 模型中，推论 A.9 给出
$$
\pi_1(BG_x)\simeq\pi_0(G_x(\mathbb C))
=A_G(x)(\mathbb C).
$$
因此 $BG_x$ 上的 finite-rank local systems 等价于有限群 $A_G(x)$ 的 finite-dimensional $E$-representations，irreducible objects 恰对应 irreducible representations。注意这里没有使用正维代数群 $G_x$ 的 algebraic representation category；connected component 在 Betti local-system monodromy 中被消去。$\square$

**例 6.4.** 对 $G=GL_n$，nilpotent orbits 由 partitions $\lambda\vdash n$ 标号，component groups 平凡。因此 Springer correspondence 在 $GL_n$ 情形中只涉及 nilpotent orbit，不涉及非平凡 equivariant local system。该分类来自 Jordan normal form。

**命题 6.4.1.** 对 $G=GL_n$，nilpotent element $x$ 的 centralizer $G_x$ 连通。

**证明.** 把 $x$ 看作向量空间 $V=k^n$ 上的 nilpotent endomorphism，使 $V$ 成为有限生成 torsion $k[t]$-module，其中 $t$ 作用为 $x$。于是
$$
G_x=\operatorname{Aut}_{k[t]}(V).
$$
令 $A=\operatorname{End}_{k[t]}(V)$，则 $G_x=A^\times$。有限维代数 $A$ 的 Jacobson radical 记为 $J$。群 $1+J$ 是连通 unipotent 群，而半单商 $A/J$ 是若干矩阵代数的乘积，其可逆元群是若干 $GL_m$ 的乘积，因而连通。由正合列
$$
1\to 1+J\to A^\times\to (A/J)^\times\to1
$$
得到 $A^\times$ 连通。$\square$

**推论 6.4.2.** $GL_n$ 的 nilpotent orbits 上没有非平凡 irreducible $G$-equivariant local systems。

**证明.** 由命题 6.3，irreducible equivariant local systems 由 $A_G(x)=G_x/G_x^\circ$ 的不可约表示参数化。命题 6.4.1 给出 $G_x=G_x^\circ$，故 component group 平凡。$\square$

## 6.2 Springer correspondence 的归一化

**约定 6.5.** 本书的普通 Springer correspondence 采用如下归一化：regular nilpotent orbit 对应 trivial representation 还是 sign representation 依赖使用 top cohomology action 的 convention。最终版本必须在附录 D 中锁定。本阶段只在需要时明确说明 convention，不跨章节隐含使用。

**外部输入定理 6.6.** 普通 Springer correspondence 给出 $\operatorname{Irr}(W)$ 到 pairs $(\mathcal O,\mathcal L)$ 的单射，其像由 Springer sheaf 的 simple summands 决定。  
用途：Weyl group representation 的几何构造。来源：Springer、Borho-MacPherson、Chriss-Ginzburg。

**定义 6.7.** 对 nilpotent orbit $\mathcal O$ 和 irreducible $G$-equivariant local system $\mathcal L$，定义
$$
\operatorname{IC}(\overline{\mathcal O},\mathcal L)
$$
为 $\mathcal N$ 上的 $G$-equivariant perverse sheaf，通过 middle extension 从 $\mathcal O$ 延拓。

**命题 6.8.** Springer sheaf 的 simple perverse summands 都形如 $\operatorname{IC}(\overline{\mathcal O},\mathcal L)$。

**证明.** 外部输入定理 5.15.1 已给出
$\mathsf{Spr}\in\operatorname{Perv}_G(\mathcal N,E)$ 且 semisimple，所以它在 perverse heart 中分解为有限个 simple objects 的直和，不出现额外 cohomological shifts。$\mathcal N$ 的 $G$-orbits 有限；由定理 3.15 的 equivariant 版本，每个 simple $G$-equivariant perverse sheaf 都是某个 orbit 上 irreducible equivariant local system 的 IC extension。因此每个 simple summand 具有所述形式。该证明链的外部输入是 Springer semismallness、semismall decomposition 和 equivariant simple-perverse classification。$\square$

## 6.3 Generalized Springer correspondence

**定义 6.9.** 一个 cuspidal datum 通常写作
$$
(L,\mathcal O_L,\mathcal L_L),
$$
其中 $L\subset G$ 是 Levi subgroup，$\mathcal O_L\subset\mathcal N_L$ 是 nilpotent orbit，$\mathcal L_L$ 是 $\mathcal O_L$ 上的 irreducible $L$-equivariant local system，并满足 Lusztig 的 cuspidality 条件。

**警告 6.10.** cuspidality 不是“不能由更小 Levi 看出”的直觉短语。严格定义需要 parabolic induction/restriction functors 和 perverse sheaf vanishing 条件。当前阶段不把 cuspidality 用作证明步骤。

**外部输入定理 6.11.** Lusztig generalized Springer correspondence 把 $\mathcal N$ 上的 simple $G$-equivariant perverse sheaves 分解为由 cuspidal data 控制的 series；每个 series 内部由相应 relative Weyl group 的不可约表示参数化。  
来源：Lusztig。进入正文定理链前需补 locator。

## 6.4 Character sheaves

**定义 6.12.** 令 $G$ 作用于自身的方式为 conjugation。$D^b_G(G,E)$ 表示 conjugation-equivariant constructible derived category。Character sheaves 是 Lusztig 定义的一类 simple perverse sheaves on $G$，可由 parabolic induction、cuspidal character sheaves 和若干 admissible complexes 构造。

**外部输入定理 6.13.** 在有限域情形中，Frobenius-stable character sheaves 的 trace functions 与 finite groups of Lie type 的 irreducible characters 有深层关系。  
来源：Lusztig character sheaves theory。当前只作为研究边界和后续章节入口。

**定义 6.14.** 几何诱导 functor 的基本 correspondence 如下。给定 parabolic $P=LU$，考虑
$$
G \xleftarrow{\ a\ } G\times^P P \xrightarrow{\ b\ } L,
$$
其中 $a([g,p])=gpg^{-1}$，$b([g,p])$ 为 $p$ 在 $P/U\simeq L$ 中的像。形式上可定义
$$
\operatorname{Ind}_L^G(\mathcal F)=a_!b^\ast\mathcal F[\text{shift}],
$$
shift 由 perversity normalization 决定。

**命题 6.15.** 上述诱导 functor 的类型依赖 $a$ 的 properness 和 $b$ 的 smoothness 假设；若不记录 shift 和 Tate twist，则不能判定其 t-exactness。

**证明.** $a_!b^\ast$ 是导出 functor。perverse t-structure 下，smooth pullback 需要按相对维数 shift 才 t-exact，proper pushforward 不一般 t-exact，只有在 semismall 或 decomposition theorem 场景下可控制 perverse cohomology。因此没有 shift、twist 和几何假设，不能得出 character sheaf 的 perversity。$\square$

## 本章小结

本章定义了 nilpotent orbit、component group、equivariant local system、ordinary 和 generalized Springer correspondence 的输入数据，以及 character sheaves 的基本范畴。内部证明包括 orbit 和 equivariant local system 的类型识别，以及 $GL_n$ centralizer 连通性和 local system 平凡性；Lusztig 的分类、character sheaf theory 和 finite group character 关系均为外部输入。

## 练习

**练习 6.1.** 对 $GL_n$，证明 nilpotent element 的 Jordan type 在 conjugation 下不变，并说明每个 partition 给出一个 orbit。

**练习 6.2.** 令 $x\in\mathcal N$。证明若 $G_x$ 连通，则 $\mathcal O_x$ 上的 irreducible $G$-equivariant local system 只有平凡一个。

**练习 6.3.** 写出 $P=LU$ 时 projection $P\to L$ 如何进入 character sheaf induction correspondence。
