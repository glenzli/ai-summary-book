# 第五章：Springer resolution、Steinberg variety 与 Weyl group action

## 本章目标

本章构造 Springer resolution、Springer fibers、Steinberg variety 和 Borel-Moore homology convolution。目标是把 Weyl group 表示从 flag variety 和 nilpotent cone 的几何中产生出来，并明确哪些部分是内部构造，哪些部分是 Springer correspondence 的外部输入。

## 依赖前置知识

需要第一章的 flag variety 和 Borel 数据，第三章的 sheaf theory，以及附录 A 的 correspondence 和 convolution 模板。

## 5.1 Nilpotent cone

**约定 5.1.** 本章取 $k=\mathbb C$，$G$ 为连通复 reductive group。为避免中心造成的无关技术问题，若涉及 Killing form 识别，则默认先在 semisimple derived Lie algebra 上工作；一般 reductive 情形通过中心因子分离处理。

**定义 5.2.** nilpotent cone 定义为
$$
\mathcal N=\{x\in\mathfrak g\mid \operatorname{ad}(x)\text{ 是 nilpotent endomorphism of }\mathfrak g\}.
$$
它是 $\mathfrak g$ 的 closed $G$-stable subvariety，其中 $G$ 通过 adjoint action 作用。

**命题 5.3.** $\mathcal N$ 是 $G$-stable。

**证明.** 对 $g\in G$ 和 $x\in\mathfrak g$，
$$
\operatorname{ad}(\operatorname{Ad}_g x)
=\operatorname{Ad}_g\circ\operatorname{ad}(x)\circ\operatorname{Ad}_{g^{-1}}
$$
作为 $\mathfrak g$ 上的线性算子。若 $\operatorname{ad}(x)^N=0$，则上式的 $N$ 次幂也为零。因此 $\operatorname{Ad}_g x$ 仍 nilpotent。$\square$

**外部输入定理 5.4.** $\mathcal N$ 有有限多个 $G$-orbits；这些 orbits 称为 nilpotent orbits。  
来源：Borel、Springer 或 Collingwood-McGovern。后续第六章会系统使用。

## 5.2 Springer resolution

**定义 5.5.** 设 $\mathfrak n=\operatorname{Lie}(R_u(B))$。Springer resolution 定义为
$$
\widetilde{\mathcal N}=G\times^B \mathfrak n.
$$
其点也可写为
$$
\widetilde{\mathcal N}
=\{(x,\mathfrak b')\in\mathcal N\times\mathcal B\mid x\in[\mathfrak b',\mathfrak b']\}.
$$
Springer map 为
$$
\mu:\widetilde{\mathcal N}\to\mathcal N,\qquad [g,x]\mapsto\operatorname{Ad}_g x.
$$

**命题 5.6.** Springer map $\mu$ well-defined 且 $G$-equivariant。

**证明.** 在 $G\times^B\mathfrak n$ 中，
$$
(g,x)\cdot b=(gb,\operatorname{Ad}_{b^{-1}}x).
$$
映射 $[g,x]\mapsto\operatorname{Ad}_g x$ 对该等价关系不变，因为
$$
\operatorname{Ad}_{gb}(\operatorname{Ad}_{b^{-1}}x)=\operatorname{Ad}_g x.
$$
$G$-equivariance 来自左乘：
$$
\mu(h\cdot[g,x])=\mu([hg,x])=\operatorname{Ad}_{hg}x
=\operatorname{Ad}_h(\operatorname{Ad}_g x).
$$
$\square$

**命题 5.7.** $\mu$ 是 proper morphism。

**证明.** 将 $\widetilde{\mathcal N}$ 嵌入 $\mathcal N\times\mathcal B$：
$$
[g,x]\mapsto(\operatorname{Ad}_g x,gB).
$$
其像由条件 $x\in\operatorname{Ad}_g\mathfrak n$ 描述，是闭条件，因为它是关联向量子丛 $G\times^B\mathfrak n$ 在平凡向量丛 $\mathcal B\times\mathfrak g$ 中的闭子丛。投影
$$
\mathcal N\times\mathcal B\to\mathcal N
$$
proper，因为 $\mathcal B$ projective。$\mu$ 是该 proper map 在闭子簇上的限制，因此 proper。$\square$

**外部输入定理 5.8.** 若 $G$ semisimple，则在 invariant nondegenerate form 识别下，
$$
\widetilde{\mathcal N}\simeq T^\ast\mathcal B.
$$
更具体地，$T^\ast(G/B)\simeq G\times^B(\mathfrak g/\mathfrak b)^\ast$，而 $(\mathfrak g/\mathfrak b)^\ast\simeq\mathfrak n$。  
来源：标准 flag variety cotangent bundle 计算。

**定义 5.9.** 对 $x\in\mathcal N$，Springer fiber 定义为
$$
\mathcal B_x=\mu^{-1}(x)
=\{\mathfrak b'\in\mathcal B\mid x\in[\mathfrak b',\mathfrak b']\}.
$$

**例 5.10.** 若 $G=SL_2$，则 $\mathcal N$ 有两个 nilpotent orbits：$0$ 和 regular nilpotent orbit。对 $x=0$，
$$
\mathcal B_0=\mathcal B\simeq\mathbb P^1.
$$
对非零 nilpotent $x$，$\mathcal B_x$ 是一个点，即唯一包含 $x$ 的 Borel 的 nilradical。

后一项唯一性可用 $SL_2$ 的矩阵模型直接检查：非零 nilpotent $x$ 有一维 kernel，保持该 kernel 的 Borel 是唯一使 $x$ 成为上三角严格幂零矩阵的 Borel。

## 5.3 Steinberg variety 和卷积

**定义 5.11.** Steinberg variety 定义为 fiber product
$$
Z=\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}.
$$
等价地，
$$
Z=\{(x,\mathfrak b_1,\mathfrak b_2)\mid
x\in[\mathfrak b_1,\mathfrak b_1]\cap[\mathfrak b_2,\mathfrak b_2]\}.
$$

**定义 5.12.** 令 $H_\ast^{BM}(-)$ 表示 Borel-Moore homology。Steinberg variety 上的 convolution 定义如下。对
$$
Z_{12}=\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N},\quad
Z_{23}=\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N},
$$
取三重 fiber product
$$
Z_{123}=\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}
\times_{\mathcal N}\widetilde{\mathcal N}.
$$
设 $p_{ij}:Z_{123}\to Z$ 为投影。对 $a,b\in H_\ast^{BM}(Z)$，定义
$$
a\star b=(p_{13})_\ast(p_{12}^\ast a\cap p_{23}^\ast b),
$$
其中交叉积和 cap product 按 Borel-Moore homology 的标准 formalism 理解。

**命题 5.13.** 在 Borel-Moore homology 的 proper pull-push formalism 下，$\star$ 是结合乘法。

**证明.** 四重 fiber product
$$
\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}
\times_{\mathcal N}\widetilde{\mathcal N}\times_{\mathcal N}\widetilde{\mathcal N}
$$
同时控制 $(a\star b)\star c$ 和 $a\star(b\star c)$。两种计算都是先拉回三个 cycle 类到四重 fiber product，取交，再沿第一和第四因子投影推前。proper base change 和 projection formula 保证中间投影次序不影响所得类。因此乘法结合。$\square$

**外部输入定理 5.14.** Steinberg variety 的 top Borel-Moore homology 在 convolution 下同构于 group algebra：
$$
H_{\operatorname{top}}^{BM}(Z,\mathbb C)\simeq\mathbb C[W].
$$
更强的等变 K-theory 或 equivariant Borel-Moore homology 版本给出 affine Hecke algebra 或 graded Hecke algebra。  
来源：Kazhdan-Lusztig、Chriss-Ginzburg、Ginzburg。

## 5.4 Springer sheaf 和 Weyl group action

**定义 5.15.** Springer sheaf 定义为
$$
\mathsf{Spr}=\mu_\ast E_{\widetilde{\mathcal N}}[\dim\mathcal N]
$$
在 $D^b_G(\mathcal N,E)$ 中的对象。由于 $\mu$ proper，也可写为 $\mu_!$。

**外部输入定理 5.16.** Springer sheaf $\mathsf{Spr}$ 带有自然 $W$-作用，并且对 $x\in\mathcal N$，该作用在 Springer fiber cohomology $H^\ast(\mathcal B_x,E)$ 上给出 Springer representation。  
来源：Springer、Borho-MacPherson、Kazhdan-Lusztig、Chriss-Ginzburg。

**外部输入定理 5.17.** Springer correspondence 给出 $W$ 的不可约表示与若干对 $(\mathcal O,\mathcal L)$ 的对应，其中 $\mathcal O\subset\mathcal N$ 是 nilpotent orbit，$\mathcal L$ 是 $\mathcal O$ 上的 irreducible $G$-equivariant local system。  
限制：具体归一化和 sign representation convention 必须在第六章和附录 D 中锁定。

## 本章小结

本章内部构造了 Springer resolution、证明 Springer map proper、定义了 Springer fibers、Steinberg variety 和 convolution。Weyl group action、top Borel-Moore homology 与 $\mathbb C[W]$ 的同构、Springer correspondence 均是核心外部输入。

## 练习

**练习 5.1.** 对 $G=GL_n$，证明 nilpotent orbits 由 $n$ 的 partitions 标号。

**练习 5.2.** 对 $G=SL_2$，显式写出 $\widetilde{\mathcal N}\to\mathcal N$ 并计算两个 Springer fibers 的 cohomology。

**练习 5.3.** 写出 Steinberg variety $Z$ 到 $\mathcal B\times\mathcal B$ 的投影，并说明其像与 $G$ 在 $\mathcal B\times\mathcal B$ 上的 orbits 的关系。

