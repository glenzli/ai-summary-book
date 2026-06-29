# 第十八章：曲线、`G`-Bundles 和 Hecke 修改

## 本章目标

本章进入几何 Langlands。数论 Langlands 中的基本商是
$$
G(K)\backslash G(\mathbb A_K),
$$
而几何 Langlands 中的基本对象是光滑射影曲线 $X$ 上的 $G$-bundle 模栈
$$
\operatorname{Bun}_G(X).
$$
Hecke 算子不再只是 Hecke 代数中的卷积算子，而是由 Hecke correspondence 给出的函子。本章建立曲线、$G$-bundles、Hecke 修改和 Hecke 栈的基本语言，为几何 Satake 和 Hecke eigensheaves 做准备。

## 依赖前置知识

需要代数几何中的光滑射影曲线、主丛、代数栈、纤维积和层范畴。需要第十一章的还原群和对偶群。几何 Langlands 的完整理论涉及 derived algebraic geometry、D-modules、$\ell$-adic sheaves、perverse sheaves 和 factorization structures；本章只建立基础几何对象。

收口归一化回指：本章连接函数域 adeles 与曲线几何时采用第二十二章和 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 9 节的 sheaf-function convention。

## 18.1 曲线与函数域类比

设 $k$ 为代数闭域，$X/k$ 为光滑射影连通曲线。其函数域记为
$$
K_X=k(X).
$$

**定义 18.1.** 对闭点 $x\in X$，记完成局部环为
$$
\mathcal O_x=\widehat{\mathcal O}_{X,x},
$$
分式域为
$$
K_x=\operatorname{Frac}(\mathcal O_x).
$$
形式圆盘记为
$$
D_x=\operatorname{Spec}\mathcal O_x,
$$
穿孔形式圆盘记为
$$
D_x^\times=\operatorname{Spec}K_x.
$$

**注 18.2.** 若 $k=\mathbb F_q$，则 $K_X$ 是函数域，第一章的 adeles 可由所有 $K_x$ 的 restricted product 构造。几何 Langlands 把函数域上的 adelic 商改写为 $X$ 上的 $G$-bundles 模栈。

## 18.2 主 `G`-丛

设 $G/k$ 为 connected reductive group。

**定义 18.3.** 一个 $G$-bundle on $X$ 是 $X$ 上的右主 $G$-丛，即一个 $X$-scheme 或 stack $\mathcal P$，带右 $G$-作用，并且在 fppf 拓扑局部同构于
$$
U\times G\to U.
$$

**定义 18.4.** $G$-bundles 的模栈 $\operatorname{Bun}_G(X)$ 是如下 stack：对任意测试 scheme $S$，
$$
\operatorname{Bun}_G(X)(S)
$$
是 $X\times S$ 上的 $G$-bundles groupoid。

**外部输入定理 18.5（$\operatorname{Bun}_G$ 的代数性）.** 若 $G$ 为 smooth affine algebraic group，则 $\operatorname{Bun}_G(X)$ 是代数栈。若 $G$ reductive，则它局部有限型，但通常不是有限型。

**例 18.6.** 当 $G=\operatorname{GL}_n$ 时，$G$-bundle 等价于秩 $n$ 向量丛。于是
$$
\operatorname{Bun}_{\operatorname{GL}_n}(X)
$$
是秩 $n$ 向量丛的模栈。

**例 18.7.** 当 $G=\mathbb G_m$ 时，$G$-bundle 是 line bundle，故
$$
\operatorname{Bun}_{\mathbb G_m}(X)=\operatorname{Pic}(X)
$$
作为 stack 需要记住 line bundle 的 automorphism group $\mathbb G_m$。

## 18.3 Adelic 商与 `Bun_G`

本节设 $k$ 为有限域或代数闭域，并取函数域 $K_X$。

**外部输入定理 18.8（Weil uniformization，接口形式）.** 对适当意义下的 $k$-点，有双商描述
$$
\operatorname{Bun}_G(X)(k)
\simeq
G(K_X)\backslash G(\mathbb A_{K_X})/G(\mathcal O_{\mathbb A}),
$$
其中
$$
G(\mathcal O_{\mathbb A})=\prod_xG(\mathcal O_x).
$$

**注 18.9.** 该公式解释了几何 Langlands 与函数域自守形式的关系：自守函数是 adelic 双商上的函数，而几何 Langlands 用 $\operatorname{Bun}_G$ 上的 sheaves 替代函数。

## 18.4 Hecke 修改

Hecke 算子的几何化来自在一个点处修改 $G$-bundle。

**定义 18.10.** 设 $S$ 为测试 scheme。$X$ 上 $G$-bundle $\mathcal P$ 与 $\mathcal P'$ 在 $x:S\to X$ 处的 Hecke 修改是一个同构
$$
\beta:\mathcal P|_{(X\times S)\setminus\Gamma_x}
\xrightarrow{\sim}
\mathcal P'|_{(X\times S)\setminus\Gamma_x},
$$
其中 $\Gamma_x\subset X\times S$ 为 $x$ 的图像。

**定义 18.11.** Hecke stack $\operatorname{Hecke}_G$ 是如下 stack：其 $S$-点为四元组
$$
(x,\mathcal P,\mathcal P',\beta),
$$
其中 $x:S\to X$，$\mathcal P,\mathcal P'$ 为 $X\times S$ 上的 $G$-bundles，$\beta$ 为定义 18.10 的 Hecke 修改。

它带有自然映射
$$
\operatorname{Bun}_G
\xleftarrow{h^\leftarrow}
\operatorname{Hecke}_G
\xrightarrow{h^\rightarrow}
\operatorname{Bun}_G,
$$
以及位置映射
$$
\operatorname{supp}:\operatorname{Hecke}_G\to X.
$$

**命题 18.12.** Hecke stack 给出 $\operatorname{Bun}_G$ 上的 correspondence。

**证明.** 对象 $(x,\mathcal P,\mathcal P',\beta)$ 有两个 $G$-bundle：修改前的 $\mathcal P$ 和修改后的 $\mathcal P'$。令
$$
h^\leftarrow(x,\mathcal P,\mathcal P',\beta)=\mathcal P,\qquad
h^\rightarrow(x,\mathcal P,\mathcal P',\beta)=\mathcal P'
$$
即得到从 $\operatorname{Hecke}_G$ 到两个 $\operatorname{Bun}_G$ 因子的映射。$\square$

## 18.5 相对位置与 affine Grassmannian

Hecke 修改在点 $x$ 的局部类型由 affine Grassmannian 控制。

**定义 18.13.** 设 $F=k((t))$，$\mathcal O=k[[t]]$。Affine Grassmannian 定义为 fpqc quotient
$$
\operatorname{Gr}_G=G(F)/G(\mathcal O).
$$
更精确地，它是 functor
$$
R\mapsto G(R((t)))/G(R[[t]])
$$
的 sheafification。

**外部输入定理 18.14（Affine Grassmannian 的 Schubert 分解）.** 若 $G$ reductive，则 $\operatorname{Gr}_G$ 是 ind-projective ind-scheme。其 $G(\mathcal O)$-orbits 由 dominant coweights
$$
\lambda\in X_*(T)^+
$$
参数化。

**定义 18.15.** Hecke 修改的相对位置至多为 $\lambda$，若在局部平凡化后，其对应的 affine Grassmannian 点落入 Schubert variety
$$
\overline{\operatorname{Gr}}_G^\lambda.
$$
相对位置受限的 Hecke stack 记为
$$
\operatorname{Hecke}_G^{\le\lambda}.
$$

**例 18.16.** 当 $G=\operatorname{GL}_n$ 时，Hecke 修改等价于两个向量丛 $E,E'$ 在点 $x$ 外同构。相对位置由商 lattice 的 elementary divisors 给出，即由 dominant coweight
$$
\lambda=(\lambda_1\ge\cdots\ge\lambda_n)
$$
记录。

## 18.6 Hecke 函子

设 $\mathcal D(\operatorname{Bun}_G)$ 表示 $\operatorname{Bun}_G$ 上选定的 sheaf theory 的 derived category；可取 $\ell$-adic sheaves 或 D-modules，依 $k$ 的特征而定。

**定义 18.17.** 给定 $\operatorname{Hecke}_G$ 上的 kernel $\mathcal K$，Hecke 函子形式上定义为
$$
\mathsf H_{\mathcal K}(\mathcal F)
=
h^\rightarrow_!\left(h^{\leftarrow,*}\mathcal F\otimes\mathcal K\right).
$$
若保留修改点，则它是函子
$$
\mathcal D(\operatorname{Bun}_G)
\to
\mathcal D(\operatorname{Bun}_G\times X).
$$

**注 18.18.** 完整定义需要选择 $!$-pushforward、$*$-pushforward 或 renormalized pushforward，并处理 $\operatorname{Bun}_G$ 非紧性。几何 Langlands 的技术困难之一正来自这些函子的正确范畴设置。

**定义 18.19.** 对 dominant coweight $\lambda$，由 Schubert variety $\overline{\operatorname{Gr}}_G^\lambda$ 上的交叉上同调 sheaf 得到 Hecke kernel，记为 $\mathcal K_\lambda$。相应 Hecke 函子记为
$$
\mathsf H_\lambda.
$$

## 18.7 本章小结

几何 Langlands 把函数域自守理论几何化：adelic 双商被 $\operatorname{Bun}_G(X)$ 取代，Hecke 算子被 Hecke correspondence 和 Hecke 函子取代，局部 Hecke 代数被 affine Grassmannian 的几何结构取代。下一章的几何 Satake 将说明：$\operatorname{Gr}_G$ 上的 $G(\mathcal O)$-等变 perverse sheaves 的张量范畴等价于对偶群 $\widehat G$ 的表示范畴。

## 练习

**练习 18.1.** 对 $G=\operatorname{GL}_n$，证明 $G$-bundle 等价于秩 $n$ 向量丛。

**练习 18.2.** 解释 $\operatorname{Bun}_{\mathbb G_m}(X)$ 与 Picard stack 的关系。

**练习 18.3.** 对两个向量丛的 Hecke 修改，写出修改点外同构的定义。

**练习 18.4.** 对 $G=\operatorname{GL}_2$，用 lattice 描述相对位置 $\lambda=(1,0)$ 的 Hecke 修改。

**练习 18.5.** 说明 Hecke stack 如何给出 $\operatorname{Bun}_G$ 上的 correspondence。

**练习 18.6.** 解释 affine Grassmannian 的 $G(\mathcal O)$-orbits 为什么由 dominant coweights 参数化。
