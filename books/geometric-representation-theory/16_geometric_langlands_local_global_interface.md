# 第十六章：Geometric Langlands 的局部和全局接口

在曲线的一点修改 $G$-bundle，formal disk 上看到的局部模型正是 affine Grassmannian；Geometric Satake 因而把每个 $G^\vee$-表示变成作用在 $\operatorname{Bun}_G(C)$ 上的 Hecke functor。若一个 automorphic 对象同时是所有这些 functors 的本征对象，本征值应由一条 $G^\vee$-local system 给出。这个想法把 automorphic stack 与 spectral derived stack 联系起来，却不能简化为点集之间的对应：两侧都需要导出或重整化范畴，Hecke 同构还必须保持 tensor compatibility。$GL_n$ 的 elementary modification 说明局部 coweight 如何改变向量丛，$GL_1$ 的 Picard stack 则给出可直接理解的 abelian 原型。

## 16.1 两侧范畴

**约定 16.1.** 本章取 $C$ 为光滑 projective complex curve，$G$ 为 connected reductive group，$G^\vee$ 为 Langlands dual group。

**定义 16.2.** automorphic side 的基本几何对象为 moduli stack
$$
\operatorname{Bun}_G(C)
$$
of $G$-bundles on $C$。其 sheaf 或 D-module category 记为
$$
\mathsf{DMod}(\operatorname{Bun}_G).
$$

**定义 16.3.** spectral side 的基本对象为 $G^\vee$-local systems 的 derived stack
$$
\operatorname{LocSys}_{G^\vee}(C).
$$
谱侧 category 常用 quasi-coherent sheaves、ind-coherent sheaves 或 nilpotent singular support 条件下的子范畴。

**警告 16.4.** $\operatorname{LocSys}_{G^\vee}$ 必须作为 derived stack 处理，经典截断会丢失 deformation complex。geometric Langlands 的现代陈述不是普通集合上的函数对应。

两侧范畴的对象类型确定后，真正连接它们的是逐点 Hecke correspondence。其 fiber 由第十二章的 affine Grassmannian 控制，因此 relative position 与 Satake IC 层都可在全局曲线上复用。

## 16.2 Hecke correspondence

**定义 16.5.** 对点 $x\in C$ 和 dominant coweight $\lambda$，Hecke correspondence $\mathsf H_{\lambda,x}$ 参数化三元组
$$
(\mathcal P,\mathcal P',\varphi)
$$
其中 $\mathcal P,\mathcal P'$ 是 $G$-bundles，$\varphi$ 是 $C\setminus\{x\}$ 上的同构，且相对位置不超过 $\lambda$。有 correspondence
$$
\operatorname{Bun}_G
\xleftarrow{h_1}
\mathsf H_{\lambda,x}
\xrightarrow{h_2}
\operatorname{Bun}_G.
$$

**定义 16.6.** Hecke functor 形式上定义为
$$
\mathsf H_{\lambda,x}(\mathcal F)=h_{2!}h_1^\ast\mathcal F[\text{shift}],
$$
具体使用 $!$ 还是 $\ast$、是否 renormalized，以及 shift 如何确定，依赖所选 D-module/category 模型。

**外部输入定理 16.7.** Geometric Satake 把 Hecke functors 的 tensor structure 与 $G^\vee$ 的 representations 相连。

**例 16.8.** 对 $G=GL_n$，$G$-bundle 就是 rank $n$ vector bundle。Hecke modification at $x$ 是两个 vector bundles $E,E'$ 以及同构
$$
E|_{C\setminus\{x\}}\simeq E'|_{C\setminus\{x\}}.
$$
相对位置由两个 lattice
$$
E_x\otimes\mathcal K_x,\qquad E'_x\otimes\mathcal K_x
$$
之间的 elementary divisors 给出。dominant coweight $\lambda=(\lambda_1\ge\cdots\ge\lambda_n)$ 记录 quotient 的长度分布。

**命题 16.9.** 对 $GL_n$，minuscule coweight $\lambda=(1,0,\ldots,0)$ 的 Hecke modification 等价于在点 $x$ 处选择一维 quotient 或 hyperplane modification，取决于方向 convention。

**证明.** 局部化到 formal disk。两个 lattice $L,L'\subset\mathcal K_x^n$ 相对位置为 $(1,0,\ldots,0)$ 意味着
$$
L\subset L'\subset z^{-1}L
$$
且 $L'/L$ 长度为 $1$。这样的中间 lattice 等价于选择 $z^{-1}L/L\simeq L/zL$ 中的一维子空间；对偶 convention 下等价于选择 $L/zL$ 的一维 quotient。全局上只在点 $x$ 修改，故得到 elementary Hecke modification。$\square$

这个计算把抽象 correspondence 的 fiber 化成射影空间中的 line choice。Geometric Satake 进一步要求不同表示对应的 Hecke functors 按 tensor product 组合，这正是 eigensheaf 定义中相容条件不可省略的原因。

## 16.3 Hecke eigensheaves 和 Langlands functor

**定义 16.10.** 给定 $G^\vee$-local system $E$，一个 Hecke eigensheaf 是 $\mathcal F\in\mathsf{DMod}(\operatorname{Bun}_G)$ 连同对每个 $V\in\operatorname{Rep}(G^\vee)$ 的同构
$$
\mathsf H_V(\mathcal F)\simeq V_E\boxtimes\mathcal F
$$
并满足 tensor compatibility。这里 $V_E$ 是由 $E$ 和 $V$ 诱导的 local system。

**例 16.11.** 对 $G=GL_1$，$\operatorname{Bun}_{GL_1}(C)=\operatorname{Pic}(C)$。Hecke correspondence 在点 $x$ 处把 line bundle $L$ 送到 $L(x)$ 或 $L(-x)$，取决于方向 convention。Hecke eigensheaf 条件变成 Picard stack 上 D-module 对平移的 eigen 条件。这是 abelian geometric class field theory 的几何形式。

**研究边界 16.12.** Geometric Langlands proof series 构造从 automorphic side 到 spectral side 的 Langlands functor，并在精确限定的特征零 de Rham/Betti 模型中证明相应结果。由于不同篇章使用的群、范畴与 singular-support 假设并不由这里的简写唯一确定，本书不把“完整 geometric Langlands 等价”作为可调用的单一定理。

**边界说明 16.13.** 要陈述某个确定版本，至少还需 derived algebraic geometry、renormalized D-modules、ind-coherent sheaves、factorization categories、Kac--Moody localization 和 singular-support formalism。缺少其中所用模型时，只保留 Hecke correspondence 与 eigensheaf 的定义接口。

**版本条件 16.14.** 任一 geometric Langlands 陈述必须说明：

1. Betti、de Rham 还是 l-adic 版本；
2. automorphic side 使用 D-modules、constructible sheaves 还是 IndCoh；
3. spectral side 使用 QCoh 还是 IndCoh；
4. 是否施加 nilpotent singular support；
5. $G$ 是否 reductive、semisimple、adjoint 或 simply connected；
6. 曲线是否 smooth、proper、带标点或带 level structure。

$GL_n$ 的 lattice modification 把局部 Hecke fiber 具体化，$GL_1$ 的 Picard 平移则显示本征条件在交换群情形中的含义。一般 $G$ 的范畴等价还受 derived structure、singular support 与 renormalization 控制，因此只在明确版本下成立。此后六章转向辛几何和范畴化：quiver variety 将首先展示 Hamiltonian reduction 如何直接构造 Kac--Moody 表示。

## 练习

**练习 16.1.** 对 $G=GL_1$，解释 $\operatorname{Bun}_G(C)$ 与 Picard stack 的关系。

**练习 16.2.** 写出 Hecke modification 在 $GL_n$ 情形中对 vector bundles 的含义。

**练习 16.3.** 说明 Hecke eigensheaf 定义中 tensor compatibility 为什么不能省略。

**练习 16.4.** 对 $GL_2$ 和 coweight $(1,0)$，用 lattice 描述 Hecke correspondence 的 fiber。
