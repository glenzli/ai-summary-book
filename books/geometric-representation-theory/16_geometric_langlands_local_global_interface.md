# 第十六章：Geometric Langlands 的局部和全局接口

## 本章目标

本章给出 geometric Langlands 的对象语言：曲线上的 $G$-bundles、local systems、Hecke correspondences、Hecke eigensheaves、spectral side 和 automorphic side。

## 依赖前置知识

需要第十三章的 geometric Satake，第十五章的 Kac-Moody/factorization 接口，以及基本代数曲线和 stack 语言。

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

## 16.3 Hecke eigensheaves 和 Langlands functor

**定义 16.10.** 给定 $G^\vee$-local system $E$，一个 Hecke eigensheaf 是 $\mathcal F\in\mathsf{DMod}(\operatorname{Bun}_G)$ 连同对每个 $V\in\operatorname{Rep}(G^\vee)$ 的同构
$$
\mathsf H_V(\mathcal F)\simeq V_E\boxtimes\mathcal F
$$
并满足 tensor compatibility。这里 $V_E$ 是由 $E$ 和 $V$ 诱导的 local system。

**例 16.11.** 对 $G=GL_1$，$\operatorname{Bun}_{GL_1}(C)=\operatorname{Pic}(C)$。Hecke correspondence 在点 $x$ 处把 line bundle $L$ 送到 $L(x)$ 或 $L(-x)$，取决于方向 convention。Hecke eigensheaf 条件变成 Picard stack 上 D-module 对平移的 eigen 条件。这是 abelian geometric class field theory 的几何形式。

**外部输入定理 16.12.** 2024 geometric Langlands proof series 构造并研究从 automorphic side 到 spectral side 的 Langlands functor，并证明多个版本的 geometric Langlands conjecture 在特征 $0$ 的 de Rham/Betti setting 中的等价和核心结论。

**边界说明 16.13.** 当前书稿不把 GLC proof series 写成教材定理链。所需基础包括 derived algebraic geometry、renormalized D-modules、ind-coherent sheaves、factorization categories、Kac-Moody localization 和 singular support formalism，均需独立 locator。

**检查表 16.14.** 任一 geometric Langlands 陈述必须说明：

1. Betti、de Rham 还是 l-adic 版本；
2. automorphic side 使用 D-modules、constructible sheaves 还是 IndCoh；
3. spectral side 使用 QCoh 还是 IndCoh；
4. 是否施加 nilpotent singular support；
5. $G$ 是否 reductive、semisimple、adjoint 或 simply connected；
6. 曲线是否 smooth、proper、带标点或带 level structure。

## 本章小结

本章定义 geometric Langlands 的 automorphic side、spectral side、Hecke correspondence 和 eigensheaf 条件，并补充 $GL_n$ Hecke modification、$GL_1$ abelian 情形和版本检查表。2024 proof series 只作为研究边界入口。

## 练习

**练习 16.1.** 对 $G=GL_1$，解释 $\operatorname{Bun}_G(C)$ 与 Picard stack 的关系。

**练习 16.2.** 写出 Hecke modification 在 $GL_n$ 情形中对 vector bundles 的含义。

**练习 16.3.** 说明 Hecke eigensheaf 定义中 tensor compatibility 为什么不能省略。

**练习 16.4.** 对 $GL_2$ 和 coweight $(1,0)$，用 lattice 描述 Hecke correspondence 的 fiber。
