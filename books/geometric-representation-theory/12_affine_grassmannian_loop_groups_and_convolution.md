# 第十二章：Affine Grassmannian、loop groups 与 convolution

## 本章目标

本章进入仿射几何表示论，定义 loop group、arc group、affine Grassmannian 和其 Schubert stratification，并构造 spherical convolution。它是 geometric Satake 的几何基础。

## 依赖前置知识

需要第一章的 reductive group、第三章的 equivariant sheaves 和附录 F 的卷积模板。

## 12.1 Loop group 和 arc group

**定义 12.1.** 对 $k$-代数 $R$，定义
$$
LG(R)=G(R((z))),\qquad L^+G(R)=G(R[[z]]).
$$
$LG$ 称为 loop group，$L^+G$ 称为 positive loop group 或 arc group。

**定义 12.2.** affine Grassmannian 的 functor of points 为
$$
\operatorname{Gr}_G(R)=LG(R)/L^+G(R)
$$
的 sheafification。等价地，它参数化形式圆盘上 $G$-bundle 连同 punctured disk 上平凡化的数据。

**外部输入定理 12.3.** 若 $G$ reductive，则 $\operatorname{Gr}_G$ 可表示为 ind-projective ind-scheme。  
来源：Beauville-Laszlo、Faltings、Zhu 等。

**例 12.4.** 对 $G=GL_n$，$\operatorname{Gr}_G$ 的 $k$-点可解释为 $k((z))^n$ 中的 lattices，即 $k[[z]]$-submodules $L$ 满足
$$
z^N k[[z]]^n\subset L\subset z^{-N}k[[z]]^n
$$
对某个 $N$ 成立。

**命题 12.5.** 对 $G=GL_1$，有
$$
\operatorname{Gr}_{GL_1}(k)\simeq\mathbb Z.
$$

**证明.** 任一 Laurent series $f\in k((z))^\times$ 可唯一写成
$$
f=z^m u,\qquad m\in\mathbb Z,\quad u\in k[[z]]^\times.
$$
因此
$$
k((z))^\times/k[[z]]^\times\simeq\mathbb Z,
$$
商映射为 valuation。$\square$

## 12.2 Orbit stratification

**定义 12.6.** dominant coweights 集合记为 $X_\ast(T)^+$。对 $\lambda\in X_\ast(T)^+$，令 $z^\lambda\in G(k((z)))$ 为对应 loop。定义 Schubert cell
$$
\operatorname{Gr}^\lambda=L^+G\cdot z^\lambda
$$
和 Schubert variety
$$
\overline{\operatorname{Gr}}^\lambda=\overline{\operatorname{Gr}^\lambda}.
$$

**外部输入定理 12.7.** $L^+G$-orbits on $\operatorname{Gr}_G$ 由 dominant coweights 参数化，且
$$
\operatorname{Gr}_G=\coprod_{\lambda\in X_\ast(T)^+}\operatorname{Gr}^\lambda.
$$
每个 $\overline{\operatorname{Gr}}^\lambda$ 是 projective variety，闭包关系由 dominance order 控制。

**定义 12.8.** spherical Satake category 的初步版本为
$$
\operatorname{Sat}_G=\operatorname{Perv}_{L^+G}(\operatorname{Gr}_G,E),
$$
即 $L^+G$-equivariant perverse sheaves，支撑在有限维 Schubert varieties 上。

**例 12.9.** 对 $G=GL_n$，dominant coweight 可写为
$$
\lambda=(\lambda_1\ge\cdots\ge\lambda_n).
$$
对应 lattice 的相对位置为
$$
L\simeq z^{\lambda_1}k[[z]]e_1\oplus\cdots\oplus z^{\lambda_n}k[[z]]e_n
$$
在合适基下成立。该 Smith normal form 描述是 orbit classification 的具体线性代数版本。

## 12.3 Convolution Grassmannian

**定义 12.10.** convolution Grassmannian 定义为
$$
\operatorname{Gr}_G\widetilde\times\operatorname{Gr}_G
=LG\times^{L^+G}\operatorname{Gr}_G.
$$
有 correspondence
$$
\operatorname{Gr}_G\times\operatorname{Gr}_G
\xleftarrow{\ p\ }
\operatorname{Gr}_G\widetilde\times\operatorname{Gr}_G
\xrightarrow{\ m\ }
\operatorname{Gr}_G,
$$
其中 $m([g,x])=gx$。

**定义 12.11.** 对 $\mathcal F,\mathcal G\in D^b_{L^+G}(\operatorname{Gr}_G,E)$，定义
$$
\mathcal F\star\mathcal G=m_!p^\ast(\mathcal F\boxtimes\mathcal G)
$$
并按需要加入 perversity shift。Satake category 中的 t-exactness 是外部定理，不由定义自动给出。

**命题 12.12.** convolution functor 的结合性由三重 convolution Grassmannian 和 loop group 乘法结合律给出。

**证明.** 三重卷积对象为
$$
LG\times^{L^+G}LG\times^{L^+G}\operatorname{Gr}_G.
$$
两种加括号方式对应先商去第一或第二个中间 $L^+G$ 作用。contracted product 的 associativity 给出两个 ind-schemes 的自然同构；乘法 $LG\times LG\times LG\to LG$ 结合；六函子 base change 给出相应 functor 的自然同构。$\square$

**命题 12.13.** 对 $G=GL_1$，convolution 对应整数加法。

**证明.** 由命题 12.5，$\operatorname{Gr}_{GL_1}$ 的点由 valuation $m\in\mathbb Z$ 标号。loop multiplication 满足
$$
z^m\cdot z^n=z^{m+n}.
$$
因此 convolution correspondence 把分支 $m$ 与分支 $n$ 送到分支 $m+n$。$\square$

## 本章小结

本章定义了 affine Grassmannian、$L^+G$-orbit stratification、Satake category 的初步版本和 convolution product，并补充了 $GL_1$、$GL_n$ lattice 模型和 $GL_1$ 卷积计算。ind-projectivity、一般 orbit classification、perversity t-exactness 和 symmetric monoidal structure 都是 geometric Satake 前的外部输入。

## 练习

**练习 12.1.** 对 $G=GL_1$，计算 $\operatorname{Gr}_G(k)$ 并说明其由 $\mathbb Z$ 标号。

**练习 12.2.** 对 $G=GL_n$，证明 lattice 描述与 $LG/L^+G$ 描述一致。

**练习 12.3.** 写出二重和三重 convolution Grassmannian 的 moduli 解释。
