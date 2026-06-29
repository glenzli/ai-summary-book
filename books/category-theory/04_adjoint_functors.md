# 第四章：伴随函子

## 本章目标

本章定义伴随函子，建立 Hom 自然同构、单位-余单位和泛箭头三种等价语言，并证明左伴随保持余极限、右伴随保持极限。

## 依赖前置知识

需要自然变换、函子范畴、极限和余极限。

## 4.1 伴随的定义

**定义 4.1.** 设 $F:\mathcal C\to\mathcal D$ 与 $G:\mathcal D\to\mathcal C$ 为函子。称 $F$ 左伴随于 $G$，记作 $F\dashv G$，若存在对 $X\in\mathcal C$ 与 $Y\in\mathcal D$ 自然的双射

$$
\Phi_{X,Y}:\mathcal D(FX,Y)\xrightarrow{\cong}\mathcal C(X,GY).
$$

此时 $G$ 称为 $F$ 的右伴随。

**例子 4.2.** 自由群函子

$$
F:\mathbf{Set}\to\mathbf{Grp}
$$

左伴随于忘却函子 $U:\mathbf{Grp}\to\mathbf{Set}$。伴随双射为

$$
\mathbf{Grp}(F(S),G)\cong\mathbf{Set}(S,U(G)),
$$

它表达自由群由集合映射 $S\to U(G)$ 唯一延拓为群同态 $F(S)\to G$。

## 4.2 单位、余单位与三角恒等式

**定义 4.3.** 若 $F\dashv G$，单位（unit）为自然变换

$$
\eta:\operatorname{id}_{\mathcal C}\Rightarrow GF
$$

其分量 $\eta_X:X\to GFX$ 是 $\operatorname{id}_{FX}$ 在伴随双射下的像。余单位（counit）为自然变换

$$
\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}
$$

其分量 $\varepsilon_Y:FGY\to Y$ 是 $\operatorname{id}_{GY}$ 在逆伴随双射下的像。

**定理 4.4.** 给出伴随 $F\dashv G$ 等价于给出自然变换

$$
\eta:\operatorname{id}_{\mathcal C}\Rightarrow GF,\qquad
\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}
$$

满足三角恒等式

$$
\varepsilon_{F X}\circ F(\eta_X)=\operatorname{id}_{F X},
\qquad
G(\varepsilon_Y)\circ\eta_{G Y}=\operatorname{id}_{G Y}.
$$

**证明.** 从 Hom 自然同构构造 $\eta,\varepsilon$ 如定义 4.3。自然性给出任意 $f:FX\to Y$ 与 $g:X\to GY$ 的转置公式

$$
\Phi(f)=G(f)\circ\eta_X,
\qquad
\Phi^{-1}(g)=\varepsilon_Y\circ F(g).
$$

把 $f=\operatorname{id}_{FX}$ 与 $g=\operatorname{id}_{GY}$ 代入并要求两次转置回到原态射，得到两个三角恒等式。

反过来，若有 $\eta,\varepsilon$ 满足三角恒等式，定义

$$
\Phi_{X,Y}(f)=G(f)\circ\eta_X,
\qquad
\Psi_{X,Y}(g)=\varepsilon_Y\circ F(g).
$$

计算得

$$
\Psi(\Phi(f))
=\varepsilon_Y\circ F(G(f)\circ\eta_X)
=\varepsilon_Y\circ FG(f)\circ F(\eta_X)
=f\circ\varepsilon_{FX}\circ F(\eta_X)
=f,
$$

其中第三个等号用 $\varepsilon$ 的自然性，最后一个等号用三角恒等式。类似地 $\Phi(\Psi(g))=g$。自然性由 $\eta,\varepsilon$ 的自然性验证。$\square$

## 4.3 泛箭头

**定义 4.5.** 给定 $G:\mathcal D\to\mathcal C$ 和对象 $X\in\mathcal C$，从 $X$ 到 $G$ 的泛箭头是对象 $Y\in\mathcal D$ 与态射 $\eta:X\to GY$，使得对任意 $Z\in\mathcal D$ 和任意 $f:X\to GZ$，存在唯一 $\bar f:Y\to Z$ 满足

$$
G(\bar f)\circ\eta=f.
$$

**命题 4.6.** 函子 $G:\mathcal D\to\mathcal C$ 有左伴随，当且仅当对每个 $X\in\mathcal C$ 都存在从 $X$ 到 $G$ 的泛箭头，并且这些泛箭头可函子化。

**证明.** 若 $F\dashv G$，令 $\eta_X:X\to GFX$ 为伴随单位。给定 $Z\in\mathcal D$ 和 $f:X\to GZ$，令 $\bar f:FX\to Z$ 为 $f$ 在伴随双射

$$
\mathcal D(FX,Z)\cong\mathcal C(X,GZ)
$$

下的逆像。转置公式给出

$$
G(\bar f)\circ\eta_X=f.
$$

若 $h:FX\to Z$ 也满足 $G(h)\eta_X=f$，则 $h$ 与 $\bar f$ 在伴随双射下有同一像，故 $h=\bar f$。所以 $\eta_X$ 是从 $X$ 到 $G$ 的泛箭头。

反过来，假设对每个 $X$ 选择泛箭头

$$
\eta_X:X\to G(FX).
$$

对态射 $u:X\to X'$，由 $\eta_X$ 的泛性应用于复合

$$
X\xrightarrow{u}X'\xrightarrow{\eta_{X'}}G(FX')
$$

得到唯一态射 $F(u):FX\to FX'$，满足

$$
G(F(u))\eta_X=\eta_{X'}u.
$$

若 $u=\operatorname{id}_X$，则 $F(u)$ 与 $\operatorname{id}_{FX}$ 都满足同一等式，故相等。若 $X\xrightarrow{u}X'\xrightarrow{v}X''$，则 $F(v)F(u)$ 与 $F(vu)$ 都满足

$$
G(-)\eta_X=\eta_{X''}vu,
$$

故由唯一性相等。因此 $F$ 是函子，且 $\eta$ 是自然变换 $\operatorname{id}_{\mathcal C}\Rightarrow GF$。

现在定义

$$
\Phi_{X,Z}:\mathcal D(FX,Z)\to\mathcal C(X,GZ),
\qquad
h\mapsto G(h)\eta_X.
$$

泛箭头性质说明 $\Phi_{X,Z}$ 对每个 $X,Z$ 是双射。对 $u:X'\to X$ 和 $k:Z\to Z'$，等式

$$
G(kh)\eta_X=G(k)G(h)\eta_X,\qquad
G(hF(u))\eta_{X'}=G(h)\eta_Xu
$$

分别给出对 $Z$ 和 $X$ 的自然性。所以 $F\dashv G$。$\square$

## 4.4 伴随保持极限

**定理 4.7.** 左伴随保持所有存在的余极限；右伴随保持所有存在的极限。

**证明.** 证明左伴随情形。设 $F:\mathcal C\to\mathcal D$ 左伴随于 $G$，并设 $L=\operatorname{colim}_{j\in\mathcal J}D(j)$。对任意 $Y\in\mathcal D$，有自然双射

$$
\mathcal D(F L,Y)\cong\mathcal C(L,GY).
$$

由于 $L$ 是余极限，

$$
\mathcal C(L,GY)\cong
\lim_{j\in\mathcal J^{\operatorname{op}}}\mathcal C(D(j),GY).
$$

再用伴随，

$$
\mathcal C(D(j),GY)\cong\mathcal D(FD(j),Y).
$$

合并得到

$$
\mathcal D(F L,Y)\cong
\lim_{j\in\mathcal J^{\operatorname{op}}}\mathcal D(FD(j),Y),
$$

这正是 $F L$ 作为图形 $F D$ 的余极限的表示性条件。右伴随保持极限由对偶论证得到。$\square$

## 4.5 伴随的唯一性

**命题 4.8.** 若 $G:\mathcal D\to\mathcal C$ 有两个左伴随 $F$ 与 $F'$，则存在唯一自然同构 $F\cong F'$ 与相应伴随结构相容。

**证明.** 对每个 $X$，对象 $F X$ 与 $F'X$ 都表示函子

$$
Y\longmapsto \mathcal C(X,GY).
$$

具体地，伴随给出自然同构

$$
\mathcal D(FX,Y)\cong\mathcal C(X,GY)\cong\mathcal D(F'X,Y)
$$

对 $Y$ 自然。由 Yoneda 引理，存在唯一同构

$$
\theta_X:F'X\to FX
$$

诱导上述自然同构。若 $u:X\to X'$，则 $F(u)\theta_X$ 与 $\theta_{X'}F'(u)$ 是 $F'X\to FX'$ 的两个态射。对任意 $Y$，它们在表示函子同构下诱导同一个自然变换，因为二者都对应于预复合 $u$ 对函子 $Y\mapsto\mathcal C(X',GY)$ 的作用。由 Yoneda 的忠实性，

$$
F(u)\theta_X=\theta_{X'}F'(u).
$$

故 $\theta:F'\Rightarrow F$ 是自然同构。若另有相容自然同构，则其每个分量诱导同一个表示同构，仍由 Yoneda 唯一性相等。$\square$

## 4.6 本章小结

伴随是范畴论中最重要的结构之一。它可以写成 Hom 集自然双射，也可以写成单位、余单位和三角恒等式。左伴随保持余极限，右伴随保持极限，这一事实解释了自由构造、遗忘构造和许多几何/代数反向函子的形式性质。

## 练习

**练习 4.1.** 证明自由阿贝尔群函子 $\mathbb Z[-]:\mathbf{Set}\to\mathbf{Ab}$ 左伴随于忘却函子。

**练习 4.2.** 写出积函子 $A\times -:\mathbf{Set}\to\mathbf{Set}$ 的右伴随，并证明伴随双射。

**练习 4.3.** 从单位和余单位出发，证明右伴随保持终对象。

**练习 4.4.** 设 $F\dashv G$。证明 $F$ 完全忠实当且仅当单位 $\eta:\operatorname{id}\Rightarrow GF$ 是自然同构。

**练习 4.5.** 对偶化定理 4.7，写出右伴随保持极限的完整证明。
