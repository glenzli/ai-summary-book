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

## 4.6 全忠实伴随与反射子范畴

**命题 4.9.** 设 $F:\mathcal C\rightleftarrows\mathcal D:G$，且 $F\dashv G$，单位和余单位分别为 $\eta$ 与 $\varepsilon$。

1. $F$ 完全忠实当且仅当 $\eta:\operatorname{id}_{\mathcal C}\Rightarrow GF$ 是自然同构。
2. $G$ 完全忠实当且仅当 $\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}$ 是自然同构。

**证明.** 证明第一条。对任意 $X,Y\in\mathcal C$，伴随双射给出

$$
\mathcal D(FX,FY)\cong\mathcal C(X,GFY).
$$

在该双射下，态射 $F(f):FX\to FY$ 对应于

$$
X\xrightarrow{f}Y\xrightarrow{\eta_Y}GFY.
$$

因此自然映射

$$
\mathcal C(X,Y)\to\mathcal D(FX,FY)
$$

经伴随识别后正是

$$
\mathcal C(X,Y)\xrightarrow{\mathcal C(X,\eta_Y)}\mathcal C(X,GFY).
$$

若 $F$ 完全忠实，则该映射对所有 $X$ 为双射。由 Yoneda 引理，$\eta_Y:Y\to GFY$ 是同构。对所有 $Y$ 如此，故 $\eta$ 是自然同构。

反过来，若 $\eta$ 是自然同构，则任意 $h:FX\to FY$ 对应于

$$
G(h)\eta_X:X\to GFY.
$$

再与 $\eta_Y^{-1}:GFY\to Y$ 复合得到

$$
\eta_Y^{-1}G(h)\eta_X:X\to Y.
$$

这个构造是 $F$ 在 Hom 集上诱导映射的逆映射，故 $F$ 完全忠实。第二条对偶，或把第一条用于反伴随 $G^{op}\dashv F^{op}$。$\square$

**定义 4.10.** 设 $\mathcal A$ 是 $\mathcal C$ 的全子范畴，包含函子为

$$
I:\mathcal A\hookrightarrow\mathcal C.
$$

若 $I$ 有左伴随 $L:\mathcal C\to\mathcal A$，则称 $\mathcal A$ 是 $\mathcal C$ 的反射子范畴（reflective subcategory），$L$ 称为反射函子，单位

$$
\eta_X:X\to I L X
$$

称为 $X$ 到 $\mathcal A$ 的反射。若 $I$ 有右伴随，则称 $\mathcal A$ 为余反射子范畴（coreflective subcategory）。

**命题 4.11.** 对全子范畴 $I:\mathcal A\hookrightarrow\mathcal C$，给出左伴随 $L\dashv I$ 等价于对每个 $X\in\mathcal C$ 给出对象 $LX\in\mathcal A$ 与态射

$$
\eta_X:X\to I L X
$$

使得对任意 $A\in\mathcal A$，预复合 $\eta_X$ 给出双射

$$
\mathcal A(LX,A)\cong \mathcal C(X,I A).
$$

在此情形下，余单位

$$
L I A\to A
$$

是同构。

**证明.** 若 $L\dashv I$，伴随双射正给出上述双射，而 $\eta_X$ 是单位。由于 $I$ 完全忠实，命题 4.9 的对偶形式说明余单位 $LIA\to A$ 是同构。

反过来，若对每个 $X$ 已给出上述泛性质，则对态射 $u:X\to Y$，把复合

$$
X\xrightarrow{u}Y\xrightarrow{\eta_Y}I L Y
$$

代入 $X$ 的泛性质，得到唯一态射

$$
L u:LX\to LY
$$

满足

$$
I(Lu)\eta_X=\eta_Yu.
$$

恒等和复合由唯一性验证，故 $L$ 是函子。上述双射对 $X$ 与 $A$ 自然，因而给出伴随 $L\dashv I$。$\square$

## 4.7 反射子范畴中的余极限

**命题 4.12.** 设 $\mathcal A\subseteq\mathcal C$ 是反射子范畴，反射函子为 $L:\mathcal C\to\mathcal A$，包含为 $I:\mathcal A\hookrightarrow\mathcal C$。令 $D:\mathcal J\to\mathcal A$ 为图形。若 $\mathcal C$ 中存在余极限

$$
C=\operatorname{colim}_{j\in\mathcal J}I D(j),
$$

则 $\mathcal A$ 中存在余极限，并由

$$
L C
$$

给出。对偶地，余反射子范畴中的极限可由先在环境范畴取极限再余反射得到。

**证明.** 记 $i_j:I D(j)\to C$ 为 $\mathcal C$ 中的余极限结构映射。复合

$$
I D(j)\xrightarrow{i_j}C\xrightarrow{\eta_C}I L C
$$

由于 $I$ 全忠实，对应于 $\mathcal A$ 中的态射

$$
\bar i_j:D(j)\to L C.
$$

它们构成 $D$ 的余锥。对任意 $A\in\mathcal A$，有自然双射

$$
\mathcal A(L C,A)
\cong \mathcal C(C,I A)
\cong \lim_{j\in\mathcal J^{op}}\mathcal C(I D(j),I A)
\cong \lim_{j\in\mathcal J^{op}}\mathcal A(D(j),A).
$$

第一步是伴随，第二步是 $C$ 的余极限表示性，第三步用 $I$ 全忠实。该表示性正说明 $L C$ 是 $\mathcal A$ 中的余极限。对偶命题同理。$\square$

## 4.8 例子与边界条件

**例子 4.13（偏序中的伴随）.** 把偏序集 $P,Q$ 看作薄范畴。单调映射

$$
f:P\to Q,\qquad g:Q\to P
$$

构成伴随 $f\dashv g$，当且仅当对所有 $p\in P,q\in Q$ 有

$$
f(p)\le q\quad\Longleftrightarrow\quad p\le g(q).
$$

这正是 Galois connection。单位条件是 $p\le g f(p)$，余单位条件是 $f g(q)\le q$；三角恒等式在薄范畴中自动由偏序的至多一个态射性质给出。

**例子 4.14（积和余积作为伴随）.** 若 $\mathcal C$ 有二元积，则对角函子

$$
\Delta:\mathcal C\to\mathcal C\times\mathcal C,\qquad X\mapsto(X,X)
$$

有右伴随

$$
\times:\mathcal C\times\mathcal C\to\mathcal C,\qquad (A,B)\mapsto A\times B,
$$

因为有自然双射

$$
\mathcal C(X,A\times B)\cong\mathcal C(X,A)\times\mathcal C(X,B)
\cong(\mathcal C\times\mathcal C)(\Delta X,(A,B)).
$$

若 $\mathcal C$ 有二元余积，则 $\Delta$ 有左伴随 $\sqcup$，因为

$$
\mathcal C(A\sqcup B,X)\cong\mathcal C(A,X)\times\mathcal C(B,X)
\cong(\mathcal C\times\mathcal C)((A,B),\Delta X).
$$

因此积和余积不是孤立构造，而是同一个对角函子的右、左伴随。

**例子 4.15（左伴随不必保持极限）.** 自由群函子

$$
F:\mathbf{Set}\to\mathbf{Grp}
$$

是左伴随，但不保持终对象。$\mathbf{Set}$ 的终对象是单点集 $1$，而 $F(1)\cong\mathbb Z$。$\mathbf{Grp}$ 的终对象是平凡群，$\mathbb Z$ 不同构于平凡群。因此左伴随保持余极限，但一般不保持极限。

对偶地，右伴随一般不保持余极限。忘却函子 $U:\mathbf{Grp}\to\mathbf{Set}$ 是右伴随，但不保持始对象：$\mathbf{Grp}$ 的始对象是平凡群，其底层集合是单点集，而 $\mathbf{Set}$ 的始对象是空集。

## 4.9 本章小结

伴随是范畴论中最重要的结构之一。它可以写成 Hom 集自然双射，也可以写成单位、余单位和三角恒等式。左伴随保持余极限，右伴随保持极限。全忠实伴随可由单位或余单位是否为同构来检测；反射子范畴把“在环境范畴中自由逼近某类对象”的思想精确化，并解释了许多局部化与 sheaf 化构造的形式模式。

## 练习

**练习 4.1.** 证明自由阿贝尔群函子 $\mathbb Z[-]:\mathbf{Set}\to\mathbf{Ab}$ 左伴随于忘却函子。

**练习 4.2.** 写出积函子 $A\times -:\mathbf{Set}\to\mathbf{Set}$ 的右伴随，并证明伴随双射。

**练习 4.3.** 从单位和余单位出发，证明右伴随保持终对象。

**练习 4.4.** 设 $F\dashv G$。证明 $F$ 完全忠实当且仅当单位 $\eta:\operatorname{id}\Rightarrow GF$ 是自然同构。

**练习 4.5.** 对偶化定理 4.7，写出右伴随保持极限的完整证明。

**练习 4.6.** 直接证明 $G$ 完全忠实当且仅当余单位 $\varepsilon:FG\to\operatorname{id}_{\mathcal D}$ 是自然同构。

**练习 4.7.** 设 $\mathcal A\subseteq\mathcal C$ 是反射子范畴。证明若 $D:\mathcal J\to\mathcal A$ 的环境余极限已经落在 $\mathcal A$ 中，则该对象也是 $\mathcal A$ 中的余极限。

**练习 4.8.** 证明命题 4.11 中的余单位 $LIA\to A$ 是同构，并说明这正是包含函子全忠实的反映。

**练习 4.9.** 证明阿贝尔化函子
$$
(-)_{\operatorname{ab}}:\mathbf{Grp}\to\mathbf{Ab}
$$
使 $\mathbf{Ab}$ 成为 $\mathbf{Grp}$ 的反射子范畴。

**练习 4.10.** 写出余反射子范畴的对偶定义，并证明其中的极限由环境极限再余反射给出。

**练习 4.11.** 设 $f:P\to Q$、$g:Q\to P$ 是偏序集间单调映射。证明 $f\dashv g$ 当且仅当 $f(p)\le q\Leftrightarrow p\le g(q)$。

**练习 4.12.** 证明若 $\mathcal C$ 有终对象，则唯一函子 $\mathcal C\to *$ 有右伴随；若 $\mathcal C$ 有始对象，则它有左伴随。

**练习 4.13.** 当 $\mathcal C$ 为笛卡尔闭范畴时，证明二元积函子 $A\times -$ 左伴随于指数函子 $(-)^A$。

**练习 4.14.** 给出一个右伴随不保持余积的例子。
