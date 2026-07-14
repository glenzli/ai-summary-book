# 第四章：伴随函子

自由群、张量--Hom、逆像--直像等构造都呈现同一种不对称的最佳逼近：一个方向上的态射可以自然地改写为另一个方向上的态射。伴随把这种现象表达为 Hom 集的双自然同构，也可等价地编码为单位、余单位和三角恒等式，或逐对象的泛箭头。本章证明这些语言彼此等价，并从泛性质推出左伴随保持余极限、右伴随保持极限；这些结论将在 Kan 延拓、单子和局部化中反复出现。

$\mathcal C,\mathcal D$ 默认局部 $\mathcal U$-小，因此伴随双射发生在 $\mathbf{Set}_{\mathcal U}$ 中。“保持所有小极限或余极限”只量化 $\mathcal U$-小索引范畴；改变大小轮廓时，伴随存在性与完备性条件也需重新声明。

## 4.1 伴随的定义

**定义 4.1.** 设 $F:\mathcal C\to\mathcal D$ 与 $G:\mathcal D\to\mathcal C$ 为函子。称 $F$ 左伴随于 $G$，记作 $F\dashv G$，若存在对 $X\in\mathcal C$ 与 $Y\in\mathcal D$ 自然的双射

$$
\Phi_{X,Y}:\mathcal D(FX,Y)\xrightarrow{\cong}\mathcal C(X,GY).
$$

此时 $G$ 称为 $F$ 的右伴随。

这里“对两个变量自然”不是附加口号。对
$u:X'\to X$、$v:Y\to Y'$ 和 $f:FX\to Y$，它精确要求

$$
\Phi_{X',Y}(f\circ F(u))
=\Phi_{X,Y}(f)\circ u,
$$

$$
\Phi_{X,Y'}(v\circ f)
=G(v)\circ\Phi_{X,Y}(f).
$$

第一变量反变，第二变量协变；这两个等式也固定了后文所有转置公式的方向。

**例子 4.2.** 自由群函子

$$
F:\mathbf{Set}_{\mathcal U}\to\mathbf{Grp}
$$

左伴随于忘却函子
$U:\mathbf{Grp}\to\mathbf{Set}_{\mathcal U}$，其中
$\mathbf{Grp}$ 只取底层集合为 $\mathcal U$-小的群。伴随双射为

$$
\mathbf{Grp}(F(S),G)\cong\mathbf{Set}_{\mathcal U}(S,U(G)),
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

**证明.** 先从 Hom 自然同构 $\Phi$ 构造单位和余单位。令

$$
\eta_X=\Phi_{X,FX}(\operatorname{id}_{FX}),
\qquad
\varepsilon_Y=\Phi^{-1}_{GY,Y}(\operatorname{id}_{GY}).
$$

先验证它们确实自然。若 $u:X\to X'$，分别对 $\Phi$ 的第二变量和第一变量使用定义 4.1 的自然性，得到

$$
\begin{aligned}
GF(u)\eta_X
&=GF(u)\Phi_{X,FX}(\operatorname{id}_{FX})\\
&=\Phi_{X,FX'}(F(u))\\
&=\Phi_{X',FX'}(\operatorname{id}_{FX'})u
=\eta_{X'}u.
\end{aligned}
$$

故 $\eta:\operatorname{id}_{\mathcal C}\Rightarrow GF$ 自然。记
$\Psi=\Phi^{-1}$。逆双射的自然性等式与 $\Phi$ 等价。若
$v:Y\to Y'$，则

$$
\begin{aligned}
v\varepsilon_Y
&=v\Psi_{GY,Y}(\operatorname{id}_{GY})\\
&=\Psi_{GY,Y'}(G(v))\\
&=\Psi_{GY',Y'}(\operatorname{id}_{GY'})F(G(v))\\
&=\varepsilon_{Y'}FG(v).
\end{aligned}
$$

所以 $\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}$ 自然。

同一自然性给出完整转置公式。若 $f:FX\to Y$，则

$$
\Phi_{X,Y}(f)
=G(f)\Phi_{X,FX}(\operatorname{id}_{FX})
=G(f)\eta_X.
$$

若 $g:X\to GY$，则

$$
\Psi_{X,Y}(g)
=\Psi_{GY,Y}(\operatorname{id}_{GY})F(g)
=\varepsilon_YF(g).
$$

把 $\eta_X=\Phi(\operatorname{id}_{FX})$ 代入
$\Psi\Phi=\operatorname{id}$，得到

$$
\varepsilon_{FX}F(\eta_X)=\operatorname{id}_{FX}.
$$

把 $\varepsilon_Y=\Psi(\operatorname{id}_{GY})$ 代入
$\Phi\Psi=\operatorname{id}$，得到

$$
G(\varepsilon_Y)\eta_{GY}=\operatorname{id}_{GY}.
$$

这就是两个三角恒等式。

反过来，给定自然变换 $\eta,\varepsilon$ 满足三角恒等式，定义

$$
\Phi_{X,Y}(f)=G(f)\eta_X,
\qquad
\Psi_{X,Y}(g)=\varepsilon_YF(g).
$$

若 $f:FX\to Y$，则

$$
\begin{aligned}
\Psi_{X,Y}(\Phi_{X,Y}(f))
&=\varepsilon_YFG(f)F(\eta_X)\\
&=f\varepsilon_{FX}F(\eta_X)\\
&=f,
\end{aligned}
$$

其中第二行用 $\varepsilon$ 对 $f:FX\to Y$ 的自然性，第三行用第一三角恒等式。若 $g:X\to GY$，则

$$
\begin{aligned}
\Phi_{X,Y}(\Psi_{X,Y}(g))
&=G(\varepsilon_Y)GF(g)\eta_X\\
&=G(\varepsilon_Y)\eta_{GY}g\\
&=g,
\end{aligned}
$$

其中第二行用 $\eta$ 对 $g:X\to GY$ 的自然性，第三行用第二三角恒等式。因此 $\Phi$ 与 $\Psi$ 逐对互逆。

最后逐项验证自然性。对 $u:X'\to X$、$v:Y\to Y'$，

$$
\begin{aligned}
\Phi_{X',Y}(fF(u))
&=G(f)GF(u)\eta_{X'}
=G(f)\eta_Xu
=\Phi_{X,Y}(f)u,\\
\Phi_{X,Y'}(vf)
&=G(v)G(f)\eta_X
=G(v)\Phi_{X,Y}(f).
\end{aligned}
$$

相应地，

$$
\begin{aligned}
\Psi_{X',Y}(gu)
&=\varepsilon_YF(g)F(u)
=\Psi_{X,Y}(g)F(u),\\
\Psi_{X,Y'}(G(v)g)
&=\varepsilon_{Y'}FG(v)F(g)
=v\varepsilon_YF(g)
=v\Psi_{X,Y}(g).
\end{aligned}
$$

故这些双射对两个变量自然，确实给出伴随。由重构双射再提取单位和余单位时，

$$
\Phi_{X,FX}(\operatorname{id}_{FX})=\eta_X,\qquad
\Psi_{GY,Y}(\operatorname{id}_{GY})=\varepsilon_Y,
$$

而从原 Hom 双射提取的 $\eta,\varepsilon$ 又由转置公式重构原双射。因此两种数据转换互逆。$\square$

## 4.3 泛箭头

**定义 4.5.** 给定 $G:\mathcal D\to\mathcal C$ 和对象 $X\in\mathcal C$，从 $X$ 到 $G$ 的泛箭头是对象 $Y\in\mathcal D$ 与态射 $\eta:X\to GY$，使得对任意 $Z\in\mathcal D$ 和任意 $f:X\to GZ$，存在唯一 $\bar f:Y\to Z$ 满足

$$
G(\bar f)\circ\eta=f.
$$

**命题 4.6（逐对象泛箭头的函子化）.** 若为每个
$X\in\mathcal C$ 给定一个从 $X$ 到 $G:\mathcal D\to\mathcal C$
的泛箭头

$$
\eta_X:X\to G(FX),
$$

则这些数据唯一决定 $F$ 在态射上的作用，使 $F$ 成为
$G$ 的左伴随且 $\eta$ 成为其单位。反之，任何伴随的单位逐分量都是这样的泛箭头。

因此，在附录 A 的 $\mathcal V$-小选择公理下，“每个 $X$ 至少存在一个
泛箭头”等价于“$G$ 有左伴随”。若不给选择原则，正确输入必须是已经选定的
泛箭头族，而不能把逐对象存在量词自动提升为一个族。

**证明.** 若 $F\dashv G$，令 $\eta_X:X\to GFX$ 为伴随单位。给定 $Z\in\mathcal D$ 和 $f:X\to GZ$，令 $\bar f:FX\to Z$ 为 $f$ 在伴随双射

$$
\mathcal D(FX,Z)\cong\mathcal C(X,GZ)
$$

下的逆像。转置公式给出

$$
G(\bar f)\circ\eta_X=f.
$$

若 $h:FX\to Z$ 也满足 $G(h)\eta_X=f$，则 $h$ 与 $\bar f$ 在伴随双射下有同一像，故 $h=\bar f$。所以 $\eta_X$ 是从 $X$ 到 $G$ 的泛箭头。

反过来，使用命题中给定的泛箭头族

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

**定理 4.7.** 左伴随保持所有存在的 $\mathcal U$-小余极限；右伴随保持所有存在的 $\mathcal U$-小极限。

**证明.** 设 $F:\mathcal C\to\mathcal D$ 左伴随于 $G$，单位为
$\eta$。取 $\mathcal U$-小范畴 $\mathcal J$、图形
$H:\mathcal J\to\mathcal C$ 及其余极限锥

$$
\iota_j:H(j)\to L.
$$

对任意 $Y\in\mathcal D$，依次使用伴随、余极限的表示性刻画和逐项伴随，得到双射

$$
\begin{aligned}
\mathcal D(FL,Y)
&\cong\mathcal C(L,GY)\\
&\cong\lim_{j\in\mathcal J^{\operatorname{op}}}
       \mathcal C(H(j),GY)\\
&\cong\lim_{j\in\mathcal J^{\operatorname{op}}}
       \mathcal D(FH(j),Y).
\end{aligned}
$$

三步都对 $Y$ 自然，所以复合也是自然双射。还需核对该双射对应的是
给定余锥 $(F\iota_j)$，而不是某个未指明的余锥。若
$q:FL\to Y$，第一步把它送到 $G(q)\eta_L$；第二步得到族

$$
G(q)\eta_L\iota_j.
$$

由 $\eta$ 对 $\iota_j$ 的自然性，

$$
G(q)\eta_L\iota_j
=G(q)GF(\iota_j)\eta_{H(j)}
=G(qF(\iota_j))\eta_{H(j)}.
$$

最后一步的逆伴随恰把该分量送回
$qF(\iota_j):FH(j)\to Y$。所以复合双射正是

$$
q\longmapsto(qF(\iota_j))_j,
$$

即余锥 $(F\iota_j)$ 的泛性质。因此 $FL$ 是 $FH$ 的余极限。

对右伴随的结论，考虑反范畴中的伴随

$$
G^{\operatorname{op}}\dashv F^{\operatorname{op}}:
\mathcal D^{\operatorname{op}}\rightleftarrows
\mathcal C^{\operatorname{op}}.
$$

刚证的结论说明左伴随 $G^{\operatorname{op}}$ 保持余极限；把反范畴中的余极限翻回原范畴，正得到 $G$ 保持极限。索引范畴的
$\mathcal U$-小性在取反后不变。$\square$

## 4.5 伴随的唯一性

**命题 4.8（伴随结构的唯一性）.** 设
$F\dashv G$ 与 $F'\dashv G$ 是两个伴随，单位、余单位分别记为

$$
(\eta,\varepsilon),\qquad(\eta',\varepsilon').
$$

则存在唯一自然同构 $\theta:F\Rightarrow F'$ 满足

$$
G(\theta_X)\eta_X=\eta'_X
\qquad(\forall X\in\mathcal C).
$$

同一个 $\theta$ 也满足等价的余单位相容式

$$
\varepsilon'_Y\theta_{GY}=\varepsilon_Y
\qquad(\forall Y\in\mathcal D).
$$

因此“两个左伴随唯一同构”指唯一的伴随结构相容自然同构，不表示
$F$ 与 $F'$ 之间只有一个任意自然同构。

**证明.** 由命题 4.6，$\eta_X:X\to G(FX)$ 是从 $X$ 到 $G$ 的泛箭头。把
$\eta'_X:X\to G(F'X)$ 代入其泛性质，得到唯一态射

$$
\theta_X:FX\to F'X
$$

满足 $G(\theta_X)\eta_X=\eta'_X$。交换两个伴随后，同理得到唯一
$\rho_X:F'X\to FX$ 满足 $G(\rho_X)\eta'_X=\eta_X$。于是

$$
G(\rho_X\theta_X)\eta_X
=G(\rho_X)\eta'_X
=\eta_X.
$$

$\operatorname{id}_{FX}$ 也满足这个等式；由泛箭头的唯一性，
$\rho_X\theta_X=\operatorname{id}_{FX}$。交换 $F,F'$ 得
$\theta_X\rho_X=\operatorname{id}_{F'X}$，故每个 $\theta_X$ 可逆。

若 $u:X\to X'$，比较
$\theta_{X'}F(u)$ 与 $F'(u)\theta_X:FX\to F'X'$。分别施加
$G$ 并预复合 $\eta_X$，有

$$
\begin{aligned}
G(\theta_{X'}F(u))\eta_X
&=G(\theta_{X'})\eta_{X'}u
=\eta'_{X'}u,\\
G(F'(u)\theta_X)\eta_X
&=GF'(u)\eta'_X
=\eta'_{X'}u.
\end{aligned}
$$

第一行使用 $\eta$ 的自然性和 $\theta_{X'}$ 的定义，第二行使用
$\theta_X$ 的定义和 $\eta'$ 的自然性。由 $\eta_X$ 的泛箭头唯一性，

$$
\theta_{X'}F(u)=F'(u)\theta_X.
$$

所以 $\theta:F\Rightarrow F'$ 是自然同构。

再证余单位相容。按第一个伴随的转置公式，
$\varepsilon_Y:FGY\to Y$ 的转置是 $\operatorname{id}_{GY}$。另一方面，

$$
\begin{aligned}
G(\varepsilon'_Y\theta_{GY})\eta_{GY}
&=G(\varepsilon'_Y)G(\theta_{GY})\eta_{GY}\\
&=G(\varepsilon'_Y)\eta'_{GY}\\
&=\operatorname{id}_{GY},
\end{aligned}
$$

最后一行是第二个伴随的三角恒等式。因此
$\varepsilon'_Y\theta_{GY}$ 与 $\varepsilon_Y$ 在第一个伴随双射下有同一转置，故二者相等。

最后，若 $\widetilde\theta:F\Rightarrow F'$ 也满足单位相容式，则每个
$\widetilde\theta_X$ 与 $\theta_X$ 都是泛箭头 $\eta_X$ 对
$\eta'_X$ 的唯一分解，故逐分量相等。于是 $\widetilde\theta=\theta$。

余单位条件也唯一刻画同一个 $\theta$。事实上，对任意自然变换
$\tau:F\Rightarrow F'$，令

$$
\delta_Y=\varepsilon'_Y\tau_{GY}:FGY\to Y.
$$

由 $\tau$ 对 $\eta'_X:X\to GF'X$ 的自然性和第二个伴随的三角恒等式，

$$
\delta_{F'X}F(\eta'_X)
=\varepsilon'_{F'X}\tau_{GF'X}F(\eta'_X)
=\varepsilon'_{F'X}F'(\eta'_X)\tau_X
=\tau_X.
$$

所以 $\tau\mapsto\varepsilon'(\tau G)$ 是单射。满足
$\varepsilon'(\tau G)=\varepsilon$ 的自然变换至多一个；已经构造的
$\theta$ 满足该式，故余单位条件与单位条件确实选出同一个相容同构。$\square$

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

## 4.9 伴随作为泛构造机器

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
