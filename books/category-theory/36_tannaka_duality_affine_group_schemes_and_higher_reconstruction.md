# 第三十六章：Tannaka duality、仿射群概形与高阶重构

## 本章目标

本章讨论 Tannaka duality 的范畴论形式。经典 Tannaka 理论从一个带纤维函子的对称幺半阿贝尔范畴重构仿射群概形；现代高阶 Tannaka 理论则从对称幺半稳定 $\infty$-范畴、$\operatorname{QCoh}$ 和保张量函子重构 derived 或 spectral stacks。核心思想是：几何对象可以由其 sheaf 范畴及张量结构恢复。

## 依赖前置知识

需要幺半范畴、闭范畴、阿贝尔范畴、Grothendieck 范畴、presentable $\infty$-categories、$\operatorname{Pr}^L$、$\operatorname{QCoh}$、descent、Barr-Beck-Lurie 和 derived stacks。

## 36.1 经典 Tannakian 范畴

**定义 36.1.** 设 $k$ 为域。一个 neutral Tannakian category 是刚性 $k$-线性阿贝尔对称幺半范畴 $\mathcal C$，配有 faithful exact $k$-线性对称幺半函子

$$
\omega:\mathcal C\to\operatorname{Vect}_k^{fd}.
$$

称 $\omega$ 为 fiber functor。

**定义 36.2.** Fiber functor 的张量自同构群函子定义为

$$
\operatorname{Aut}^{\otimes}(\omega)(R)=
\operatorname{Aut}^{\otimes}(\omega_R)
$$

其中 $\omega_R:\mathcal C\to\operatorname{Proj}_R$ 由标量扩张得到。

**外部输入定理 36.3（经典 Tannaka duality）.** 若 $(\mathcal C,\omega)$ 是 neutral Tannakian category，则 $G=\operatorname{Aut}^{\otimes}(\omega)$ 是仿射群概形，并有对称幺半等价

$$
\mathcal C\simeq\operatorname{Rep}_k^{fd}(G),
$$

且 $\omega$ 对应遗忘函子。

**例子 36.4.** 若 $G$ 为仿射群概形，则 $\operatorname{Rep}_k^{fd}(G)$ 连同遗忘函子到 $\operatorname{Vect}_k^{fd}$ 是 neutral Tannakian category。在此情形，Tannaka duality 重构出原群概形 $G$。

## 36.2 Coend 形式的重构

**定义 36.5.** 对小刚性幺半范畴 $\mathcal C$ 和 fiber functor $\omega:\mathcal C\to\operatorname{Vect}_k^{fd}$，定义 coalgebra

$$
\mathcal O(G)=\int^{X\in\mathcal C}\omega(X)^\vee\otimes\omega(X).
$$

该 coend 称为 matrix coefficient coalgebra。

**命题 36.6.** 若 $\mathcal C=\operatorname{Rep}_k^{fd}(G)$，则 $\mathcal O(G)$ 的 coend 公式恢复 $G$ 的坐标 Hopf algebra。

**证明.** 对表示 $V$，元素 $V^\vee\otimes V$ 给出矩阵系数函数 $G\to\mathbb A^1$。Coend 关系正是要求这些矩阵系数对表示态射自然。所有有限维表示的矩阵系数生成仿射群概形的坐标 Hopf algebra；张量结构给出乘法，单位表示给出单位，对偶给出 antipode。因此 coend 恢复 $\mathcal O(G)$。严格生成性依赖经典 Tannaka 定理。$\square$

**命题 36.7.** 刚性条件使张量自同构由其在对象上的矩阵系数控制。

**证明.** 刚性给出评价和余评价

$$
X^\vee\otimes X\to\mathbb 1,\qquad \mathbb 1\to X\otimes X^\vee.
$$

张量自然自同构必须与这些结构相容，因此在 $\omega(X)$ 上的线性自同构与在 $\omega(X)^\vee$ 上的对偶自同构互相决定。矩阵系数 $\omega(X)^\vee\otimes\omega(X)$ 记录这些线性自同构的坐标；对所有 $X$ 取 coend 并施加自然性关系后得到整个自同构群函子。$\square$

## 36.3 高阶 Tannaka：从 $\operatorname{QCoh}$ 重构栈

**定义 36.8.** 设 $X$ 为 derived stack。其 quasi-coherent category

$$
\operatorname{QCoh}(X)
$$

是一个 presentable 对称幺半稳定 $\infty$-范畴。

**定义 36.9.** 对 derived stack $X$ 和 connective $E_\infty$-ring $A$，有 pullback 函子

$$
x^*:\operatorname{QCoh}(X)\to\operatorname{Mod}_A
$$

对应点 $x:\operatorname{Spec}A\to X$。这给出自然映射

$$
X(A)\to
\operatorname{Fun}^{L,\otimes}(\operatorname{QCoh}(X),\operatorname{Mod}_A).
$$

**外部输入定理 36.10（高阶 Tannaka duality）.** 对满足适当几何性、完备性和保存 connective/flat 对象条件的 derived stacks $X$，自然映射

$$
X(A)\to
\operatorname{Fun}^{L,\otimes}_{\operatorname{good}}(\operatorname{QCoh}(X),\operatorname{Mod}_A)
$$

为等价。也就是说，$X$ 可从 $\operatorname{QCoh}(X)$ 的对称幺半 presentable $\infty$-范畴结构中重构。

**命题 36.11.** 若 $X=\operatorname{Spec}R$，则高阶 Tannaka 映射退化为

$$
\operatorname{Map}_{\operatorname{CAlg}}(R,A)
\simeq
\operatorname{Fun}^{L,\otimes}(\operatorname{Mod}_R,\operatorname{Mod}_A).
$$

**证明.** 左边是仿射 functor of points。右边中保小余极限的对称幺半函子由单位对象 $R\in\operatorname{Mod}_R$ 的像决定。强幺半条件要求单位送单位，且 $R$-代数结构在 $\operatorname{Mod}_A$ 中对应一个 $E_\infty$-ring map $R\to A$。给定 $R\to A$，函子为

$$
-\otimes_R A:\operatorname{Mod}_R\to\operatorname{Mod}_A.
$$

两构造互逆。$\square$

## 36.4 Descent 与 Tannaka 重构

**命题 36.12.** 若 $X$ 有仿射覆盖 $U\to X$，且 $\operatorname{QCoh}$ 满足 Cech descent，则 $\operatorname{QCoh}(X)$ 可由 cosimplicial diagram $\operatorname{QCoh}(U_\bullet)$ 重构。

**证明.** 由 faithfully flat 或相应拓扑的 descent，

$$
\operatorname{QCoh}(X)\simeq\operatorname{Tot}\operatorname{QCoh}(U_\bullet).
$$

这里 $U_\bullet$ 是 Cech nerve。Totalization 正是把覆盖上的 quasi-coherent categories 与所有高阶相容同构粘合起来。$\square$

**命题 36.13.** Tannaka 重构与 descent 相容：若 $X$ 由 $U_\bullet$ 粘合，则从 $\operatorname{QCoh}(X)$ 得到的点等价于 compatible systems of points of $U_\bullet$。

**证明.** 对任意 $A$，高阶 Tannaka 把 $X(A)$ 识别为合适张量函子

$$
\operatorname{QCoh}(X)\to\operatorname{Mod}_A.
$$

若 $\operatorname{QCoh}(X)\simeq\operatorname{Tot}\operatorname{QCoh}(U_\bullet)$，则从 totalization 出发的张量函子等价于从每个 $\operatorname{QCoh}(U_n)$ 出发并满足 cosimplicial 相容的数据。再次用仿射或局部 Tannaka，后者等价于 $U_\bullet(A)$ 的 compatible system，即 $X(A)$ 的 descent data。$\square$

## 36.5 群对象、类ifying stacks 与表示范畴

**定义 36.14.** 设 $G$ 为 affine group scheme 或 derived affine group stack。其 classifying stack $BG$ 定义为把 $A$ 送到 $G$-torsors on $\operatorname{Spec}A$ 的空间。

**外部输入定理 36.15.** 在适当有限性假设下，

$$
\operatorname{QCoh}(BG)\simeq\operatorname{Rep}(G)
$$

作为对称幺半 presentable $\infty$-范畴。若 $G$ 是经典仿射群概形，compact/dualizable 部分恢复有限维表示范畴。

**命题 36.16.** 若 $\operatorname{QCoh}(BG)\simeq\operatorname{QCoh}(BH)$ 作为带 fiber functor 的对称幺半范畴等价，则 $G\simeq H$。

**证明.** Fiber functor 对应基点 $\operatorname{Spec}k\to BG$。其张量自同构群是 loop group

$$
\Omega_*BG\simeq G.
$$

若带 fiber functor 的对称幺半范畴等价识别 $\operatorname{QCoh}(BG)$ 与 $\operatorname{QCoh}(BH)$，则两者的张量自同构群函子等价，故 $G\simeq H$。这是 Tannaka 重构在 classifying stacks 上的特例。$\square$

## 36.6 Tannaka 映射的形式后果

**命题 36.17（由 $\operatorname{QCoh}$ 判别栈等价）.** 设 $X,Y$ 属于定理 36.10 适用的 derived stacks，且

$$
\Phi:\operatorname{QCoh}(X)\simeq\operatorname{QCoh}(Y)
$$

是保持所需 good 条件的保小余极限对称幺半等价。则 $\Phi$ 诱导 functor of points 的等价

$$
X(A)\simeq Y(A)
$$

对所有 connective $E_\infty$-ring $A$ 成立，因而 $X\simeq Y$。

**证明.** 对每个 $A$，预合成 $\Phi$ 给出等价

$$
\operatorname{Fun}^{L,\otimes}_{\operatorname{good}}(\operatorname{QCoh}(Y),\operatorname{Mod}_A)
\simeq
\operatorname{Fun}^{L,\otimes}_{\operatorname{good}}(\operatorname{QCoh}(X),\operatorname{Mod}_A).
$$

由定理 36.10，这两侧分别等价于 $Y(A)$ 与 $X(A)$。因此 $X$ 与 $Y$ 的 functor of points 等价。由 Yoneda 判别，$X\simeq Y$。$\square$

**命题 36.18（态射的重构）.** 在定理 36.10 的假设下，态射 $f:X\to Y$ 由对称幺半函子

$$
f^*:\operatorname{QCoh}(Y)\to\operatorname{QCoh}(X)
$$

决定。

**证明.** 对任意 $A$ 与点 $x:\operatorname{Spec}A\to X$，复合

$$
\operatorname{QCoh}(Y)\xrightarrow{f^*}\operatorname{QCoh}(X)\xrightarrow{x^*}\operatorname{Mod}_A
$$

是对应点 $f\circ x:\operatorname{Spec}A\to Y$ 的张量 pullback。由高阶 Tannaka，所有这样的张量函子确定 $Y(A)$ 中的点；随 $A$ 自然变化后，$f$ 作为 functor of points 的自然变换被 $f^*$ 确定。$\square$

**命题 36.19.** 若 $G$ 为 affine group stack，则基点 $*\to BG$ 的环路对象由带 fiber functor 的 $\operatorname{QCoh}(BG)$ 的张量自同构群表示。

**证明.** 基点给出 fiber functor

$$
\omega:\operatorname{QCoh}(BG)\to\operatorname{QCoh}(*) .
$$

张量自同构 $\operatorname{Aut}^{\otimes}(\omega)$ 按定义记录保持所有准凝聚表示张量结构的基点自同构。Classifying stack 的基点自同构空间正是 loop group $\Omega_*BG$，而 $\Omega_*BG\simeq G$。因此该张量自同构群表示 $G$。$\square$

## 36.7 本章小结

Tannaka duality 是“由表示范畴重构对称对象”的范畴论机制。经典理论从刚性阿贝尔张量范畴和 fiber functor 重构仿射群概形；coend 公式给出坐标 Hopf algebra；高阶理论从 $\operatorname{QCoh}(X)$ 的对称幺半 presentable $\infty$-范畴结构重构 derived stack $X$。Barr-Beck-Lurie descent 保证重构与覆盖粘合相容。

## 练习

**练习 36.1.** 定义 neutral Tannakian category。

**练习 36.2.** 定义 fiber functor 的张量自同构群函子。

**练习 36.3.** 陈述经典 Tannaka duality。

**练习 36.4.** 写出 matrix coefficient coalgebra 的 coend 公式。

**练习 36.5.** 解释为什么刚性条件对 Tannaka 重构重要。

**练习 36.6.** 写出高阶 Tannaka 中的自然映射 $X(A)\to\operatorname{Fun}^{L,\otimes}(\operatorname{QCoh}(X),\operatorname{Mod}_A)$。

**练习 36.7.** 证明仿射情形 $\operatorname{Spec}R$ 的 Tannaka 公式。

**练习 36.8.** 说明 $\operatorname{QCoh}$ descent 如何参与 Tannaka 重构。

**练习 36.9.** 定义 classifying stack $BG$。

**练习 36.10.** 说明 $\operatorname{QCoh}(BG)$ 与 $\operatorname{Rep}(G)$ 的关系。

**练习 36.11.** 证明带 fiber functor 的 $\operatorname{QCoh}(BG)$ 可恢复 $G$。

**练习 36.12.** 证明在高阶 Tannaka 假设下，$\operatorname{QCoh}(X)\simeq\operatorname{QCoh}(Y)$ 的合适张量等价推出 $X\simeq Y$。

**练习 36.13.** 说明态射 $f:X\to Y$ 如何由 $f^*:\operatorname{QCoh}(Y)\to\operatorname{QCoh}(X)$ 重构。
