# 第十一章：Fukaya-Seidel category 与 Picard-Lefschetz theory

## 本章目标

本章引入 Landau-Ginzburg A-model 的 Fukaya-Seidel category。它由 Lefschetz fibration、vanishing paths、thimbles 和 directed $A_\infty$ category 构成，是 Fano/Landau-Ginzburg HMS 中最常用的 A-side 模型之一。

## 依赖前置知识

需要第四章的 Fukaya category、第六章的非紧 exact 几何、第八章的 HMS 断言模板。

## 11.1 Lefschetz fibration

**定义 11.1.** 一个 exact symplectic Lefschetz fibration 是映射
$$
\pi:E\to\mathbb C
$$
使得临界点非退化、临界值互异，并且在临界点附近可取复坐标使
$$
\pi(z_1,\ldots,z_n)=z_1^2+\cdots+z_n^2+\text{constant}.
$$
同时 $E$ 带 exact symplectic 结构并与 fibration 兼容。

**定义 11.2.** 取基点 $b\in\mathbb C$ 避开临界值。连接 $b$ 到临界值 $c_i$ 的嵌入路径 $\gamma_i$ 称为 vanishing path。沿 $\gamma_i$ 平行移动并在临界点塌缩得到 fiber $F_b$ 中的 vanishing cycle $V_i\subset F_b$ 和 total space 中的 Lefschetz thimble $\Delta_i\subset E$。

**命题 11.3.** vanishing cycle $V_i$ 是 $F_b$ 中的 Lagrangian sphere。

**证明路线（外部输入）.** Lefschetz 临界点的局部模型为复 Morse 函数。沿路径接近临界值时，fiber 中一族实中维球面收缩到临界点。平行移动保持辛形式，因此所得子流形 Lagrangian；局部模型给出其拓扑为 sphere。完整证明使用 symplectic parallel transport。证毕。

## 11.2 Directed Fukaya category

**定义 11.4.** 给定有序 vanishing paths $\gamma_1,\ldots,\gamma_m$，directed Fukaya category $\mathcal A_{\mathrm{dir}}$ 的对象是 vanishing cycles $V_1,\ldots,V_m$，morphisms 定义为
$$
\operatorname{hom}(V_i,V_j)=
\begin{cases}
CF^\ast(V_i,V_j),& i<j,\\
k\cdot e_i,& i=j,\\
0,& i>j.
\end{cases}
$$
高阶复合由 fiber 中 holomorphic polygons 计数定义，并与方向条件相容。

**定义 11.5.** Fukaya-Seidel category $\mathcal F\mathcal S(\pi)$ 是 directed Fukaya category 的适当 twisted/split closure，或等价地由 Lefschetz thimbles 在 total space 中生成的 wrapped/partially wrapped category 的模型。具体版本依赖文献口径。

**外部输入定理 11.6（Seidel Fukaya-Seidel 构造）.** exact Lefschetz fibration 的 vanishing cycles 和 thimbles 构成 directed $A_\infty$ category；其 quasi-equivalence type 在 Hurwitz moves 和合适选择下按预期变化。  
来源：Seidel, *Fukaya Categories and Picard-Lefschetz Theory*。

## 11.3 Picard-Lefschetz 变换

**定义 11.7.** 对 Lagrangian sphere $V\subset F$，Dehn twist $\tau_V$ 是 $F$ 的 compactly supported symplectomorphism，其局部模型由 cotangent bundle $T^\ast S^n$ 中的标准 twist 给出。

**外部输入定理 11.8（Picard-Lefschetz monodromy）.** 绕一个 Lefschetz 临界值的 symplectic monodromy Hamiltonian isotopic 于沿相应 vanishing cycle 的 Dehn twist。

**解释 11.9.** Picard-Lefschetz theory 把 base 中路径的 braid 操作翻译为 fiber Fukaya category 中的 Dehn twists。这是 exceptional collections、mutations 和 HMS 中 monodromy 对应的基础。

## 11.4 HMS 形态

**定义 11.10.** 对 Fano variety $X$，其镜像 Landau-Ginzburg model 常写作
$$
W:Y\to\mathbb C.
$$
Fano/LG HMS 的标准 B-to-A 形态为
$$
\mathcal F\mathcal S(Y,W)\simeq\operatorname{Perf}(X).
$$

**命题 11.11.** 若 $\operatorname{Perf}_{\mathrm{dg}}(X)$ 由 exceptional
collection $(E_1,\ldots,E_m)$ split-generate，且
$\mathcal F\mathcal S(W)$ 由 vanishing thimbles
$(\Delta_1,\ldots,\Delta_m)$ split-generate，则只要构造 strictly unital
quasi-equivalence between the two directed full subcategories，并在对象上
匹配 $\Delta_i\mapsto E_i$，即可得到 Morita HMS。Pairwise morphism
complexes 为
$$
\operatorname{hom}_{\mathcal F\mathcal S}(\Delta_i,\Delta_j)
\quad\text{与}\quad
\mathbf R\operatorname{Hom}_X(E_i,E_j).
$$

**证明.** 两边生成对象给出 full directed subcategories；所假设的
quasi-equivalence 与两边 split-generation 满足命题 8.9，故得到 Morita
equivalence。若把有限 collection 压缩为直和的 endomorphism algebra，
必须保持投影到各 summand 的 orthogonal idempotents；只逐对比较上式的
cohomology groups 或 $\mu^2$ 不足以构造该 quasi-equivalence。证毕。

## 11.5 基本例子：$\mathbb P^1$

**例 11.12.** $\mathbb P^1$ 的镜像 potential 可取
$$
W(z)=z+qz^{-1}:\mathbb C^\ast\to\mathbb C.
$$
它有两个 critical points。Fukaya-Seidel category 有两个基本 thimbles；B-side $\operatorname{Perf}(\mathbb P^1)$ 由 exceptional collection
$$
(\mathcal O,\mathcal O(1))
$$
生成。两边 directed morphism algebra 的维数匹配：
$$
\dim\operatorname{Hom}(\mathcal O,\mathcal O(1))=2,
$$
对应两个交点或两条 thimble morphisms。

**警告 11.13.** 例 11.12 的完整 HMS 证明还需比较乘法和 $A_\infty$ 结构，而不只是 morphism 维数。

## 本章小结

Fukaya-Seidel category 是 Lefschetz fibration 的 directed Fukaya category。Vanishing cycles、thimbles、Dehn twists 和 Picard-Lefschetz monodromy 提供了可计算的 A-side 模型。Fano/LG HMS 通常通过 thimbles 与 exceptional collections 的 endomorphism algebra 比较来证明。

## 练习

**练习 11.1.** 对 $W(z)=z+qz^{-1}$，求 critical points 和 critical values。

**练习 11.2.** 写出 directed category 中 $i>j$ 时 morphism 为零如何影响可能的高阶复合。

**练习 11.3.** 解释 Dehn twist 与 Picard-Lefschetz monodromy 的关系。

**练习 11.4.** 证明 $\mathcal O,\mathcal O(1)$ split-generate $\operatorname{Perf}(\mathbb P^1)$ 可归约为 Beilinson 型分解，并标为外部输入。
