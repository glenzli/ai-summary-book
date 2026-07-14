# 核心例子闭合：椭圆、toric Fano 与 Fukaya-Seidel

本文件补足在线教材层面的三组标准例子。目标不是替代原论文证明，而是把读者最容易卡住的计算链条写成可追踪形式。

## L.1 椭圆曲线：交点、Ext 与 theta 乘法

**数据 L.1.** A-side 取辛环面 $T^2$ 上的斜率圆 $L_{r,d}$。B-side 取椭圆曲线 $E$ 上 rank-degree 为 $(r,d)$ 的 stable bundle $V_{r,d}$。

**计算 L.2（morphism 维数）。** 若 $L_{r,d}$ 与 $L_{r',d'}$ 横截，则
$$
\#(L_{r,d}\cap L_{r',d'})=|rd'-r'd|.
$$
B-side Riemann-Roch 给出
$$
\chi(V_{r,d},V_{r',d'})=rd'-r'd.
$$
稳定性决定 Ext 集中于一个次数时，Ext 维数与交点数匹配。

**计算 L.3（乘法骨架）。** 设 $x\in L_0\cap L_1$，$y\in L_1\cap L_2$，$z\in L_0\cap L_2$。A-side 乘法系数为
$$
\sum_{u\in\mathcal M(z;y,x)} \pm T^{\operatorname{Area}(u)}\operatorname{hol}(u).
$$
在平坦环面模型中，$u$ 提升为 $\mathbb R^2$ 中的仿射三角形；对所有 lift 求和得到 theta series。B-side 上，stable bundles 的 section multiplication 也由 theta functions 的乘法公式表达。因此 Polishchuk-Zaslow 的完整 HMS 可理解为三角形面积加权和与 theta 乘法的逐项匹配。

**在线闭合判定 L.4.** 本书已给出对象字典、morphism 维数、乘法来源和外部输入 locator。完整 theta 恒等式证明仍引用 Polishchuk-Zaslow。

## L.2 Toric Fano：disk potential、Jacobian ring 与 closed-open 检查

**数据 L.5.** 对 smooth toric Fano $X$，moment polytope facets 给出 Maslov index $2$ disks。镜像 LG 模型为
$$
((\mathbb C^\ast)^n,W),\qquad
W=\sum_{\rho\in\Sigma(1)}c_\rho z^{m_\rho}.
$$

**计算 L.6（$\mathbb P^n$）。** 设 $q\in k^\times$。对
$$
W=z_1+\cdots+z_n+q(z_1\cdots z_n)^{-1},
$$
critical equations 是
$$
z_i=q(z_1\cdots z_n)^{-1}.
$$
故 $z_1=\cdots=z_n=z$ 且 $z^{n+1}=q$，并有
$$
\operatorname{Jac}(W)\cong k[z]/(z^{n+1}-q).
$$
若再假设 $k$ 代数闭且 $\operatorname{char}k\nmid n+1$，该多项式有
$n+1$ 个互异根，故这些 critical points 非退化，Jacobian ring 半单分解为
$n+1$ 个域因子。没有这些域与特征假设时，不能断言根数或半单性。

**计算 L.7（closed-open 检查）。** 在外部输入定理 10.5、10.8 所规定的
toric Fano/Novikov 模型中，带局部系统或 weak bounding cochain 的 torus fiber
$(L_u,\xi)$ 的 deformed Floer cohomology 由势函数临界方程控制：
$dW(\xi)=0$ 时得到相应非零分块，非临界点在标准 Koszul 模型中给出
acyclic complex。这里的结论依赖 toric Floer 计算，不能只从形式
Maurer--Cartan 方程推出。Closed--open map 将 closed-string 数据映到
Hochschild cochains，Jacobian ring 与 closed-string ring 的同构是 toric HMS
的必要检查。

**在线闭合判定 L.8.** 本书已给出 potential、critical equations、Jacobian ring 和 closed-open 检查逻辑。完整 toric HMS 的 $A_\infty$ quasi-isomorphism 仍引用 Abouzaid/FOOO。

## L.3 $\mathbb P^1$：Fukaya-Seidel directed algebra 与 exceptional collection

**数据 L.9.** A-side 取
$$
W(z)=z+qz^{-1}:\mathbb C^\ast\to\mathbb C.
$$
它有两个 critical points，因此 Fukaya-Seidel category 有两个基本 thimbles $\Delta_0,\Delta_1$。B-side 取 $\operatorname{Perf}(\mathbb P^1)$ 与 exceptional collection $(\mathcal O,\mathcal O(1))$。

**计算 L.10（B-side algebra）。**
$$
\operatorname{Hom}(\mathcal O,\mathcal O)=k,\quad
\operatorname{Hom}(\mathcal O(1),\mathcal O(1))=k,
$$
$$
\operatorname{Hom}(\mathcal O(1),\mathcal O)=H^0(\mathbb P^1,\mathcal O(-1))=0,
$$
$$
\operatorname{Hom}(\mathcal O,\mathcal O(1))=H^0(\mathbb P^1,\mathcal O(1))\cong k^2.
$$
所以 directed algebra 是 Kronecker quiver 的 path algebra。

**计算 L.11（A-side algebra）。** 两个 thimbles 的方向排序使
$$
\operatorname{hom}(\Delta_1,\Delta_0)=0,\qquad
\operatorname{hom}(\Delta_i,\Delta_i)=k e_i.
$$
适当 distinguished basis 下，$\operatorname{hom}(\Delta_0,\Delta_1)$ 由两个 intersection/chord generators 生成。由于只有两个对象且 directed，非单位的高阶复合没有足够对象序列形成非平凡输出；因此在线教材层面可把 directed algebra 识别为 Kronecker algebra。完整证明还需 Seidel 的 Fukaya-Seidel 构造与 thimble generation。

**命题 L.12（$\mathbb P^1$ 在线 HMS 闭合）。** 在外部输入“thimbles generate $\mathcal F\mathcal S(W)$”和“$(\mathcal O,\mathcal O(1))$ generate $\operatorname{Perf}(\mathbb P^1)$”下，计算 L.10 与 L.11 给出
$$
\mathcal F\mathcal S(W)\simeq_{\mathrm{Morita}}\operatorname{Perf}(\mathbb P^1)
$$
的在线教材级证明骨架。

**证明.** 两边生成对象的 full directed subcategories 都由 Kronecker algebra 描述。由生成元比较模板 K.1，full subcategories quasi-equivalent 蕴含 perfect closures Morita equivalent。证毕。

## L.4 三个例子的在线闭合状态

| 例子 | 已内部计算 | 外部输入 |
| --- | --- | --- |
| 椭圆曲线 | 斜率交点、Euler characteristic、theta 乘法来源 | Polishchuk-Zaslow 完整 theta 恒等式与 HMS |
| toric Fano | potential、critical equations、Jacobian ring、closed-open 检查逻辑 | Abouzaid/FOOO toric HMS 与 disk counts |
| $\mathbb P^1$ Fukaya-Seidel | Kronecker algebra 两侧计算、Morita 推论 | Seidel thimble generation、Beilinson generation |

## 本文件小结

标准例子现在达到在线教材闭合：每个例子都有数据、计算、外部输入和 Morita 推论。出版级别仍需补原文 theorem 编号、图示、符号约定逐项比对和完整高阶乘法校勘。
