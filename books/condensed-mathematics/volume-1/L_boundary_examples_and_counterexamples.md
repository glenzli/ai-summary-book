# 附录 L：边界例子与反例

## L.0 目标

严格教材不只需要定义和定理，也需要说明假设为什么不能删。本附录收集第一卷基础理论中最容易误用的边界例子：

1. sheaf 满射不等于逐对象满射。
2. separated presheaf 不一定是 sheaf。
3. 基子站点比较需要拉回稳定和共同细化。
4. 普通张量积不保持无限乘积。
5. 拓扑阿贝尔群范畴不是凝聚阿贝尔群范畴的替代品。

这些例子用于防止把凝聚数学退化成点集拓扑或普通代数的直觉类比。

## L.1 Sheaf 满射不是逐对象满射

令 $X=S^1$，考虑拓扑空间 $X$ 上的 sheaf：

$$
\mathcal C^0_\mathbb R(U)=\{U\to\mathbb R\text{ 连续函数}\},
$$

$$
\mathcal C^0_{S^1}(U)=\{U\to S^1\text{ 连续函数}\}.
$$

指数映射给出 sheaf 态射

$$
\exp(2\pi i-):\mathcal C^0_\mathbb R\to\mathcal C^0_{S^1}.
$$

**命题 L.1.** 该态射是 sheaf 满射，但在全局截面上不是满射。

**证明.** 对任意开集 $U$ 和连续映射 $g:U\to S^1$，每个点 $u\in U$ 有邻域 $V$，使 $g|_V$ 落在 $S^1$ 的一个去掉一点的弧中。该弧有连续 argument 分支，因此 $g|_V=\exp(2\pi if_V)$。这说明态射局部满，即为 sheaf 满射。

取全局截面 $g:S^1\to S^1$ 为恒等映射。若存在连续 $f:S^1\to\mathbb R$ 使 $g=\exp(2\pi if)$，则 $g$ 有全局连续 argument。沿 $S^1$ 绕一圈时 argument 必须增加 $1$，但连续函数 $f$ 在同一点的取值不能同时相差 $1$。矛盾。因此全局截面不满。证毕。

**意义.** 第一卷中 ED 空间的作用正是把 sheaf 满射在 ED 测试对象上变成逐点满射；没有 ED 投射性，不能把局部提升误当成全局提升。

## L.2 Separated presheaf 不一定是 sheaf

令 $X$ 是非连通拓扑空间，例如 $X=U\sqcup V$，其中 $U,V$ 非空开闭。定义 presheaf $F$：

$$
F(W)=\{\text{常值函数 }W\to\mathbb Z\}.
$$

限制映射为普通函数限制。

**命题 L.2.** $F$ 是 separated presheaf，但不是 sheaf。

**证明.** 若两个常值函数在覆盖上限制相同，则它们在每个点取值相同，因此原函数相同。这给出 separated 性。

但在覆盖 $X=U\cup V$ 上，可取 $s_U=0\in F(U)$、$s_V=1\in F(V)$。由于 $U\cap V=\varnothing$，交叠相容条件为空。若 $F$ 是 sheaf，应有 $s\in F(X)$ 粘合它们。但 $F(X)$ 只允许常值函数，不能同时在 $U$ 上为 $0$、在 $V$ 上为 $1$。故不是 sheaf。证毕。

**意义.** sheaf 条件包含唯一性和存在性；只检查 separated 性不够。

## L.3 基子站点比较的失败模式

设 $\mathcal C$ 是一个有覆盖的站点，$\mathcal B\subset\mathcal C$ 是 full subcategory。站点比较定理需要 $\mathcal B$ 对覆盖、拉回和共同细化有足够稳定性。

**例 L.3（缺少拉回稳定会丢失交叠条件）.** 在拓扑空间 $X$ 的开集站点中，令 $\mathcal B$ 只包含两个开集 $U,V$ 和 $X=U\cup V$，但不包含交集 $U\cap V$。则在 $\mathcal B$ 上检查覆盖 $U,V\to X$ 时，无法表达匹配族必须在 $U\cap V$ 上相等。

**结论 L.4.** 若子站点不包含或不能覆盖拉回 $U\times_XV$，则 sheaf 条件的等化子

$$
F(X)\to F(U)\times F(V)\rightrightarrows F(U\cap V)
$$

无法在子站点中完整检测。

**证明.** sheaf 条件的第二个箭头目标正是交叠对象。如果子站点中没有交叠，也没有能共同细化交叠的对象，则两个局部截面是否相容没有检测位置。因此限制到该子站点的数据不足以恢复原 sheaf 条件。证毕。

**意义.** 第一卷附录 B 和卷四的稳定基版本必须显式要求拉回稳定或共同细化；这不是技术装饰，而是 sheaf 条件本身需要。

## L.4 普通张量积不保持无限乘积

考虑自然映射

$$
\left(\prod_{n\ge1}\mathbb Z\right)\otimes_\mathbb Z\mathbb Q
\to
\prod_{n\ge1}\mathbb Q.
$$

**命题 L.5.** 该映射不是满射。

**证明.** 左侧任一元素可写成有限和

$$
\sum_{k=1}^r a^{(k)}\otimes q_k,
\qquad
a^{(k)}\in\prod_n\mathbb Z,\ q_k\in\mathbb Q.
$$

取整数 $N>0$，使所有 $Nq_k\in\mathbb Z$。则该元素在 $\prod_n\mathbb Q$ 中的每个坐标都属于 $\frac1N\mathbb Z$。

但序列

$$
\left(1,\frac12,\frac13,\ldots\right)\in\prod_{n\ge1}\mathbb Q
$$

不属于任何固定的 $\frac1N\mathbb Z$ 的逐坐标乘积，因为取 $n>N$ 且 $n\nmid N$ 时 $1/n\notin\frac1N\mathbb Z$。故它不在像中。证毕。

**意义.** solid 张量积不能从普通张量积的有限维直觉推出；无限乘积正是 solid 理论必须修正的地方。

## L.5 拓扑阿贝尔群不是阿贝尔范畴替代品

设 $\mathbf{TopAb}$ 为拓扑阿贝尔群范畴。令 $\mathbb R_{\mathrm{disc}}$ 表示带离散拓扑的加法群，$\mathbb R_{\mathrm{std}}$ 表示带通常拓扑的加法群。恒等群同态

$$
u:\mathbb R_{\mathrm{disc}}\to\mathbb R_{\mathrm{std}}
$$

连续，因为源空间离散。

**命题 L.6.** 若把 cokernel、image、coimage 都带上拓扑范畴中的自然拓扑，$\mathbf{TopAb}$ 中的正合性行为不满足普通阿贝尔范畴的稳定性质。

**证明.** 态射 $u$ 的 kernel 为 $0$。因此 coimage 是

$$
\operatorname{coim}(u)=\mathbb R_{\mathrm{disc}}/0=\mathbb R_{\mathrm{disc}}.
$$

态射 $u$ 的代数像是全体 $\mathbb R$，带 codomain 的子空间拓扑，因此

$$
\operatorname{im}(u)=\mathbb R_{\mathrm{std}}.
$$

canonical map

$$
\operatorname{coim}(u)\to\operatorname{im}(u)
$$

就是恒等双射

$$
\mathbb R_{\mathrm{disc}}\to\mathbb R_{\mathrm{std}}.
$$

它连续，但不是同胚，因为反向映射 $\mathbb R_{\mathrm{std}}\to\mathbb R_{\mathrm{disc}}$ 不连续：例如单点集 $\{0\}$ 在离散拓扑中开，但在通常拓扑中不开。于是 image 与 coimage 的 canonical map 不是同构。阿贝尔范畴要求每个态射的 coimage 到 image 为同构，因此 $\mathbf{TopAb}$ 不是阿贝尔范畴。证毕。

**边界说明.** 本命题只用于说明风险；第一卷主体不依赖 $\mathbf{TopAb}$ 的反例分类。若要系统研究拓扑阿贝尔群，应另设 quasi-abelian 或 exact category 框架。

## L.6 练习

**练习 L.1.** 在命题 L.1 中，把 $S^1$ 改为任一有非零 winding number 的映射，证明它没有全局连续 argument。

**练习 L.2.** 修改 L.2 的例子，构造一个 presheaf 满足粘合存在性但不满足唯一性。

**练习 L.3.** 给出一个开集基 $\mathcal B$，说明它因为对交集封闭而能够检测 sheaf 条件。

**练习 L.4.** 在命题 L.5 中，把 $\mathbb Q$ 替换为 $\mathbb Z[1/p]$，判断对应映射是否满射。
