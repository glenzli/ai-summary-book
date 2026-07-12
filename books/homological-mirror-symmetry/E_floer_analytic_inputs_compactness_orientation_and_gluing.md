# 附录 E：Floer analytic inputs、compactness、orientation 与 gluing

## E.1 本附录固定的 exact compact 模型

**清单 E.1.** 任何使用 holomorphic curve 计数的 Fukaya theory 都必须分别
处理：Fredholm 性与指数、transversality、energy/action estimate、目标端的
$C^0$ 控制、Gromov compactness、bubbling、determinant-line orientations、
gluing、以及 continuation/choice independence。Exactness 只直接处理其中
的 sphere/disk bubbling；它不自动推出其余各项。

**约定 E.1A（几何与对象）.** 本附录只处理如下模型。

1. $(M_0,\lambda)$ 是 $2n$ 维 Liouville domain：$M_0$ 紧，
   $\omega=d\lambda$ 为辛形式，且由 $\iota_Z\omega=\lambda$ 定义的
   Liouville 向量场 $Z$ 沿 $\partial M_0$ 向外。记 $\widehat M$ 为其
   completion；在柱状端 $[1,\infty)\times\partial M_0$ 上写
   $\lambda=r\alpha$。
2. 固定一个 $\mathcal U$-小的 compact exact brane 集合 $\mathscr L$；每个
   $L\in\mathscr L$ 位于 $\operatorname{int}M_0$，并带 chosen primitive
   $f_L$、相对于固定 Maslov cover 的 grading、有限秩 $k$-局部系统 $E_L$。
3. 若 $\operatorname{char}k\ne2$，再固定 background class
   $b_M\in H^2(M_0;\mathbb Z/2)$，并在每个 $L$ 上选择相对
   $\operatorname{Pin}^{\pm}$ 结构；其 obstruction 分别要求
   $b_M|_L=w_2(TL)$ 或
   $b_M|_L=w_2(TL)+w_1(TL)^2$。在定向 spin 口径可只用 relative spin。
   特征 $2$ 时可不选择 signs 所需的这些数据，但这不会消除 compactness
   或 transversality 责任。

这一定义域不含非紧 conical Lagrangians。后者配以无穷远增长 Hamiltonian
时属于第六章 wrapped 口径。

**定义 E.1B（coherent Floer data）.** 对每个有序对象对 $(L_0,L_1)$，取
compactly supported Hamiltonian $H_{01}:[0,1]\times\widehat M\to\mathbb R$
和 contact-type almost complex structures $J_{01,t}$，使从 $L_0$ 到
$L_1$ 的 time-one Hamiltonian chords 非退化。特别地，$L_0=L_1$ 时也用
该扰动；不能把自 morphisms 定义成 $L\cap L$ 的离散生成集。

对每个稳定 $(d+1)$-punctured disk 的 universal family，取 strip-like ends
以及 domain-dependent datum $(K_S,J_S)$：$K_S$ 是取值于 compactly
supported Hamiltonians 的 $1$-形式，$J_S$ 在固定紧集外为 contact type，
并在各端限制为相应 pair datum。称这些数据 coherent，若它们延拓到
Deligne--Mumford compactification，且在每个边界 stratum 上等于各组件
数据的 gluing limit。还要求 Hamiltonian curvature 有统一能量界；这是
后述 energy estimate 的输入。

## E.2 Fredholm 模空间与维数

**定义 E.2.** 令 $x_i$ 为相应 pair data 的 Hamiltonian chords。对
$p>2$，在带指数权的 $W^{1,p}$ 映射空间上考虑方程
$$
(du-X_{K_S})^{0,1}_{J_S}=0,
\tag{E.1}
$$
边界第 $i$ 段落在 $L_i$，各 strip-like end 渐近于
$x_0;x_d,\ldots,x_1$。其解连同域参数的模空间记为
$$
\mathcal M(x_0;x_d,\ldots,x_1).
$$
当 $d=1$ 时除去 strip translation。线性化算子记为 $D_u$；若 $D_u$
满射，称 $u$ regular。采用本书 cohomological grading 时，regular 分支的
维数为
$$
\operatorname{vdim}\mathcal M(x_0;x_d,\ldots,x_1)
=|x_0|-\sum_{i=1}^d|x_i|+d-2.
\tag{E.2}
$$
因此刚性解只可能满足
$|x_0|=\sum_i|x_i|+2-d$。

**警告 E.3.** “Expected dimension 为 $0$”不推出模空间是有限集合。
还必须先有 Fredholm regularity，再有 $C^0$ 控制和 compactness。反之，
compactness 也不推出 regularity；例如未经扰动的自交条件 $L_0=L_1$
通常有正维生成集。

## E.3 Exactness 排除哪些 bubbles

**命题 E.4（非平凡 sphere 与单边 disk bubbles 被排除）.** 设 $J$ 与
$\omega$ tame。若 $v:S^2\to\widehat M$ 为 $J$-holomorphic sphere，或
$v:(D,\partial D)\to(\widehat M,L)$ 为边界落在一个 compact exact
$L\in\mathscr L$ 上的 $J$-holomorphic disk，则 $v$ 为常值。

**证明.** Sphere 情形由 Stokes 定理得到
$$
\int_{S^2}v^*\omega=\int_{S^2}d(v^*\lambda)=0.
$$
Disk 情形使用 $\lambda|_L=df_L$：
$$
\int_Dv^*\omega=\int_{\partial D}v^*\lambda
=\int_{\partial D}d(f_L\circ v)=0.
$$
对 tame $J$，非恒定 $J$-holomorphic curve 的能量严格为正；故两种曲线
均只能为常值。证毕。

**边界 E.4A.** 命题 E.4 只排除非恒定 bubbles。稳定紧化中的常值组件
仍可能出现，它们记录 domain 的 associahedral 边界，并由稳定性处理。
此外，exactness 不提供柱状端的 $C^0$ 界；no-escape 仍需要 contact-type
数据与最大值原理。非 exact 情形中该证明失效，例如 $S^2$ 赤道边界的
半球给出正面积 holomorphic disks，可能产生 $\mu^0$。

## E.4 Orientation lines 与局部系统系数

**定义 E.5.** 对非退化 chord $x$，取相应 half-plane Cauchy--Riemann
算子 $D_x$。它的实 determinant line 是
$$
\det(D_x)=\Lambda^{\max}\ker D_x\otimes
(\Lambda^{\max}\operatorname{coker}D_x)^\vee.
$$
由其两个 orientations 生成并规定反向 orientation 等于负元，得到秩一
自由 $\mathbb Z$-module $o_x^{\mathbb Z}$；定义
$o_x=o_x^{\mathbb Z}\otimes_{\mathbb Z}k$。Relative Pin data 与
determinant-line gluing 给每个 rigid polygon 一个次数 $2-d$ 的 map
$$
c_u:o_{x_d}\otimes\cdots\otimes o_{x_1}\longrightarrow o_{x_0}.
\tag{E.3}
$$

若 branes 带局部系统，chord 生成元的系数为
$$
\operatorname{Hom}_k((E_{L_0})_{x(0)},(E_{L_1})_{x(1)})\otimes o_x.
\tag{E.4}
$$
沿 $u$ 的各条边界弧做 parallel transport，再按边界次序复合，得到
$$
\operatorname{PT}_u:
\bigotimes_{i=d}^{1}
\operatorname{Hom}_k((E_{L_{i-1}})_{x_i(0)},(E_{L_i})_{x_i(1)})
\longrightarrow
\operatorname{Hom}_k((E_{L_0})_{x_0(0)},(E_{L_d})_{x_0(1)}).
\tag{E.5}
$$
因此一般 polygon coefficient 是 $\operatorname{PT}_u\otimes c_u$，不是
一个标量。只有所有局部系统秩一且已选 orientation-line bases 时，才可
把它压缩成 $n(x_0;x_d,\ldots,x_1)\in k$。

## E.5 外部分析输入包

**外部输入定理 E.6（compact exact Fukaya analytic package）.** 在约定
E.1A 的 Liouville completion、compact exact branes、grading 与 relative
Pin 数据下，可选择定义 E.1B 的 coherent perturbation data，使下列结论
对构造 $A_\infty$ 运算所需的 expected dimensions $0,1$ 同时成立：

1. **Fredholm 与 regularity：** 方程 (E.1) 的线性化为 Fredholm，相关
   模空间 regular，并具有 (E.2) 的维数。
2. **Energy 与 no-escape：** 固定 asymptotic chords 后，解的能量由
   action 差和 Hamiltonian curvature 的统一常数控制；contact-type
   maximum principle 使所有解留在一个固定紧集。
3. **Compactness：** expected dimension $0$ 的模空间是紧的有限定向
   $0$-manifold。Expected dimension $1$ 的 Gromov compactification 是紧
   定向 $1$-manifold with boundary；命题 E.4 排除非恒定 sphere/disk
   bubbles，其 codimension-one boundary 只有 broken polygons：
   $$
   \coprod_{\substack{r+s+t=d\\s\ge1}}\ \coprod_y
   \mathcal M^0(x_0;x_d,\ldots,x_{r+s+1},y,x_r,\ldots,x_1)
   \times
   \mathcal M^0(y;x_{r+s},\ldots,x_{r+1}).
   \tag{E.6}
   $$
4. **Orientation compatibility：** determinant-line gluing 使 (E.3) 相干。
   本书把边界 orientation 的比较符号固定为
   $$
   (-1)^{\sum_{j=r+s+1}^{d}|s x_j|},
   \tag{E.7}
   $$
   因而 (E.6) 的带符号边界和逐项等于公式 (B.3)。
5. **Gluing：** 每个 broken configuration 附近都有唯一的充分长 neck
   gluing family，给出边界 collar；反过来，趋向该边界的序列在子列后
   来自此 gluing。故 (E.6) 既不漏边界点，也不重复计算。

来源：Seidel, *Fukaya Categories and Picard--Lefschetz Theory*，第 8--12
章；其中第 11 章处理 indices 与 determinant lines，第 12 章给出 complete
exact Fukaya category。该包还使用标准 Floer--Gromov compactness、指数
衰减与 gluing 分析。本书证明命题 E.4，但不重建此定理的 Fredholm、
transversality、compactness、orientation 和 gluing 证明。

**推论 E.7（边界计数给出 $A_\infty$ 恒等式）.** 用 expected dimension
$0$ 的解定义 $b_d$，则 $b_d$ 满足公式 (B.3)。

**证明.** 紧定向 $1$-manifold 的边界带符号计数为零。由定理 E.6(3)、
(5)，其全部边界点恰为 (E.6)；由 E.6(4)，每个 stratum 的符号恰为
(B.3) 的符号。各 stratum 的两层计数是相应嵌套 $b_{r+1+t}\circ b_s$
的系数，所以总和为零。证毕。

## E.6 适用边界与反例

**警告 E.8.** 下列替换都不由定理 E.6 覆盖。

- 把 compact branes 换成 conical branes 并让 Hamiltonian 在无穷远增长，
  需要 wrapped maximum principles、action completion 和 continuation
  direct limits。
- 去掉 exactness 后，positive-area disk bubbles 可成为 codimension-one
  边界；必须使用 monotone 或 filtered curved/Novikov theory。
- 对非正则多重覆盖曲线，不能只写“取 generic $J$”；需 virtual
  perturbation package。
- 在 $\operatorname{char}k\ne2$ 时删去 relative Pin/orientation lines，
  只能得到 mod-$2$ 计数，不能随后把这些计数无来源地视为 $k$ 中的
  signed coefficients。

## 本附录小结

Compact exact Fukaya category 的几何输入不是单一 compactness 断言，而是
Fredholm、regularity、energy、no-escape、compactification、orientation 与
gluing 的相容包。Exactness 排除非恒定 sphere/disk bubbles；真正的
$A_\infty$ 符号来自 determinant-line gluing，并与附录 B 的 suspended
恒等式逐项匹配。
