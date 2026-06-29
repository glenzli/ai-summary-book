# 附录 B：局部紧群与 Haar 测度

收口归一化回指：本附录固定 Haar 测度、商测度、restricted product 测度、卷积和开紧子群体积 convention；与 Hecke 代数、Satake、Tate thesis 和 trace formula 比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4、8 节。

## B.1 局部紧群

**定义 B.1.** 拓扑群 $G$ 称为局部紧群，若其拓扑空间局部紧且 Hausdorff。若单位元有开紧子群基，则称 $G$ 为 locally profinite group。

**外部输入定理 B.2（Van Dantzig）.** Totally disconnected locally compact group 的单位元有开紧子群基。

**例 B.3.** 若 $F$ 为非 Archimedean 局部域，$G/F$ 为代数群，则 $G(F)$ 是 locally profinite group。

## B.2 Haar 测度

**外部输入定理 B.4（Haar 测度存在唯一性）.** 每个局部紧群 $G$ 有非零左 Haar 测度 $dg$，且在正实数倍下唯一。右 Haar 测度同理存在。

**定义 B.5.** Modular character $\Delta_G:G\to\mathbb R_{>0}$ 由
$$
\int_G f(xg)\,dx=\Delta_G(g)^{-1}\int_G f(x)\,dx
$$
定义。若 $\Delta_G=1$，称 $G$ unimodular。

**命题 B.6.** 紧群和 reductive $p$-adic groups $G(F)$ 是 unimodular。

**证明草图.** 紧群上左右 Haar 测度都可归一化为总体积 $1$，故相同。Reductive $p$-adic groups 的 unimodularity 可由代数群的 modular character 对 reductive 群平凡推出。$\square$

## B.3 Restricted Product 测度

设 $G_v$ 为局部紧群，$K_v\subset G_v$ 为几乎所有 $v$ 的开紧子群。

**定义 B.7.** Restricted product
$$
\prod_v'G_v
$$
由满足 $g_v\in K_v$ 对几乎所有 $v$ 成立的元组成。

若 $dg_v$ 满足 $\operatorname{vol}(K_v)=1$ 对几乎所有 $v$，则乘积测度
$$
\prod_v dg_v
$$
定义 restricted product 上的 Haar 测度。

## B.4 卷积与 Hecke 代数

**定义 B.8.** 对 locally profinite group $G$，Hecke 代数
$$
C_c^\infty(G)
$$
的乘法为卷积
$$
(f_1*f_2)(g)=\int_G f_1(x)f_2(x^{-1}g)\,dx.
$$

若 $J\subset G$ 为开紧子群，$J$-双不变 Hecke 代数为
$$
\mathcal H(G,J)=C_c^\infty(J\backslash G/J).
$$

**命题 B.9.** 若 $\operatorname{vol}(J)=1$，则 $e_J=\mathbf 1_J$ 是幂等元。

**证明.**
$$
(e_J*e_J)(g)=\int_G\mathbf 1_J(x)\mathbf 1_J(x^{-1}g)\,dx.
$$
当 $g\in J$ 时积分域为 $J$，值为 $1$；当 $g\notin J$ 时积分域为空。故 $e_J*e_J=e_J$。$\square$

**命题 B.10（卷积结合律）.** 设 $G$ 为 unimodular locally profinite group，$f_1,f_2,f_3\in C_c^\infty(G)$。则
$$
(f_1*f_2)*f_3=f_1*(f_2*f_3).
$$

**证明.** 对 $g\in G$，
$$
((f_1*f_2)*f_3)(g)
=\int_G\int_G f_1(y)f_2(y^{-1}x)f_3(x^{-1}g)\,dy\,dx.
$$
被积函数支撑包含在紧集的乘积中，故可用 Fubini 定理交换积分。令 $x=yz$。左 Haar 测度在左平移下不变，且 $G$ unimodular 保证此换元不引入 modular character，于是上式等于
$$
\int_G\int_G f_1(y)f_2(z)f_3(z^{-1}y^{-1}g)\,dz\,dy
=\int_G f_1(y)(f_2*f_3)(y^{-1}g)\,dy.
$$
这正是 $f_1*(f_2*f_3)(g)$。$\square$

**命题 B.11（开紧平均算子的像）.** 设 $(\pi,V)$ 为 $G$ 的 smooth representation，$J\subset G$ 为开紧子群且 $\operatorname{vol}(J)=1$。则
$$
\pi(e_J)V=V^J.
$$

**证明.** 对 $v\in V$，
$$
\pi(e_J)v=\int_J\pi(j)v\,dj.
$$
若 $j_0\in J$，则
$$
\pi(j_0)\pi(e_J)v
=\int_J\pi(j_0j)v\,dj
=\int_J\pi(j)v\,dj
=\pi(e_J)v,
$$
故像包含于 $V^J$。若 $v\in V^J$，则
$$
\pi(e_J)v=\int_Jv\,dj=v.
$$
因此 $V^J$ 包含于像。$\square$

## B.5 商测度

**外部输入定理 B.12（商测度）.** 设 $G$ 为局部紧群，$H\subset G$ 为闭子群。若 modular characters 满足
$$
\Delta_G|_H=\Delta_H,
$$
则存在 $G$-不变测度 $d\dot g$ on $H\backslash G$，并且可归一化使得对 $f\in C_c(G)$ 有
$$
\int_G f(g)\,dg
=\int_{H\backslash G}\int_H f(hg)\,dh\,d\dot g.
$$

**注 B.13.** 自守商 $G(K)\backslash G(\mathbb A_K)$ 的积分、尖点条件和 trace formula 都依赖商测度。离散子群情形中 $H$ 的 Haar 测度为计数测度，公式退化为基本区域积分。

**命题 B.14.** 若 $H\subset G$ 为离散闭子群且 $G$ unimodular，则 $H\backslash G$ 存在右 $G$-不变测度。

**证明草图.** 离散群 $H$ 的 modular character 平凡。若 $G$ unimodular，则 $\Delta_G|_H=1=\Delta_H$。由定理 B.12 得商测度。$\square$

## B.6 Restricted Product 的局部-整体换元

**命题 B.15.** 设 $G=\prod_v'(G_v,K_v)$，并取 Haar 测度 $dg_v$ 使 $\operatorname{vol}(K_v)=1$ 对几乎所有 $v$ 成立。若 $f=\otimes_v f_v$ 且 $f_v=\mathbf 1_{K_v}$ 对几乎所有 $v$，则
$$
\int_G f(g)\,dg=\prod_v\int_{G_v}f_v(g_v)\,dg_v,
$$
其中右侧乘积只有有限多个因子不同于 $1$。

**证明.** 设 $S$ 包含所有 $f_v\ne\mathbf 1_{K_v}$ 或测度未归一为 $\operatorname{vol}(K_v)=1$ 的位置。函数 $f$ 支撑在
$$
\prod_{v\in S}\operatorname{supp}(f_v)\times\prod_{v\notin S}K_v
$$
上，并在该有限乘积方向上等于普通张量积函数。restricted product 测度在该柱状集合上按
$$
\prod_{v\in S}dg_v\cdot\prod_{v\notin S}\operatorname{vol}(K_v)
$$
定义。由于后一个乘积为 $1$，公式化为有限维 Fubini 公式。$\square$

**推论 B.16.** 若 $J=\prod_v'J_v$ 为开紧 restricted product 且 $\operatorname{vol}(J_v)=1$ 对几乎所有 $v$，则
$$
\operatorname{vol}(J)=\prod_v\operatorname{vol}(J_v).
$$

**证明.** 对命题 B.15 取 $f_v=\mathbf 1_{J_v}$。$\square$
