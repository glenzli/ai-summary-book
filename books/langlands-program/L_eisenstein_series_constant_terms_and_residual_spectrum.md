# 附录 L：Eisenstein Series、常数项和残余谱

本附录补充第十三、十六、十七章中的谱分解接口。尖点谱只是自守谱的一部分；一般自守谱还包括由 parabolic induction 和 Eisenstein series 产生的连续谱与残余谱。

**收口归一化回指。** 本附录涉及归一化诱导、intertwining operators、常数项、残余谱和 Eisenstein L 因子；与谱分解、迹公式和 Langlands-Shahidi 方法比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 3、4、6、8 节。

## L.1 Parabolic Data

设 $K$ 为整体域，$G/K$ 为 connected reductive group。设 $P=MN$ 为 parabolic subgroup，$M$ 为 Levi subgroup，$N$ 为 unipotent radical。

**定义 L.1.** 对 $\phi$ 为 $M(K)\backslash M(\mathbb A_K)$ 上的 cusp form，可通过 $P(\mathbb A_K)$ 的投影把它扩展到 $P(K)\backslash G(\mathbb A_K)$ 上的 section。若 $\lambda\in\mathfrak a_{M,\mathbb C}^*$，记扭曲 section 为
$$
\phi_\lambda(g)=\phi(g)e^{\langle\lambda,H_M(g)\rangle}.
$$
这里 $H_M$ 为 Harish-Chandra height map。

**定义 L.2.** Eisenstein series 定义为
$$
E(g,\phi,\lambda)
=
\sum_{\gamma\in P(K)\backslash G(K)}
\phi_\lambda(\gamma g)
$$
在绝对收敛区域中成立。

**外部输入定理 L.3（Eisenstein series 的初始收敛）.** 当 $\operatorname{Re}(\lambda)$ 位于正 Weyl chamber 足够深处时，$E(g,\phi,\lambda)$ 绝对收敛，并定义 automorphic form。

## L.2 常数项公式

设 $P'=M'N'$ 为另一个 parabolic subgroup。

**定义 L.4.** Eisenstein series 沿 $P'$ 的常数项为
$$
E_{P'}(g,\phi,\lambda)
=
\int_{N'(K)\backslash N'(\mathbb A_K)}
E(ng,\phi,\lambda)\,dn.
$$

**外部输入定理 L.5（Langlands constant term formula）.** Eisenstein series 的常数项可写为有限 Weyl 群和双陪集求和：
$$
E_{P'}(g,\phi,\lambda)
=
\sum_w M(w,\lambda)\phi_\lambda(g)
$$
的 Langlands constant term 版本，其中 $M(w,\lambda)$ 是 standard intertwining operator。精确求和集合依 $P,P'$ 和 associate parabolic classes 而定。

**定义 L.6.** Standard intertwining operator 在收敛区域中由积分
$$
M(w,\lambda)\phi(g)
=
\int_{N_w(\mathbb A_K)}\phi_\lambda(w^{-1}ng)\,dn
$$
定义，其中 $N_w$ 是由 $w$ 决定的 unipotent subgroup。解析延拓后得到 meromorphic operator。

**外部输入定理 L.7（Intertwining operators 的 meromorphic continuation）.** Operators $M(w,\lambda)$ 有 meromorphic continuation，并满足 cocycle relations。它们的 poles 控制 Eisenstein series 的 poles。

## L.3 连续谱和残余谱

**定义 L.8.** 由 parabolic induction of cuspidal data 产生的 Eisenstein series 在 unitary axis 上的积分贡献称为 continuous spectrum。

**定义 L.9.** Eisenstein series 或 intertwining operators 在 $\lambda$ 的极点处的 residues 生成的离散 $L^2$-自守表示部分称为 residual spectrum。

**外部输入定理 L.10（Langlands spectral decomposition，接口形式）.** $L^2(G(K)\backslash G(\mathbb A_K))$ 可按 cuspidal data 分解为：

1. Cuspidal discrete spectrum；
2. Residual discrete spectrum；
3. Continuous spectrum，由 Eisenstein series 的 unitary integrals 描述。

精确陈述需要固定中心特征、测度、截断算子和 Hilbert space completion。

**注 L.11.** 第十三章主要讨论 cuspidal automorphic representations，因为 Langlands 参数和 L 函数在该情形最直接。Arthur 参数和 trace formula 必须处理 residual spectrum，因此第十七章引入非 tempered Arthur 参数。

## L.4 `GL(2)` 的基本例子

设 $G=\operatorname{GL}_2$，$B=TN$ 为 Borel subgroup。令 $\chi_1,\chi_2$ 为 Hecke characters。

**定义 L.12.** 对 section $f_s$ in
$$
\operatorname{Ind}_{B(\mathbb A)}^{G(\mathbb A)}
(\chi_1|\cdot|^s\otimes\chi_2|\cdot|^{-s}),
$$
定义 Eisenstein series
$$
E(g,s,f)=\sum_{\gamma\in B(K)\backslash G(K)}f_s(\gamma g).
$$

**命题 L.13.** `GL(2)` Eisenstein series 的常数项沿 $B$ 具有形式
$$
E_B(g,s,f)=f_s(g)+M(s)f_s(g),
$$
其中 $M(s)$ 为标准 intertwining operator。

**证明路线（外部输入）.** 对
$$
B(K)\backslash G(K)/B(K)
$$
使用 Bruhat decomposition。`GL(2)` 的 Weyl group 有两个元素 $1$ 和 $w$。单位元轨道给出 $f_s(g)$，非平凡 Weyl 元轨道给出 intertwining integral $M(s)f_s(g)$。$\square$

**外部输入定理 L.14（`GL(2)` Eisenstein series 的解析性质）.** `GL(2)` Eisenstein series 有 meromorphic continuation 和函数方程。其 poles 由相关 Hecke L 函数的 poles 控制；特别地，平凡 character 的 zeta pole 产生 residual contribution。

## L.5 与 L 函数的关系

**外部输入定理 L.15（Intertwining operators 的归一化）.** 对许多 reductive groups，standard intertwining operators 可由 Langlands-Shahidi 或 Gindikin-Karpelevich 公式归一化，使归一化因子由 L 函数商给出。例如非分歧球向量上，
$$
M(w,\lambda)
$$
的标量可写为某些 Euler factors 的乘积。

**命题 L.16.** 若 Eisenstein series 的 pole 来自归一化因子中的 L 函数 pole，则其 residue 可产生 residual automorphic representation。

**证明路线（外部输入）.** Constant term formula L.5 把 Eisenstein series 的 meromorphic behavior 降到 intertwining operators。若某个 $M(w,\lambda)$ 在 $\lambda_0$ 有 pole，且该 pole 未被不同 Weyl 项抵消，则 $E(g,\phi,\lambda)$ 在 $\lambda_0$ 有 pole。取 residue 得到 automorphic form；若其 $L^2$ norm 有限，则它生成 residual spectrum 中的表示。$\square$

## L.6 Arthur 参数中的残余谱

**命题 L.17.** 非 tempered Arthur 参数可解释为 residual spectrum 中非 tempered 表示的组织语言之一。

**证明路线（外部输入）.** Arthur 参数
$$
\psi:L_K\times\operatorname{SL}_2(\mathbb C)\to{}^LG
$$
中非平凡 $\operatorname{SL}_2$ 因子对应偏离 tempered 的方向。Eisenstein residues 由 parabolic induction 和 intertwining operator poles 产生，通常不是 tempered。Arthur multiplicity formula 把这些 residual pieces 与含非平凡 $\operatorname{SL}_2$ 因子的参数配对。完整证明属于 Arthur 分类。$\square$

## L.7 Trace Formula 中的角色

**注 L.18.** Arthur trace formula 的谱侧不仅包含 cuspidal representations。它还包含由 Levi 子群 cuspidal data 诱导出的连续谱和 residual terms。截断 trace formula 的复杂性很大程度来自 Eisenstein series 的常数项和 intertwining operators。

**外部输入定理 L.19（Arthur truncation，接口形式）.** Arthur 截断算子 $\Lambda^T$ 使非紧自守商上的核函数积分可正则化。截断后的 trace formula 谱侧由离散谱、连续谱和 intertwining operators 的导数项组成。

## L.8 本附录小结

本附录建立如下接口：

1. Eisenstein series 由 Levi 上的 cuspidal data 构造。
2. 常数项公式由 Weyl 群和 standard intertwining operators 控制。
3. Intertwining operators 的 poles 给出 Eisenstein series 的 poles。
4. Residues 生成 residual spectrum。
5. Arthur 参数中的非平凡 $\operatorname{SL}_2$ 因子组织非 tempered residual phenomena。
6. Trace formula 谱侧必须处理 Eisenstein series 和截断。

## 练习

**练习 L.1.** 对 `GL(2)`，写出 Bruhat decomposition 并说明为何常数项有两个 Weyl 项。

**练习 L.2.** 解释为什么 cusp forms 的 Eisenstein series 输入来自 proper Levi，而不是直接来自 $G$ 本身。

**练习 L.3.** 在 `GL(2)` 情形，说明平凡 character 的 zeta pole 如何可能产生 residual spectrum。

**练习 L.4.** 比较 cuspidal spectrum、residual spectrum 和 continuous spectrum 的定义差异。

**练习 L.5.** 解释第十七章非 tempered Arthur 参数为什么需要一个额外的 $\operatorname{SL}_2(\mathbb C)$ 因子。
