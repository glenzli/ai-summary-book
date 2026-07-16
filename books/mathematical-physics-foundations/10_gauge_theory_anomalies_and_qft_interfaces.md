# 第十章：规范量子场论、异常与几何接口

规范理论把第四章的联络语言推到场论中心：场强是曲率，作用量是曲率范数，物质场是伴随或关联丛截面。量子化后，规范冗余要求固定规范或转向 BRST 复形；某些经典对称性还会因正规化而失效，形成异常。本章只建立 Yang-Mills 变分、BRST 局部形式和异常-指标接口，不声称完成非微扰规范场论构造。

## 10.1 Yang--Mills 几何

**定义 10.1.** 设 $X$ 是定向 Riemannian $d$-流形，$G$ 是带 $\operatorname{Ad}$-不变内积 $\operatorname{tr}$ 的紧矩阵 Lie 群，$P\to X$ 是主 $G$-丛。对联络 $A$，Yang--Mills 作用量为
$$
S_{\rm YM}[A]=\frac1{2g^2}\int_X \operatorname{tr}(F_A\wedge *F_A).
$$

**命题 10.1 (`P`).** 规范变换下 $S_{\rm YM}$ 不变。

**证明.** 命题 4.2 给出 $F_{A^g}=g^{-1}F_Ag$。Hodge 星只作用在微分形式指标，故 $*(g^{-1}F_Ag)=g^{-1}(*F_A)g$。于是 $\operatorname{Ad}$-不变性给出
$$
\operatorname{tr}(F_{A^g}\wedge *F_{A^g})
=\operatorname{tr}(g^{-1}F_Ag\wedge g^{-1}(*F_A)g)
=\operatorname{tr}(F_A\wedge *F_A).
$$
两边在 $X$ 上积分即得作用量不变。$\square$

**命题 10.2 (`P`).** 若 $X$ 无边界，或只取紧支撑联络变分，则 Yang--Mills 作用量的临界联络满足
$$
D_A*F_A=0.
$$

**证明.** 取 $A_\epsilon=A+\epsilon a$，其中 $a\in\Omega^1(X,\operatorname{ad}P)$。由曲率定义，
$$
F_{A+\epsilon a}=F_A+\epsilon D_Aa
+\frac{\epsilon^2}{2}[a\wedge a],
$$
故 $\delta F_A=D_Aa$。$\operatorname{tr}$ 对称且 Hodge 星线性，一阶变分为
$$
\delta S_{\rm YM}=\frac1{g^2}\int\operatorname{tr}(D_Aa\wedge *F_A).
$$
不变内积给出
$$
d\operatorname{tr}(a\wedge *F_A)
=\operatorname{tr}(D_Aa\wedge *F_A)
-\operatorname{tr}(a\wedge D_A*F_A).
$$
积分后的全微分项由无边界或紧支撑条件消失。因此
$$
\delta S_{\rm YM}
=\frac1{g^2}\int_X\operatorname{tr}(a\wedge D_A*F_A).
$$
若 $D_A*F_A$ 在某点非零，可在局部平凡化中选取紧支撑 $a$ 与其配对，使积分非零；故临界性对所有 $a$ 成立当且仅当 $D_A*F_A=0$。$\square$

**例 10.3（Abelian 极限）.** 对平凡 $U(1)$-丛，Lie 括号为零，故 $F=dA$、$D_A=d$。在四维 Minkowski 坐标中取
$S[A]=-\frac14\int F_{\mu\nu}F^{\mu\nu}\,d^4x$。紧支撑变分给出
$$
\delta S
=-\frac12\int F^{\mu\nu}
(\partial_\mu\delta A_\nu-\partial_\nu\delta A_\mu)\,d^4x
=\int(\partial_\mu F^{\mu\nu})\delta A_\nu\,d^4x,
$$
所以场方程是 $\partial_\mu F^{\mu\nu}=0$。Bianchi 恒等式 $dF=0$ 给出另两条齐次 Maxwell 方程；规范变换 $A\mapsto A+d\lambda$ 因 $d^2=0$ 不改变 $F$。

## 10.2 BRST 局部形式

**定义 10.2.** 在局部规范固定中，引入 ghost 场 $c$，BRST 变换形式写作
$$
sA=D_Ac,\qquad sc=-\frac12[c,c].
$$

**命题 10.3 (`S`).** BRST 变换满足 $s^2A=0$ 与 $s^2c=0$ 的形式幂零性。

**推导说明（标准物理口径）.** 对 $A$，
$$
s^2A=s(D_Ac)=D_A(sc)+[sA,c]
=-\frac12D_A[c,c]+[D_Ac,c]=0
$$
其中使用分次 Jacobi 恒等式。对 $c$ 同理。完整量子 BRST cohomology 还依赖规范固定、ghost 数、正规化和物理态条件，本书不重证。$\square$

## 10.3 异常和指标

**定义 10.3.** 若经典对称性在量子正规化后不能同时保持作用量和测度不变，则称该对称性有 anomaly。

**命题 10.4 (`S`, Fujikawa 形式).** 手征变换下费米子路径积分测度的 Jacobian 可产生
$$
\partial_\mu j_5^\mu
=\frac{1}{16\pi^2}\operatorname{tr}(F_{\mu\nu}\widetilde F^{\mu\nu})
$$
型异常项。

**推导说明（标准物理口径）.** 将费米场展开为 Euclidean Dirac 算符的本征函数，手征旋转使 Grassmann 形式测度获得无限维 Jacobian。用热核因子 $e^{-D^2/\Lambda^2}$ 正则化迹，在保持规范协变的 regulator 下取 $\Lambda\to\infty$，热核的局部系数产生所示密度。公式假设四维、无质量 Dirac 费米子和给定生成元归一化；不同表示带来相应群论迹，若 regulator、边界或手征电流定义改变，还需加入边界项或局部 counterterm。这里的无穷维 Jacobian 不是书内构造的测度定理。$\square$

**定理 10.5 (`E`, Atiyah-Singer 指标定理).** 闭偶维 spin 流形 $X$ 上、由复向量丛 $E$ 扭曲的正手征 Dirac 算符 $D_E^+$ 的指标等于特征类乘积的最高次分量积分：
$$
\operatorname{ind}D_E^+=\int_X \bigl[\widehat A(TX)\operatorname{ch}(E)\bigr]_{\dim X}.
$$

**外部输入边界.** 本书使用该定理连接零模计数、手征异常和拓扑荷；不证明椭圆算子与 K 理论部分。所用闭 spin 流形与扭 Dirac 算子版本见 [SOURCES.md](SOURCES.md) 的 `E-10.5`。

## 练习

**练习 10.1.** 对 Abelian 规范理论从 $S=-\frac14\int F_{\mu\nu}F^{\mu\nu}d^dx$ 推出 Maxwell 方程。

**练习 10.2.** 验证 BRST 变换 $sA=D_Ac$ 与规范变换的无穷小形式一致，只是参数换成 ghost。
