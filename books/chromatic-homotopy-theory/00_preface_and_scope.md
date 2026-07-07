# 序章：范围、严格性标准和资料源

## 本章目标

本章说明本书的对象、约定、资料源和严格性标准。读者在进入第一章前，应当知道本书中“chromatic homotopy theory”默认研究什么，哪些定理属于可在本书内部证明的稳定范畴形式事实，哪些定理属于 Hopkins-Smith/Ravenel/Goerss-Hopkins-Miller 体系的外部输入，哪些内容只是 2026 年前沿边界。

## 依赖前置知识

需要熟悉范畴、函子、自然变换、极限、余极限、三角范畴或稳定 infinity-范畴的基本语言。复 cobordism、形式群、Adams-Novikov 谱序列和 Morava theory 不预设，但后续章节会快速进入严格定义。

## 0.1 本书研究的对象

**约定 0.1.** 全书固定素数 $p$。除非特别说明，所有有限谱、局部化和高度均在 $p$-局部稳定同伦论中讨论。

**约定 0.2.** 本书默认的基础范畴是谱的稳定 infinity-范畴 $\mathbf{Sp}$ 及其 $p$-局部子范畴 $\mathbf{Sp}_{(p)}$。若使用模型范畴，如 symmetric spectra、orthogonal spectra 或 $S$-modules，该模型只服务于构造和引用，最终陈述在稳定 infinity-范畴中表达。

**定义 0.3.** Chromatic homotopy theory 在本书中指如下研究纲领：用复定向同调理论给出的形式群高度、Morava K-theory $K(n)$、Johnson-Wilson theory $E(n)$ 和相关 Bousfield localization 来分解和研究稳定谱，尤其是有限 $p$-局部谱与球谱。

这个定义故意不把 chromatic theory 限定为计算稳定 stems。计算是重要目标，但本书的主线是结构：

1. 稳定谱的 Bousfield localization；
2. 复定向和形式群高度；
3. 有限谱的 type 和 $v_n$-周期性；
4. chromatic tower 和 monochromatic layers；
5. $K(n)$-局部范畴和 Morava stabilizer group descent；
6. telescope、redshift、higher semiadditivity 和近期前沿。

## 0.2 严格性标准

**约定 0.4.** 本书采用如下证明标准。

- 稳定范畴形式命题必须给出完整证明，至少写出 fiber/cofiber 序列、localization 泛性质或 compactness 使用点。
- 复定向和形式群相关命题必须说明坐标、次数和完备化。
- $K(n)$、$E(n)$、$E_n$ 的系数和高度必须在固定素数 $p$ 下书写。
- nilpotence、periodicity、thick subcategory theorem、chromatic convergence、Goerss-Hopkins-Miller theorem 和 Devinatz-Hopkins descent 不在正文重证，作为外部输入定理引用。
- 2023-2026 研究结果除非完成 theorem locator 和假设翻译，只作为研究边界记录。

**例 0.5.** “$K(n)$ 是高度 $n$ 的场”不是严格陈述。严格写法至少要拆成两部分：

1. $K(n)_*\cong \mathbb F_p[v_n^{\pm1}]$ 是 graded field；
2. 对 $K(n)$-module spectrum $M$，若采用相应 module category 的标准结果，则 $M$ 由其 graded homotopy groups 控制，特别可分解为若干悬挂的 $K(n)$ 的 wedge。第二点依赖 module category 结构，不是系数环为 graded field 的纯代数自动结论。

## 0.3 为什么高度从形式群出现

若 $E$ 是复定向 multiplicative cohomology theory，选择复定向
$$
x\in E^2(\mathbb CP^\infty)
$$
后，有同构
$$
E^*(\mathbb CP^\infty)\cong E^*[[x]].
$$
张量积线丛给出乘法
$$
\mu:\mathbb CP^\infty\times \mathbb CP^\infty\to \mathbb CP^\infty,
$$
于是
$$
\mu^*x=F_E(x_1,x_2)\in E^*[[x_1,x_2]]
$$
定义一个形式群律。Quillen 的核心发现是：$MU$ 上的这个形式群律是 universal formal group law。chromatic filtration 正是从形式群在特征 $p$ 下的高度分层进入稳定同伦论。

**命题 0.6.** 若 $E$ 是复定向 multiplicative cohomology theory，则上式定义的 $F_E$ 满足一维交换形式群律公理。

**证明.** 单位公理由平凡线丛 $1$ 满足 $L\otimes 1\cong L$ 得到：对应到 classifying map 上，$\mu$ 与基点包含的复合诱导 $F_E(x,0)=x$ 和 $F_E(0,x)=x$。结合律由线丛张量积的自然同构
$$
(L_1\otimes L_2)\otimes L_3\cong L_1\otimes (L_2\otimes L_3)
$$
推出，即两个从 $(\mathbb CP^\infty)^3$ 到 $\mathbb CP^\infty$ 的 classifying maps 同伦，从而
$$
F_E(F_E(x_1,x_2),x_3)=F_E(x_1,F_E(x_2,x_3)).
$$
交换律由 $L_1\otimes L_2\cong L_2\otimes L_1$ 推出。所有等式都发生在 $E^*[[x_1,x_2,x_3]]$ 中。证毕。

## 0.4 外部输入与研究边界

本书将大型结果分成三类。

**外部输入 0.7.** Hopkins-Smith nilpotence/periodicity/thick subcategory theorem、Hopkins-Ravenel chromatic convergence、Goerss-Hopkins-Miller theorem、Devinatz-Hopkins homotopy fixed point theory，是本书的基础外部输入。正文可以使用，但必须在 `THEOREM_LEDGER.md` 中登记。

**边界 0.8.** Telescope conjecture 在 2023 年之后不能作为假设外的默认事实使用。Burklund-Hahn-Levy-Schlank 的结果表明在每个素数且高度至少 $2$ 的相应层次，telescopic 与 chromatic localization 发生差异。本书会把旧文献中的 telescope conjecture 表述改写为历史命题、条件命题或失败模式。

**边界 0.9.** Redshift、higher semiadditivity、transchromatic character、rational $K(n)$-local sphere 和 2026 年 syntomic/K-theory of $BP\langle n\rangle$ 结果均进入前沿章。除非完成精确定位和假设翻译，不用于证明基础章节命题。

## 本章小结

本书把 chromatic homotopy theory 作为稳定同伦论中的严格结构理论处理：先固定谱、局部化和复定向口径，再进入形式群高度、Morava theories、有限谱 type、chromatic tower 和 $K(n)$-局部范畴。大型定理作为外部输入清楚标注；近期前沿纳入资料边界，但不削弱正文证明链。

## 练习

**练习 0.1.** 说明为什么在讨论 $K(n)$ 时必须固定素数 $p$。要求写出 $|v_n|$ 如何依赖 $p$。

**练习 0.2.** 证明线丛张量积的交换性给出形式群律的交换律。要求把 classifying map 和 $E^*(\mathbb CP^\infty\times\mathbb CP^\infty)$ 的坐标写出。

**练习 0.3.** 给出一个陈述，其中 $K(n)$-local 和 $E(n)$-local 不能互换。要求说明两个 localization functor 的定义差异。
