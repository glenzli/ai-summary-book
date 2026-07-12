# 定理账本：内部证明、外部输入与研究边界

本文件把当前《Chromatic Homotopy Theory》草稿中的结论分成三类：

1. **内部证明**：正文已经给出稳定范畴或代数层面的证明。
2. **外部输入**：正文可引用，但必须在资料源中定位；locator 的完成度
   按本文件第 6 节和 source index 记录。
3. **边界说明**：只用于范围、失败模式或研究方向，不得作为证明步骤。

## 1. 基础稳定范畴

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第一章 | finite spectra compact/dualizable；localization 泛性质、exact/幂等与消失判据；有限 dualizable base change；嵌套 acyclic 类的两种复合公式 | Bousfield localization 存在性的一般定理 | base change 只要求有限输入，不能升级成 $L_E$ smashing；不同文献 Bousfield 偏序方向不同 |
| 第五章 | $L_m,L_n$ 嵌套方向；$M_nX$ 的 $L_n$-local/$L_{n-1}$-acyclic 性；tower 余项的 holim fiber；在已列外部输入下证明有限 type $n$ 谱满足 $M_nF\simeq L_nF\simeq L_{K(n)}F$ | $\langle E(n)\rangle=\langle\bigvee_{i=0}^nK(i)\rangle$ 与 Morava K 正交性；Hopkins--Ravenel smash product theorem；Milnor exact sequence；全体谱上的 chromatic fracture square；有限谱 chromatic convergence | 一般 $M_nX\ne L_{K(n)}X$；一般谱不自动 convergence；convergence 不给有限高度或 uniform error bound |

## 2. 形式群和复定向

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 序章/第二章 | 复定向给形式群律的公理检查；$K(n)_*$ 是 graded field | projective bundle formula；Quillen theorem；$BP$ splitting；Landweber exactness | 系数环商不自动给结构化 ring quotient |
| 附录 A | 加法/乘法形式群公理；$p$-series 计算；乘法形式群高度一；坐标改变保持高度 | 无新增外部输入 | Hazewinkel 坐标不可与任意坐标混用 |

## 3. Morava theories

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第三章 | 零谱被所有 $K(n)$ 检测；Bousfield 等价推出相同 acyclic 类 | finite spectra 被 Morava K theories 检测；Lubin-Tate deformation；Goerss-Hopkins-Miller；$E_n/\mathfrak m$ 与 $K(n)$ Bousfield 等价 | $E_n$ 的 $\mathbb E_\infty$ 结构不能由系数环推出 |
| 第六章 | $K(n)$-local 对等价封闭；$S_{K(n)}\simeq E_n^{hG_n}$ 由 descent 取 $X=S$ 推出 | Devinatz-Hopkins homotopy fixed point theorem；Morava descent spectral sequence | profinite group action 不能离散化 |

## 4. 有限谱和周期性

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第四章 | type 的较低高度消失由最小值定义直接推出；$\mathcal C_n$ 是 thick 子范畴；由 Serre finiteness 与 DHS 推出 Nishida nilpotence；$M(p)$ 对每个素数均由长正合列算得 type $1$ | finite detection；有限谱高度单调性；Serre finiteness；DHS I Theorem 1(i)；Hopkins--Smith II Theorem 7、Theorem 9、Corollaries 3.7/3.8、Theorem 14 | type 是首次非消失高度；更高 $K(m)$ 均非零属于外部单调性；$v_0$ 次数为零，不能套用 $n\ge1$ 定义 |

## 5. 前沿和近期研究

| 范围 | 内部证明 | 外部输入 | 边界说明 |
| --- | --- | --- | --- |
| 第七章 | redshift 假设清单；telescope 旧表述改写规则 | Hahn-Wilson redshift；Chromatic Nullstellensatz；cyclotomic redshift/descent | BHLŠ telescope 反例改变默认口径；2026 syntomic/K-theory 预印本暂作前沿 |
| 第八章 | ordinary/supersingular 的逐点高度分层；tmf 与 $E_2$ 类型区别说明 | 椭圆曲线形式群高度；Hasse invariant 判别；tmf sheaf 构造；Goerss-Hopkins-Miller/Lurie derived moduli | tmf 不是单个 Morava E-theory |
| 附录 B | spectral sequence 使用检查表 | Adams-Novikov、Morava descent、Tate spectral sequences 的收敛定理 | hidden extensions 不可忽略 |
| 第九章 | 0-semiadditivity；有限集合 cardinality 标量 | Hopkins-Lurie/CSY higher semiadditivity；HKR character theory；Ben-Moshe transchromatic/semiadditive results | redshift 不自动推出 semiadditivity |
| 第十章 | splitting square 退化判据；Picard/exotic 定义 | rational $K(n)$-local sphere；Gross-Hopkins duality；Picard descent | dualizing object convention 不可省略 |
| 第十一章 | genuine/geometric fixed point 基础检查 | equivariant Balmer/formal group laws；Behrens-Carlisle equivariant periodicity；motivic $MGL$ 基础；synthetic reconstruction | naive/genuine、topological/motivic 不可混用 |
| 第十二章 | 计算检查表；Tate construction 定义 | Adams-Novikov SS；chromatic SS；Morava change-of-rings；HFPSS/Tate SS | $E_\infty$ 页不等于 abutment |
| 附录 C | invariant ideal quotient 的结构映射下降；comodule Ext 口径 | quotient flatness；$BP_*BP$ 结构公式；change-of-rings | Hopf algebroid Ext 不是普通环 Ext |
| 附录 E | failure modes 清单 | 无新增定理 | 作为审稿约束 |
| 附录 F | 高度 0 有理化；Moore spectrum 的 $K(0)$ 消失与 type $1$ 书内计算 | $v_1$ 周期性、低高度 $K(1)$ 模型、tmf 局部模型 | type 判定不再错误归给 periodicity theorem；具体周期仍为外部输入 |
| 附录 G | frontier 升级协议 | 无新增定理 | 约束预印本使用 |
| 附录 H | acyclics localizing；locals 对 limits 封闭；smashing tensor ideal；接受实现输入后按齐次基分解 $K(n)$-modules | Hopkins--Smith II, Propositions 1.4/1.5；stable Whitehead theorem | field-like 不适用于普通谱 |
| 附录 I | telescope 的 filtered colimit 计算；较低高度消失；$T(0)\simeq H\mathbb Q$ convention | periodicity Theorem 9；Corollaries 3.7/3.8；finite-spectrum class invariance Theorem 14 | $v_0$ 与正次数 $v_n$ 分开；$T(n)\ne K(n)$ 默认 |
| 附录 J | Morava module semilinear action；descent spectral sequence 目标推导 | Devinatz-Hopkins；Morava change-of-rings | 连续性不可省略 |
| 附录 K | elliptic/tmf/version convention | tmf sheaf；supersingular local model；power operations | weak elliptic datum 不给 $\mathbb E_\infty$ sheaf |
| 附录 L | GH/Picard convention template | Gross-Hopkins duality；Picard group computations | 未定位公式不得简化 |
| 附录 M | hidden extension 定义；计算协议 | ANSS 低阶表来源 | 模板不是计算结果 |
| 附录 N | 任意谱的高度 0 有理化；全体谱的 height-1 fiber 公式；有限 $M(p)$ 的 overlap 消失、type $1$ 与 $L_1\simeq L_{K(1)}$ | $v_1$ 具体周期；$K(1)$ Adams operation model；$K(2)$ tmf local model | $L_0X=0$ 对非有限 $X$ 不足以删除 overlap；低高度公式需 convention |
| 出版闭包矩阵 | 内容范围和内部完整性判定 | 无新增定理 | 指导后续收口 |

## 6. Locator 状态

本轮核验的十组外部输入如下。完整书目信息与稳定 URL 见 `SOURCES.md`，
量词和禁止用途见两份 `P0_REFERENCE_LOCATORS_BATCH_*.md`。

| 编号 | 外部输入 | 一手精确 locator | 证明角色 |
| --- | --- | --- | --- |
| CHT-P0-01 | Quillen universal FGL | Quillen, Theorem 2, pp. 1294--1295 | 主线 P0 |
| CHT-P0-02 | $BP$ 的 $p$-typical summand/coefficients | Quillen, Theorem 4, pp. 1296--1297；Ravenel, Theorem 4.1.12(c), p. 108、Theorem 4.1.18(a), p. 111、Theorem A2.1.25, p. 349、(A2.2.1)/Theorem A2.2.3, pp. 354--355 | 主线 P0 |
| CHT-P0-03 | Landweber exact functor | Landweber, Theorem 2.6 与 Corollary 2.7；Ravenel, Chapter 4, Section 2, pp. 115--116 | 主线 P0 |
| CHT-P0-12 | Goerss--Hopkins--Miller | Goerss--Hopkins, Section 7, Proposition 7.1、Corollaries 7.6--7.7, pp. 198--199 | 主线 P0 |
| CHT-P0-13、CHT-P0-14 | Devinatz--Hopkins fixed points/descent | Devinatz--Hopkins, Theorem 1(iii)--(iv), pp. 3--4、Definition 1.5, p. 4、Theorem 2(i)--(ii), p. 5、Proposition 6.7, pp. 34--35 | 主线 P0 |
| CHT-P1-10 | HKR characters | Hopkins--Kuhn--Ravenel, Theorem C, pp. 557--558；Sections 6.3--6.4, pp. 583--586；Theorem D, p. 558 | 非主线 P1，已定位 |
| CHT-P0-19、CHT-P0-20、CHT-P1-21 | elliptic spectra/tmf | Silverman, Chapter IV, Theorem 7.4/Corollary 7.5, p. 134；Ando--Hopkins--Strickland, Definition 1.2、Definition 2.40、Corollary 2.50、Theorem 2.53；Goerss, Theorem 1.2, pp. 224--225 与 Definition 1.3, p. 225 | height/tmf 构造为 P0；orientation 扩展为 P1 |
| CHT-P1-12 | Gross--Hopkins duality | Strickland, Proposition 1, pp. 1021--1022、Theorem 2, p. 1022、Theorem 20 | 非主线 P1，已定位 |
| CHT-P1-13、CHT-P1-14 | Picard/profinite descent | Mor, arXiv:2306.05393v2, Theorem A、Proposition 3.21、Corollary 3.24、Theorem 4.4；Goerss--Henn--Mahowald--Rezk, Theorems 1.1--1.2 | 非主线 P1，已定位 |
| CHT-P1-17、CHT-P1-22 | ANSS/chromatic spectral sequence | Ravenel, Theorem 4.4.1, p. 130；Definition 5.1.7、Proposition 5.1.8、Definition 5.1.10, p. 150 | 计算接口 P1，已定位 |

此前已闭合的 finite chromatic 主链继续采用：DHS I, Theorem 1(i)；
Hopkins--Smith II, Theorems 7、9、14 与 Corollaries 3.7、3.8；Ravenel,
Theorems 7.5.6、7.5.7 与 Section 8.6；Lurie Lecture 23, Proposition 2、
Theorem 4、Proposition 5。

Hovey--Strickland 中未被正文逐项调用的全书级性质、近期 telescope/redshift、
higher semiadditivity 以及 equivariant/motivic 扩展统一归入非主线 P1 或
Frontier。它们只有在登记具体命题、版本和假设后才能进入新的证明步骤。

## 7. 当前状态

当前草稿达到“教材内容基本收口稿”状态。本轮涉及的主线 P0 外部输入已有
精确 locator，列出的 P1 接口也具有一手定位但不承担主线证明。出版级扩展仍
包括更丰富的计算样例、完整 convention tables，以及另行立项的
equivariant/motivic 和近期前沿内容；这些项目不改变当前 P0 闭合判定。
