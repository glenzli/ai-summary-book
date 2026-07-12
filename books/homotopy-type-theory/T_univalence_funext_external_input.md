# 附录 T：单值性推出函数外延性的外部输入

## T.0 目标

本附录处理第六章定理 6.11 的证明边界：在同一个单值 universe 内，univalence 蕴含依赖函数外延性。正文为了保持前几章依赖清晰，仍把层级多态函数外延性单独列为基本外延原则；本附录说明若采用“只假设单值性”的口径，同一 universe 内的实例可作为外部数学定理引入，而不需要额外公理。

**外部定理 T.0.1（同一 universe 内，单值性推出依赖函数外延性）.** 设 $A:\mathcal U_i$、$P:A\to\mathcal U_i$，并假设同时容纳基底和 fibers 的宇宙 $\mathcal U_i$ 满足 universe univalence $\mathsf{UA}_i$。则对任意
$$
f,g:\prod_{a:A}P(a),
$$
规范映射
$$
\mathsf{happly}_{f,g}:(f=g)\to\prod_{a:A}f(a)=g(a)
$$
是等价。

结论类型位于 $\mathcal U_i$，而作为对 $A$ 与宇宙值族 $P$ 的量化声明，T.0.1 的公理类型位于更高 universe。这里没有使用累积性、resizing 或隐式 lift。特别地，T.0.1 不声称仅由 $\mathsf{UA}_j$ 就能处理 $A:\mathcal U_i$、$P:A\to\mathcal U_j$ 且 $i\ne j$ 的混合层级族。

## T.1 陈述层级

**定义 T.1.1（普通函数外延性）.** 对 $X:\mathcal U_i$、$Y:\mathcal U_j$ 和 $f,g:X\to Y$，逐点路径
$$
\prod_{x:X}f(x)=g(x)
$$
推出函数路径 $f=g$。

**定义 T.1.2（依赖函数外延性）.** 对 $A:\mathcal U_i$、依赖族 $P:A\to\mathcal U_j$ 和截面 $f,g:\prod_{a:A}P(a)$，逐点路径
$$
\prod_{a:A}f(a)=g(a)
$$
推出截面路径 $f=g$。

**定义 T.1.3（强函数外延性）.** 对所有依赖族和截面，$\mathsf{happly}_{f,g}$ 是等价。

强函数外延性蕴含依赖函数外延性，方法是取 $\mathsf{happly}_{f,g}$ 的逆。第六章使用的正是强形式。

## T.2 证明路线

本书不把该外部输入伪装成压缩的书内证明。所引用来源把证明分成以下两个精确结果。

**外部步骤 T.2.1（弱函数外延性）.** 固定单值宇宙 $\mathcal U_i$。HoTT Book 定义 4.9.1 的弱函数外延性断言：对 $A:\mathcal U_i$ 和 $P:A\to\mathcal U_i$，若每个 $P(a)$ 可收缩，则 $\prod_{a:A}P(a)$ 可收缩。定理 4.9.4 从 $\mathsf{UA}_i$ 推出这个同宇宙原则。

**外部步骤 T.2.2（弱形式推出强形式）.** HoTT Book 定理 4.9.5 在不再使用单值性的前提下，从上述弱函数外延性推出：对 $A:\mathcal U_i$、$P:A\to\mathcal U_i$ 与截面 $f,g$，$\mathsf{happly}_{f,g}$ 是等价。

**来源与未重证边界.** T.0.1 是 HoTT Book 第 4.9 节、尤其定理 4.9.4 与 4.9.5 的合成。本书在此只记录精确输入、层级和依赖，不重证其中关于 pointed universe、retract 和 contractible total space 的长构造。因此后文只能把 T.0.1 作为外部输入使用，不能把 T.2.1-T.2.2 称为书内证明。

## T.3 本书采用方式

本书允许两种一致阅读：

1.  **公理化 HoTT 阅读。** 同时声明函数外延性与单值性。此时 T.0.1 只是说明同一 universe 内的函数外延性实例在更小公理组中可导出。
2.  **最小单值阅读。** 只把单值性作为外延原则，并引用 T.0.1 得到每个 $\mathcal U_i$ 内部的函数外延性。

正文采用第一种阅读，以便把路径代数、等价和单值性的使用位置分开审查。若读者采用第二种阅读，只有基底与 fibers 同属一个 $\mathcal U_i$ 的函数外延性结果可把 T.0.1 加入依赖链；混合层级结果还需要显式函数外延性实例或已证明正确的 lift。第一章的非累积口径不提供这种隐式 lift。

## T.4 引用边界

T.0.1 是关于同一个 universe $\mathcal U_i$ 中基底与 fibers 的 universe univalence 定理，不能由 categorical univalence、directed univalence 或某个模型中的弱相似原则替代。凡引用“单值性推出函数外延性”，必须确认所用单值性是第六章的类型等价到 universe path 的强形式，并确认基底与 fibers 都属于它；否则必须显式记录 lift 或额外的函数外延性实例。
