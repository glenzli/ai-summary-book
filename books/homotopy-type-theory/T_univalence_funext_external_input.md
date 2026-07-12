# 附录 T：单值性推出函数外延性的外部输入

## T.0 目标

本附录处理第六章定理 6.11 的证明边界：在单值基础中，univalence 蕴含依赖函数外延性。正文为了保持前几章依赖清晰，仍把函数外延性单独列为基本外延原则；本附录说明若采用“只假设单值性”的口径，函数外延性可作为外部数学定理引入，而不需要额外公理。

**外部定理 T.0.1（单值性推出依赖函数外延性）.** 设 $A:\mathcal U_i$、$P:A\to\mathcal U_j$，并假设分类 fibers $P(a)$ 的宇宙 $\mathcal U_j$ 满足 universe univalence $\mathsf{UA}_j$。则对任意
$$
f,g:\prod_{a:A}P(a),
$$
规范映射
$$
\mathsf{happly}_{f,g}:(f=g)\to\prod_{a:A}f(a)=g(a)
$$
是等价。

基底 $A$ 不必属于 $\mathcal U_j$。结论类型位于 $\mathcal U_{\max(i,j)}$，而作为对 $A$ 与 $P$ 的层级多态外部定理，其整体声明位于更高宇宙；这里没有使用累积性或 resizing。

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

**外部步骤 T.2.1（弱函数外延性）.** HoTT Book 定义 4.9.1 的弱函数外延性断言：对任意基底 $A$ 和 $P:A\to\mathcal U_j$，若每个 $P(a)$ 可收缩，则 $\prod_{a:A}P(a)$ 可收缩。定理 4.9.4 从 $\mathsf{UA}_j$ 推出该原则。

**外部步骤 T.2.2（弱形式推出强形式）.** HoTT Book 定理 4.9.5 在不再使用单值性的前提下，从弱函数外延性推出：对所有 $P:A\to\mathcal U_j$ 与截面 $f,g$，$\mathsf{happly}_{f,g}$ 是等价。

**来源与未重证边界.** T.0.1 是 HoTT Book 第 4.9 节、尤其定理 4.9.4 与 4.9.5 的合成。本书在此只记录精确输入、层级和依赖，不重证其中关于 pointed universe、retract 和 contractible total space 的长构造。因此后文只能把 T.0.1 作为外部输入使用，不能把 T.2.1-T.2.2 称为书内证明。

## T.3 本书采用方式

本书允许两种一致阅读：

1.  **公理化 HoTT 阅读。** 同时声明函数外延性与单值性。此时 T.0.1 只是说明函数外延性在更小公理组中可导出。
2.  **最小单值阅读。** 只把单值性作为外延原则，并引用 T.0.1 得到函数外延性。

正文采用第一种阅读，以便把路径代数、等价和单值性的使用位置分开审查。若读者采用第二种阅读，所有依赖函数外延性的结果应把 T.0.1 加入依赖链。

## T.4 引用边界

T.0.1 是关于 fiber universe $\mathcal U_j$ 的 universe univalence 定理，不能由 categorical univalence、directed univalence 或某个模型中的弱相似原则替代。凡引用“单值性推出函数外延性”，必须确认所用单值性是第六章的类型等价到 universe path 的强形式，并记录它作用的 fiber universe。
