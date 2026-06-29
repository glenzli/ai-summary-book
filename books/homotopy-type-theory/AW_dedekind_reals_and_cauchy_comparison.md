# 附录 AW：Dedekind 实数、Locatedness 与 Cauchy 实数比较

附录 AK-AR 以 Cauchy 实数 HIIT 为主。本附录补入 Dedekind 实数口径，并说明 Cauchy 实数与 Dedekind 实数在构造性基础中的比较需要哪些原则。

## AW.1 Dedekind cuts

**定义 AW.1（Dedekind cut）.** 一个 Dedekind 实数由两个谓词
$$
L,U:\mathbb Q\to\mathsf{Prop}
$$
组成，满足：

1.  inhabited：存在 $q,r:\mathbb Q$，$L(q)$ 且 $U(r)$；
2.  lower/upper closure：若 $q<r$ 且 $L(r)$，则 $L(q)$；若 $q<r$ 且 $U(q)$，则 $U(r)$；
3.  roundedness：$L(q)$ 当且仅当存在 $r>q$ 使 $L(r)$；$U(q)$ 当且仅当存在 $r<q$ 使 $U(r)$；
4.  disjointness：不存在 $q$ 同时满足 $L(q)$ 和 $U(q)$；
5.  locatedness：对任意 $q<r$，有
    $$
    L(q)+U(r).
    $$

记 Dedekind 实数类型为 $\mathbb R_D$。

**命题 AW.2（Dedekind 实数是集合）.** $\mathbb R_D$ 是集合。

**证明.** cut 的底层数据是命题值谓词对。由函数外延性和命题外延性，两个 cut 相等由逐点双向蕴含决定；结构公理均为命题。因此 $\mathbb R_D$ 是集合。$\square$

## AW.2 序与完备性

**定义 AW.3（Dedekind 序）.** 定义
$$
x<y\coloneqq \exists q:\mathbb Q.\ U_x(q)\times L_y(q)
$$
并用命题截断封装存在性。定义 $x\le y$ 为
$$
\prod_{q:\mathbb Q}L_x(q)\to L_y(q).
$$

**命题 AW.4（located order）.** 若 $x,y:\mathbb R_D$，则对任意 $q<r$，locatedness 可用于判定 $q<x$ 或 $x<r$ 的近似信息；特别地，Dedekind 序比 Cauchy 序更直接携带 locatedness。

**证明.** 直接由定义 AW.1 的 locatedness 条件。$\square$

**定理 AW.5（Dedekind 完备性，证明核 / 构造性定理）.** 若 $S:\mathbb R_D\to\mathsf{Prop}$ 非空且有上界，则满足 located upper set 条件时存在 least upper bound
$$
\sup S:\mathbb R_D.
$$

**证明核.** 定义 cut：
$$
L_{\sup S}(q)\coloneqq \exists x.\ S(x)\times L_x(q),
$$
$$
U_{\sup S}(q)\coloneqq \exists r>q.\ \prod_{x}S(x)\to U_x(r).
$$
lower/upper closure 和 roundedness 由各 $x$ 的 cut 性质与有理数稠密性给出。disjointness 来自上界条件。locatedness 需要 $S$ 的 located upper set 假设，用于在 $q<r$ 时决定 $L(q)$ 或 $U(r)$。最小上界性质按 $L$ 的定义逐点证明。$\square$

## AW.3 Cauchy 到 Dedekind

**定义 AW.6（Cauchy 实数到 Dedekind cut）.** 对 $x:\mathbb R_C$，定义 cut：
$$
L_x(q)\coloneqq q<x,\qquad
U_x(q)\coloneqq x<q
$$
其中右侧使用附录 AR 的构造性严格序。

**命题 AW.7（Cauchy 到 Dedekind 的 cut 条件）.** 若 $\mathbb R_C$ 的序满足 locatedness，则 AW.6 给出 Dedekind cut。

**证明（证明核）.** inhabited 由局部有界性 AR.5 给出上下有理界；closure 和 roundedness 由有理数稠密性、加法单调性和正性定义推出；disjointness 由严格序的反自反性；locatedness 正是额外假设。$\square$

**命题 AW.8（映射保持序和有理嵌入）.** 映射
$$
\mathbb R_C\to\mathbb R_D
$$
保持有理数嵌入和严格序。

**证明.** 有理数情形按 $q<r$ 的有序域定义展开。序保持由 AW.6 的定义。$\square$

## AW.4 Dedekind 到 Cauchy

**输入 AW.9（可选择有理近似）.** 对 Dedekind cut $x$，若给定选择函数，能对每个 $\varepsilon:\mathbb Q_{>0}$ 选出 $q_\varepsilon:\mathbb Q$，使
$$
L_x(q_\varepsilon)
$$
且 $q_\varepsilon$ 在宽度 $\varepsilon$ 内逼近 $x$。

**命题 AW.10（Dedekind 到 Cauchy 近似）.** 在输入 AW.9 下，$\varepsilon\mapsto\mathsf{rat}(q_\varepsilon)$ 是 $\mathbb R_C$ 中的 Cauchy 近似。

**证明（证明核）.** 若 $q_\varepsilon$ 和 $q_\delta$ 都在 cut 的宽度 $\varepsilon,\delta$ 内，则 locatedness 与 disjointness 给出
$$
|q_\varepsilon-q_\delta|<\varepsilon+\delta.
$$
把有理距离嵌入 $\mathbb R_C$ 即得 Cauchy 条件。$\square$

**边界 AW.11.** AW.9 通常需要 countable choice、dependent choice 或把 Dedekind cut 定义为携带近似函数的 located cut。无选择原则时，Dedekind 实数到 Cauchy 实数的函数不能无条件构造。

## AW.5 比较定理

**定理 AW.12（Cauchy-Dedekind 比较，条件版）.** 在以下假设下：

1.  $\mathbb R_C$ 的序 located；
2.  Dedekind cuts 带有可选择有理近似，或假设足够的选择原则；
3.  两侧实数均满足相应完备性；

则有等价
$$
\mathbb R_C\simeq\mathbb R_D
$$
并保持 $0,1,+,\cdot,<,\le$。

**证明架构.** 从 Cauchy 到 Dedekind 用 AW.6；从 Dedekind 到 Cauchy 用 AW.10 和 AK.9 取极限。两个复合在 Dedekind 侧由 cut extensionality AW.2 和有理近似密度证明；在 Cauchy 侧由 Cauchy 完备性和极限唯一性 AK.8 证明。代数和序保持逐项由有理近似和极限唯一性推出。$\square$

## AW.6 本附录关闭的缺口

Cauchy 实数适合计算和 HIIT 构造；Dedekind 实数适合序和上确界。两者等价不是无条件的集合论事实，而依赖 locatedness 与选择原则。本附录把这部分依赖显式化。
