# 附录 BL：逻辑原则、Resizing、选择与构造性边界

本附录集中登记 HoTT 中常见逻辑原则的强度：排中律、双重否定消去、命题 resizing、唯一选择、可数选择、依赖选择和一般选择。它们影响实数比较、商结构、序数理论和构造性计算内容，不能隐式使用。

## BL.1 命题与否定

**定义 BL.1（否定）。** 定义
$$
\neg A\coloneqq A\to\mathbf 0.
$$
若 $A$ 是命题，则 $\neg A$ 也是命题。

**命题 BL.2（否定是命题，书内证明核）。** 对任意 $A$，$\neg A$ 是命题。

**证明.** 若 $f,g:A\to\mathbf 0$，由函数外延性，只需对每个 $a:A$ 证明 $f(a)=g(a)$。但 $f(a):\mathbf 0$，由空类型消去得到任意路径。$\square$

**定义 BL.3（稳定命题）。** 命题 $P$ 称为 stable，若
$$
\neg\neg P\to P.
$$

**命题 BL.4（否定命题稳定）。** $\neg A$ stable。

**证明.** 给定 $h:\neg\neg\neg A$，需构造 $\neg A$。若 $a:A$，则 $\lambda k.k(a):\neg\neg A$，于是 $h(\lambda k.k(a)):\mathbf 0$。$\square$

## BL.2 排中律与双重否定消去

**定义 BL.5（LEM）。** 命题排中律是原则
$$
\mathsf{LEM}\coloneqq\prod_{P:\mathsf{Prop}}(P+\neg P).
$$

**定义 BL.6（DNE）。** 双重否定消去是原则
$$
\mathsf{DNE}\coloneqq\prod_{P:\mathsf{Prop}}(\neg\neg P\to P).
$$

**命题 BL.7（LEM 推出 DNE，书内证明核）。** $\mathsf{LEM}\to\mathsf{DNE}$。

**证明.** 给定 $P$ 和 $h:\neg\neg P$。由 LEM 对 $P$ 分情况。若 $p:P$，返回 $p$。若 $n:\neg P$，则 $h(n):\mathbf 0$，由空类型消去得到 $P$。$\square$

**命题 BL.8（DNE 推出 LEM，书内证明核）。** $\mathsf{DNE}\to\mathsf{LEM}$。

**证明.** 对命题 $P$，$P+\neg P$ 是命题需要命题外延或和类型在两个互斥命题下的命题性。先证明 $\neg\neg(P+\neg P)$：若 $k:\neg(P+\neg P)$，则可构造 $\neg P$，因为 $p:P$ 给出 $\mathsf{inl}(p)$ 与 $k$ 矛盾；于是 $\mathsf{inr}(\neg P):P+\neg P$，再与 $k$ 矛盾。由 DNE 得 $P+\neg P$。$\square$

## BL.3 Propositional resizing

**定义 BL.9（propositional resizing）。** Resizing 原则断言高宇宙命题可等价替换为低宇宙命题：
$$
\prod_{P:\mathsf{Prop}_{\mathcal U_j}}
\sum_{Q:\mathsf{Prop}_{\mathcal U_i}}(Q\simeq P)
$$
其中 $i<j$。

**使用边界 BL.10.** Resizing 常用于把子对象分类器、powerset、局部化和 sheaf 条件放回同一宇宙。它不是本书默认规则；使用时必须标注宇宙层级和 resizing 方向。

## BL.4 唯一选择

**定义 BL.11（contractible fiber choice）。** 若
$$
\prod_{a:A}\mathsf{isContr}(B(a)),
$$
则可定义函数
$$
\prod_{a:A}B(a)
$$
为每个 fiber 的中心。

**定理 BL.12（唯一选择，书内证明核）。** 若
$$
\prod_{a:A}\left\|\sum_{b:B(a)}\prod_{b':B(a)}(b=b')\right\|,
$$
则
$$
\prod_{a:A}B(a).
$$

**证明.** 目标 $B(a)$ 不一定是命题，不能直接从截断消去取见证。先证明
$$
\sum_{b:B(a)}\prod_{b'}(b=b')
$$
是命题：两个中心由唯一性路径相等，第二分量由函数外延性和路径类型的相应命题性得到。于是可对命题截断消去，得到每个 $a$ 的唯一中心，再取第一投影。$\square$

## BL.5 选择原则层级

**定义 BL.13（可数选择）。** 可数选择为
$$
\left(\prod_{n:\mathbb N}\|B(n)\|\right)
\to
\left\|\prod_{n:\mathbb N}B(n)\right\|.
$$

**定义 BL.14（依赖选择）。** 依赖选择断言：若关系 $R:A\to A\to\mathcal U$ 满足
$$
\prod_{a:A}\left\|\sum_{b:A}R(a,b)\right\|,
$$
则从初值 $a_0:A$ 可仅仅构造序列 $a:\mathbb N\to A$，满足
$$
a(0)=a_0,\qquad \prod_n R(a(n),a(n+1)).
$$

**定义 BL.15（一般选择）。** 一般选择为
$$
\left(\prod_{a:A}\|B(a)\|\right)
\to
\left\|\prod_{a:A}B(a)\right\|.
$$
若结论未截断，则更强，通常不构造性。

**事实 BL.16（选择影响）。** 可数选择常用于 Cauchy/Dedekind 比较、序列紧致性和实分析定理；依赖选择用于递归构造逼近序列；一般选择会显著改变构造性强度。附录 AW、BA、BI 中凡涉及选择均需标注具体版本。

## BL.6 Diaconescu 型风险

**警告 BL.17（选择与排中律的相互作用）。** 在有足够 quotient、powerset 和 extensionality 的系统中，强选择原则可推出排中律。这类 Diaconescu 型结果说明：不能把“集合层选择”无条件加入并仍声称完全构造性。

**使用规则 BL.18.** 本书允许三种模式：

1.  构造性模式：不使用 LEM、DNE、一般选择和 resizing；
2.  局部原则模式：在定理前显式列出可数选择、locatedness、resizing 等假设；
3.  classical HoTT 模式：明确假设 LEM 和选择，并标注计算性结果失效或需重新解释。

## BL.7 与 canonicity 的关系

**原则 BL.19.** 若基础系统有计算 canonicity，加入非计算性公理如 LEM、choice 或 resizing 可能破坏 closed natural number term 的归约解释，除非采用 proof-irrelevant、公理化或 sheaf/realizability 模型给出替代语义。

**边界 BL.20.** Cubical univalence 的计算性不自动覆盖任意 classical axiom。把 cubical 口径下的证明与 classical HoTT 定理混用时必须区分数据内容和命题内容。

## BL.8 逻辑强度的边界

LEM、DNE、resizing、唯一选择、可数选择、依赖选择与一般选择是不同原则。它们之间的独立性需要模型论证明；具体分析定理则应逐项寻找最小假设。一个 classical corollary 不自动带有构造性替代，也不能仅凭 cubical univalence 保留原有 canonicity。
