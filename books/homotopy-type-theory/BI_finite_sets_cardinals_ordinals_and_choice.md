# 附录 BI：有限集、基数、序数与选择原则

本附录补齐 HoTT 中集合层基础数学的另一部分：有限集、基数、序数、良序和选择原则。它们属于 0-型数学，但在单值基础中必须通过等价、截断和商类型严格处理。

## BI.1 有限标准集

**定义 BI.1（$\mathsf{Fin}(n)$）。** 定义有限标准集
$$
\mathsf{Fin}(0)\coloneqq\mathbf 0,\qquad
\mathsf{Fin}(n+1)\coloneqq\mathsf{Fin}(n)+\mathbf 1.
$$

**命题 BI.2（$\mathsf{Fin}(n)$ 是集合）。** 对所有 $n:\mathbb N$，$\mathsf{Fin}(n)$ 是集合。

**证明.** 对 $n$ 归纳。$\mathbf 0$ 是集合；若 $\mathsf{Fin}(n)$ 是集合，则和类型 $\mathsf{Fin}(n)+\mathbf 1$ 是集合，见附录 AE。$\square$

**定义 BI.3（finite set）。** 集合 $A$ 是有限的，若
$$
\mathsf{isFinite}(A)\coloneqq
\left\|\sum_{n:\mathbb N}(A\simeq\mathsf{Fin}(n))\right\|.
$$

**定义 BI.4（cardinality of finite set）。** 若给定未截断数据 $(n,e:A\simeq\mathsf{Fin}(n))$，称 $n$ 为 $A$ 的基数。若只给出 $\mathsf{isFinite}(A)$，则基数存在于命题截断中。

## BI.2 基数唯一性

**定理 BI.5（标准有限集等价推出自然数相等，证明架构）。** 若
$$
\mathsf{Fin}(m)\simeq\mathsf{Fin}(n),
$$
则 $m=n$。

**证明架构.** 对 $m,n$ 双重归纳。$0$ 与 successor 情形由 $\mathbf 0$ 不等价于 inhabited 类型排除。successor-successor 情形取等价
$$
\mathsf{Fin}(m)+\mathbf 1\simeq\mathsf{Fin}(n)+\mathbf 1
$$
并移除一个点；需要证明 finite set 中 complements 的基数相容。完整证明可用鸽巢原理或 decidable equality 上的删点归纳。

**推论 BI.6（有限集基数唯一）。** 若 $A\simeq\mathsf{Fin}(m)$ 且 $A\simeq\mathsf{Fin}(n)$，则 $m=n$。

**证明.** 合成等价得到 $\mathsf{Fin}(m)\simeq\mathsf{Fin}(n)$，再用 BI.5。$\square$

## BI.3 有限集闭包

**命题 BI.7（有限集闭包，证明核）。** 若 $A$ 与 $B$ 有限，则 $A+B$ 和 $A\times B$ 有限。

**证明.** 由命题截断归纳，可假设
$$
A\simeq\mathsf{Fin}(m),\qquad B\simeq\mathsf{Fin}(n).
$$
和类型由
$$
\mathsf{Fin}(m)+\mathsf{Fin}(n)\simeq\mathsf{Fin}(m+n)
$$
给出；积类型由
$$
\mathsf{Fin}(m)\times\mathsf{Fin}(n)\simeq\mathsf{Fin}(mn)
$$
给出。这两个等价对自然数归纳构造。$\square$

**命题 BI.8（有限集有 decidable equality，证明核）。** 若 $A$ 有限，则 $A$ 有 decidable equality。

**证明.** 对截断有限性做命题目标归纳。若 $A\simeq\mathsf{Fin}(n)$，则把 $\mathsf{Fin}(n)$ 的 decidable equality 沿等价 transport 到 $A$。$\mathsf{Fin}(n)$ 的 decidable equality 对 $n$ 归纳。$\square$

## BI.4 基数

**定义 BI.9（基数）。** 在 universe $\mathcal U$ 中，基数可定义为集合的等价类：
$$
\mathsf{Card}_{\mathcal U}
\coloneqq
\left(\sum_{A:\mathcal U}\mathsf{isSet}(A)\right)\big/{\simeq}.
$$
这里商使用集合商 HIT。

**定义 BI.10（基数加法与乘法）。** 对代表元定义
$$
[A]+[B]\coloneqq[A+B],\qquad
[A]\cdot[B]\coloneqq[A\times B].
$$

**证明义务.** 需证明若 $A\simeq A'$、$B\simeq B'$，则
$$
A+B\simeq A'+B',\qquad A\times B\simeq A'\times B'.
$$
该证明由和类型与积类型的等价函子性给出。

## BI.5 序数与良序

**定义 BI.11（well-founded relation）。** 关系 $<_A:A\to A\to\mathsf{Prop}$ 是 well-founded，若每个 $a:A$ 可访问：
$$
\mathsf{Acc}(a)\coloneqq
\prod_{b:A}(b<_A a\to\mathsf{Acc}(b)).
$$

**定义 BI.12（ordinal，HoTT Book 口径）。** 序数可定义为集合 $A$ 加关系 $<$，满足：

1.  $<$ 为命题值关系；
2.  well-founded；
3.  extensional：若两个元素有相同前段，则相等；
4.  transitive：$c<b<a$ 推出 $c<a$。

**命题 BI.13（序数相等由双向初段嵌入控制，证明说明）。** 序数的等同性可由保持并反映 $<$ 的等价给出。

**证明说明.** 由单值性把载体等价转为路径；关系保持和反映证明 transport 后的关系相等。extensionality 保证 automorphism 平凡，well-foundedness 支持按前段递归。

## BI.6 选择原则

**定义 BI.14（选择原则，截断形式）。** 对类型族 $B:A\to\mathcal U$，一般选择原则是
$$
\left(\prod_{a:A}\|B(a)\|\right)
\to
\left\|\prod_{a:A}B(a)\right\|.
$$

**定义 BI.15（集合选择）。** 若 $A$ 是集合且每个 $B(a)$ 是集合，可考虑更强的未截断选择：
$$
\left(\prod_{a:A}\|B(a)\|\right)
\to
\prod_{a:A}B(a).
$$
这不是 HoTT 的默认定理。

**原则 BI.16（有限选择）。** 若 $A$ 有限，则从逐点 merely inhabited 的族可构造 merely inhabited 的依赖积：
$$
\mathsf{isFinite}(A)\to
\left(\prod_{a:A}\|B(a)\|\right)
\to
\left\|\prod_{a:A}B(a)\right\|.
$$

**证明.** 对有限性做命题目标归纳，化到 $A\simeq\mathsf{Fin}(n)$。对 $n$ 归纳：$0$ 情形依赖积为单位；successor 情形由归纳假设和最后一个元素的截断见证合并。$\square$

**警告 BI.17（排中律与选择）。** 排中律、唯一选择、可数选择、依赖选择和一般选择是不同原则。引入其中任一项都必须记录其对构造性、canonicity 和实数比较定理的影响。

## BI.7 有限性与选择的边界

有限选择只产生截断后的依赖积，并不推出一般选择。$\mathsf{Fin}(n)$ 的基数唯一性、序数等同性与基数算术只有在相应归纳和 well-defined 证明已给出时才能作为书内结果；选择原则之间的独立性则属于模型论外部输入，不由本附录的定义推出。
