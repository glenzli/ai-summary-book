# 第七章：单值性的基本后果

## 本章目标

本章发展单值性的初步数学后果：等价不变性、结构等同性、universe 的非集合性、集合层对象的外延原则。重点是把“等价结构可以视为相等”写成类型论中可 transport 的命题。

## 依赖前置知识

本章依赖第六章的函数外延性和单值性，以及第五章的等价。高阶归纳类型仍未使用。

## 7.1 等价不变性

**定义 7.1.** 一个类型族 $P:\mathcal U\to\mathcal U'$ 称为等价不变，若对任意 $e:A\simeq B$，有函数
$$
P(A)\to P(B)
$$
并且该函数由沿 $\mathsf{ua}(e):A=B$ 的 transport 给出。

**命题 7.2（性质沿等价运输）.** 假设单值性。对任意 $P:\mathcal U\to\mathcal U'$ 和 $e:A\simeq B$，有
$$
\mathsf{transport}^{P}(\mathsf{ua}(e)):P(A)\to P(B).
$$

**证明（书内证明）.** 这是 transport 的直接应用；路径由单值性给出的 $\mathsf{ua}(e):A=B$。$\square$

**例 7.3.** 若 $P(X)\coloneqq\mathsf{isSet}(X)$，则等价 $A\simeq B$ 可把 $A$ 是集合的证明运输为 $B$ 是集合的证明。第五章也能直接证明同伦层级保持性；单值性给出统一的 transport 解释。

## 7.2 结构等同性

**定义 7.4.** 设结构由族 $S:\mathcal U\to\mathcal U'$ 给出。带结构类型为
$$
\mathsf{Str}\coloneqq\sum_{A:\mathcal U}S(A).
$$
若 $S$ 对等价具有合适的 transport 行为，则两个结构对象 $(A,s)$ 与 $(B,t)$ 的相等可由结构保持等价刻画。

**定理 7.5（结构等同性原则，SIP）.** 对由结构族 $S:\mathcal U\to\mathcal V$ 给出的带结构对象，若结构等价定义为底层等价加沿 $\mathsf{ua}$ transport 后的结构相等，则单值性推出
$$
((A,s)=(B,t))\simeq((A,s)\cong(B,t)).
$$

**证明（书内证明）.** 见附录 I.3。由 $\Sigma$ 路径刻画，左边等价于 $\sum_{p:A=B}\mathsf{transport}^{S}(p,s)=t$；单值性把 $p:A=B$ 替换为 $e:A\simeq B$，得到规范结构等价。通常代数结构的同构条件由展开 transport 得到，见附录 I.7。$\square$

## 7.3 Universe 不是集合

**命题 7.6（Universe 的高阶性）.** 若 universe $\mathcal U$ 中存在有非平凡自等价的类型 $A$，则 $\mathcal U$ 通常不是集合。

**证明（书内证明）.** 见附录 H.7。取 $A\equiv\mathbf 2$，其取反函数给出非平凡自等价。若 $\mathcal U$ 是集合，则 $\mathsf{ua}(\mathsf{negBool})$ 与 $\mathsf{refl}_{\mathbf 2}$ 相等；对该路径作用 $\mathsf{idtoequiv}$ 后得到取反等价等于恒等等价，从而底层函数相等，与 $\mathsf{false}\ne\mathsf{true}$ 矛盾。$\square$

**警告 7.7.** 这不是矛盾。HoTT 允许 universe 具有高阶路径结构；它不要求所有类型都是集合。

## 7.4 集合层外延原则

**命题 7.8（命题子类型的相等）.** 设 $A$ 是类型，$P,Q:A\to\mathcal U$ 是命题值族。假设函数外延性和命题外延性。若
$$
\prod_{x:A}(P(x)\leftrightarrow Q(x)),
$$
则 $P=Q$。

**证明（书内证明）.** 见附录 F.6。对每个 $x:A$，由 $P(x)$ 与 $Q(x)$ 是命题和双向蕴含，命题外延性给出 $P(x)=Q(x)$。再由函数外延性得到族相等 $P=Q$。$\square$

**例 7.9.** 在集合层数学中，两个子集相等可由逐点等价证明。这是传统外延性在 HoTT 中的一个精确版本。

## 本章小结

单值性把等价提升为路径，使结构可以沿等价运输。它不是简单的“同构即相等”口号，而是由 $\mathsf{idtoequiv}$ 的等价性控制的类型论原则。

## 练习

**练习 7.1.** 设 $e:A\simeq B$。写出把 $A$ 上的二元运算 $A\to A\to A$ 运输到 $B$ 上的公式。

**练习 7.2.** 对命题 $P,Q$，用单值性和双向蕴含构造 $P=Q$。

**练习 7.3.** 说明为什么结构等同性原则需要区分“结构”和“性质”。

**练习 7.4.** 分析布尔类型上的取反函数为什么应给出 universe 中的非平凡环路。
