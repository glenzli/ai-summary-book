# 第七章：单值性的基本后果

单值性最容易被误写成“同构的东西就是同一个东西”。精确说法更有内容：一个等价 $e:A\simeq B$ 产生路径 $\mathsf{ua}(e):A=B$，任何依赖于类型的结构族都可以沿这条路径 transport。于是问题不再是口头上是否把 $A$ 与 $B$ 视为相同，而是搬运后的运算、关系或证明究竟得到什么。

本章从元素 transport 的可计算例子进入，再把同一机制推广到带结构对象的 $\Sigma$ 型。函数外延性和单值性已经从第六章可用，高阶归纳类型仍未出现。布尔取反还会显示另一面：类型自等价在宇宙中生成真实环路，所以单值宇宙一般不能是集合。

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

**例 7.3.1（布尔取反的元素 transport）.** 令
$$
\mathsf{negEquiv}:\mathbf 2\simeq\mathbf 2
$$
的底层函数交换 $\mathsf{false}$ 与 $\mathsf{true}$，并以自身为逆。取
$$
p\coloneqq\mathsf{ua}(\mathsf{negEquiv}):\mathbf 2=\mathbf 2.
$$
由命题 6.9.1，在恒等类型中有
$$
\mathsf{transport}^{\lambda X:\mathcal U.\,X}
(p,\mathsf{false})
=
\mathsf{negEquiv}.1(\mathsf{false})
\equiv
\mathsf{true}.
$$
最后一步是布尔递归的 judgmental 计算，前一步则只是路径。这个例子同时展示了单值 transport 的作用和两种相等强度的差别。

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

**命题 7.6（Universe 的高阶性）.** 假设 $\mathcal U$ 单值。若存在 $A:\mathcal U$ 与自等价 $e:A\simeq A$，使
$$
e\ne\mathsf{idEquiv}_A,
$$
则 $\neg\mathsf{isSet}(\mathcal U)$。

**证明（书内证明）.** 若 $\mathcal U$ 是集合，则两条 loop
$\mathsf{ua}(e),\mathsf{refl}_A:A=A$ 相等。对该路径作用
$\mathsf{idtoequiv}$，再用单值性的逆律，得到
$e=\mathsf{idEquiv}_A$，与假设矛盾。附录 H.7 取
$A\equiv\mathbf 2$、$e\equiv\mathsf{negBool}$，并由
$\mathsf{false}\ne\mathsf{true}$ 验证该自等价确实非平凡。$\square$

**警告 7.7.** 这不是矛盾。HoTT 允许 universe 具有高阶路径结构；它不要求所有类型都是集合。

## 7.4 集合层外延原则

**命题 7.8（命题子类型的相等）.** 设 $A$ 是类型，$P,Q:A\to\mathcal U$ 是命题值族。假设函数外延性和命题外延性。若
$$
\prod_{x:A}(P(x)\leftrightarrow Q(x)),
$$
则 $P=Q$。

**证明（书内证明）.** 见附录 F.6。对每个 $x:A$，由 $P(x)$ 与 $Q(x)$ 是命题和双向蕴含，命题外延性给出 $P(x)=Q(x)$。再由函数外延性得到族相等 $P=Q$。$\square$

**例 7.9.** 在集合层数学中，两个子集相等可由逐点等价证明。这是传统外延性在 HoTT 中的一个精确版本。

## 7.5 从等价运输到结构恒等

单值性提供的不是省略结构保持条件的许可证，而是统一产生这些条件的 transport 原理。布尔取反说明 transport 可以改变元素，SIP 则用 $\Sigma$ 路径把底层等价与结构分量的相容性合在一起。与此同时，自等价在宇宙中形成环路，证明宇宙保留了高阶信息。下一章的截断会反向操作：在明确的泛性质下忘掉一部分高阶信息。

## 练习

**练习 7.1.** 设 $e:A\simeq B$。写出把 $A$ 上的二元运算 $A\to A\to A$ 运输到 $B$ 上的公式。

**练习 7.2.** 对命题 $P,Q$，用单值性和双向蕴含构造 $P=Q$。

**练习 7.3.** 说明为什么结构等同性原则需要区分“结构”和“性质”。

**练习 7.4.** 分析布尔类型上的取反函数为什么应给出 universe 中的非平凡环路。
