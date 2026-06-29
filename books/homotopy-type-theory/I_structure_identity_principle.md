# 附录 I：结构等同性原则证明核

## 目标

本附录给出一个可直接证明的结构等同性原则（Structure Identity Principle, SIP）。它不试图一次性覆盖所有可能的“结构”语法，而是给出一个精确版本：若结构由类型族
$$
S:\mathcal U\to\mathcal V
$$
给出，则带结构对象的路径等价于底层等价加上结构沿该等价 transport 后相等。通常代数结构的同构定义可通过展开 transport 与这个精确版本比较。

## I.1 带结构对象

**定义 I.1.** 给定结构族 $S:\mathcal U\to\mathcal V$，定义带结构类型
$$
\mathsf{Str}(S)\coloneqq \sum_{A:\mathcal U}S(A).
$$
其元素写为 $(A,s)$。

**定义 I.2（规范结构等价）.** 对 $(A,s),(B,t):\mathsf{Str}(S)$，定义
$$
(A,s)\simeq_S(B,t)
\coloneqq
\sum_{e:A\simeq B}
\mathsf{transport}^{S}(\mathsf{ua}(e),s)=t.
$$
这里的结构相容性不是额外猜测的公式，而是沿单值性路径 transport 后的相等。

## I.2 SIP 的精确形式

**定理 I.3（结构等同性原则，精确版）.** 假设单值性。对任意结构族 $S:\mathcal U\to\mathcal V$ 和任意 $(A,s),(B,t):\mathsf{Str}(S)$，有等价
$$
((A,s)=(B,t))\simeq((A,s)\simeq_S(B,t)).
$$

**证明.** 由 $\Sigma$ 路径刻画，
$$
((A,s)=(B,t))
\simeq
\sum_{p:A=B}\mathsf{transport}^{S}(p,s)=t.
$$
由单值性，
$$
(A=B)\simeq(A\simeq B),
$$
其中从右到左的方向为 $\mathsf{ua}$，并且 $\mathsf{idtoequiv}(\mathsf{ua}(e))=e$。把上式中的 $p$ 沿这个等价替换为 $e:A\simeq B$，得到
$$
\sum_{e:A\simeq B}\mathsf{transport}^{S}(\mathsf{ua}(e),s)=t,
$$
这正是定义 I.2。$\square$

**说明 I.4.** 定理 I.3 的证明完全一般，但它使用的是“规范结构等价”。若某一数学分支有传统同构定义，例如群同构、环同构、拓扑空间同胚，则还需证明传统定义与规范结构等价一致。

## I.3 命题性公理不产生额外结构

**定理 I.5（性质分量唯一）.** 设结构族写作
$$
S(A)\coloneqq\sum_{d:D(A)}P(A,d),
$$
其中对每个 $A,d$，$P(A,d)$ 是命题。则比较结构时，$P$ 的证明分量不产生额外相容性条件。

**证明.** 给定两个结构 $(d,p)$ 与 $(d',p')$。由 $\Sigma$ 路径刻画，结构路径由 $d=d'$ 和 transport 后的 $P$-证明相等组成。第二个目标位于某个命题 $P(A,d')$ 中，因此由命题性自动给出。$\square$

## I.4 代数结构的通常同构

**例 I.6（一个二元运算结构）.** 令
$$
S(A)\coloneqq A\to A\to A.
$$
若 $e:A\simeq B$，则沿 $\mathsf{ua}(e)$ transport 一个运算 $\mu:A\to A\to A$，得到 $B$ 上的运算。该运算 propositionally 等于
$$
\lambda b_1\,b_2.\ e\bigl(\mu(e^{-1}(b_1),e^{-1}(b_2))\bigr),
$$
其中 $e^{-1}$ 是等价的逆函数。

**证明.** 见附录 AG.3。对 $e$ 使用等价归纳，即由单值性把问题化到 $e$ 为恒等等价的情形。恒等情形中 transport 计算为自身。$\square$

**定理 I.7（命题性公理代数结构的 SIP）.** 设代数结构由有限多个运算、常元和命题性公理组成，并把结构同构定义为底层等价保持全部运算和常元。则该同构类型等价于定理 I.3 的规范结构等价。

**证明.** 见附录 AG.7。对每个常元和运算使用 AG.1-AG.4 的 transport 计算；对公理分量使用定理 I.5 删除证明相容性；有限乘积和 $\Sigma$ 的路径由附录 D.6 反复展开。$\square$

## I.5 群对象实例

**定理 I.8（群对象的相等与群同构）.** 在单值性下，若群定义为集合 $G$ 加乘法、单位、逆和命题性群公理，则群对象的路径等价于通常意义的群同构。

**证明.** 群结构是定理 I.7 的实例。底层结构包括：
$$
G:\mathcal U,\quad \mathsf{isSet}(G),\quad
\mu:G\to G\to G,\quad e:G,\quad i:G\to G,
$$
以及结合律、单位律、逆元律。$\mathsf{isSet}(G)$ 和群律都是命题性分量；由定理 I.5，它们不产生额外结构相容性。运算、单位、逆的 transport 相容性展开后，正是群同态保持乘法、单位和逆的条件。底层等价加这些保持条件即通常的群同构。由定理 I.3 得到结论。$\square$
