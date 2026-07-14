# 附录 AA：Weak equivalence 与 Rezk 泛性质的外部输入

Rezk 完备化的对象可以在预层范畴中直接写出，但其泛性质要求从“每个对象仅仅有一个原像”构造整个扩张函子。命题截断禁止任意选择代表，因此真正的证明要把每个待选数据组织成可收缩类型，再取其规范中心。本附录记录这一机制和本书采用的精确外部定理；它不把省略的 transport 与代表元相容计算冒充书内证明。

固定预范畴 $\mathcal A,\mathcal B,\mathcal E$，并假设 $\mathcal E$ 是单值范畴。所有 Hom 都是集合，函数与自然变换相等使用函数外延性，essential surjectivity 使用命题截断。

## AA.1 Weak equivalence 与限制函子

**定义 AA.1（fully faithful）.** 函子 $F:\mathcal A\to\mathcal B$ fully faithful，若对任意 $a,a':\mathcal A$，
$$
F_{a,a'}:\mathcal A(a,a')\to\mathcal B(Fa,Fa')
$$
是类型等价。

**定义 AA.2（essentially surjective）.** $F$ essentially surjective，若
$$
\prod_{b:\mathcal B}
\left\|
\sum_{a:\mathcal A}Fa\cong b
\right\|.
$$

**定义 AA.3（weak equivalence）.** $F$ 是 weak equivalence，若它 fully faithful 且 essentially surjective。第二项只给出截断后的代表存在性，不给出函数 $\mathcal B_0\to\mathcal A_0$。

**定义 AA.4（限制函子）.** 预合成给出
$$
F^*:[\mathcal B,\mathcal E]\to[\mathcal A,\mathcal E],
\qquad
H\longmapsto H\circ F.
$$
对自然变换 $\alpha:H\Rightarrow K$，其限制分量为
$$
(F^*\alpha)_a\coloneqq\alpha_{Fa}.
$$

## AA.2 自然变换为什么能从本质像恢复

**引理 AA.5（书内证明）.** 若 $F$ essentially surjective，且
$\alpha,\beta:H\Rightarrow K$ 在每个 $Fa$ 上分量相等，则 $\alpha=\beta$。

**证明.** 固定 $b:\mathcal B$。目标 $\alpha_b=\beta_b$ 是 Hom 集合中的路径，因而是命题；可对
$\left\|\sum_a Fa\cong b\right\|$
消去。取 $(a,i)$ 后，自然性给出
$$
K(i)\circ\alpha_{Fa}=\alpha_b\circ H(i),
$$
$$
K(i)\circ\beta_{Fa}=\beta_b\circ H(i).
$$
代入 $\alpha_{Fa}=\beta_{Fa}$，再与同构 $H(i)$ 的逆复合，得到
$\alpha_b=\beta_b$。对 $b$ 使用函数外延性，并用自然性证明的命题性比较自然变换记录，即得 $\alpha=\beta$。$\square$

存在性比唯一性困难，因为要实际定义 $\bar\gamma_b$。来源论文不是先挑一个代表再证明选择无关，而是使用如下可收缩候选类型。

**构造 AA.6（分量候选）.** 给定 $\gamma:HF\Rightarrow KF$ 与 $b:\mathcal B$，令
$$
\mathsf{Comp}(b)\coloneqq
\sum_{u:\mathcal E(Hb,Kb)}
\prod_{(a,i):\sum_{a:\mathcal A}Fa\cong b}
\bigl(
\gamma_a=K(i^{-1})\circ u\circ H(i)
\bigr).
$$
若临时取代表 $(a_0,i_0)$，唯一可能的第一分量是
$$
u_0\coloneqq
K(i_0)\circ\gamma_{a_0}\circ H(i_0^{-1}).
$$
另一代表 $(a_1,i_1)$ 与它之间的比较同构
$i_1^{-1}\circ i_0:Fa_0\cong Fa_1$
由 fully faithful 性唯一提升到 $\mathcal A$ 中；$\gamma$ 对该提升的自然性恰好证明两种公式相等。由于 Hom 是集合，其余相容证明都是命题。

这段说明给出了来源证明的关键等式，但本书没有展开：同构提升与复合的全部 transport、$\mathsf{Comp}(b)$ 的依赖对路径、以及由其中心组成自然变换时的自然性。因此下面的存在定理明确作为外部输入。

## AA.3 限制函子定理

**外部输入引理 AA.7（限制函子的 fully faithful 性）.** 若 $F$ full 且 essentially surjective，则对任意预范畴 $\mathcal E$，
$$
F^*:[\mathcal B,\mathcal E]\to[\mathcal A,\mathcal E]
$$
fully faithful。

**来源.** Ahrens--Kapulkin--Shulman 2015, Lemmas 8.1 与 8.2。Lemma 8.1 是引理 AA.5 的来源版本；Lemma 8.2 用与构造 AA.6 等价的可收缩类型（论文公式 (8.3)）构造扩张自然变换。本书不重证后一个可收缩性论证的全部依赖路径。

**外部输入定理 AA.8（weak equivalence 的限制泛性质）.** 若 $F:\mathcal A\to\mathcal B$ 是 weak equivalence，且 $\mathcal E$ 是单值范畴，则
$$
F^*:[\mathcal B,\mathcal E]\to[\mathcal A,\mathcal E]
$$
是预范畴同构。

**来源与精确版本.** Ahrens--Kapulkin--Shulman 2015, Theorem 8.4，DOI `10.1017/S0960129514000486`。论文先由 Lemmas 8.1--8.2 得到 fully faithful，再对每个 $G:\mathcal A\to\mathcal E$ 构造扩张对象与态射的可收缩候选类型，从而证明 essential surjectivity；单值目标把 weak equivalence 提升为预范畴同构。该定理使用论文意义下的 category，即本书的单值范畴。

**未重证边界.** 本书不承担 Theorem 8.4 中以下逐项证明：

1. 对象候选类型 $X_b$ 的依赖对路径与代表相容；
2. 态射候选类型 $Y_f$ 的可收缩性；
3. 由 $Y_f$ 的中心得到的恒等、复合与自然性；
4. 扩张函子限制回 $\mathcal A$ 时的 transport 计算。

因此，AA.8 是精确外部输入，不是已经在本书内部构造出的定理，也不能因列出证明机制而改变身份。

## AA.4 应用于 Yoneda 本质像

**命题 AA.9（书内构造的输入核对）.** 对附录 R 的
$$
\eta_{\mathcal C}:\mathcal C\to\widehat{\mathcal C},
$$
目标 $\widehat{\mathcal C}$ 是单值范畴，且 $\eta_{\mathcal C}$ fully faithful 并 essentially surjective。

**证明.** 单值性是 R.7，fully faithful 是 R.9，essential surjectivity 是 R.10。最后一项只向命题截断目标消去“仅仅可表”的证明，故不选择全局代表。$\square$

**外部输入推论 AA.10（Rezk 完备化泛性质）.** 对任意单值范畴 $\mathcal E$，
$$
\eta_{\mathcal C}^*:
[\widehat{\mathcal C},\mathcal E]
\longrightarrow
[\mathcal C,\mathcal E]
$$
是预范畴同构。

**证明路线（外部输入）.** 将 AA.9 的 weak equivalence
$F\coloneqq\eta_{\mathcal C}$
代入外部输入定理 AA.8。来源论文 Theorem 8.5 还直接构造同一个 Yoneda 本质像 Rezk 完备化；其 universes 与 locally-small 假设见论文 §8 和 Remark 8.6。$\square$

## AA.5 读者应保留的区分

Rezk 完备化对象、继承的 Hom、单值性以及嵌入的 weak-equivalence 性质都已在附录 R 书内给出。只有从任意 weak equivalence 向单值目标扩张函子的普遍机制采用外部输入 AA.8。这个分界使命题截断的作用清楚可见：证明“某个分量唯一”时可以消去截断；定义整个函子时，必须先证明候选类型可收缩，不能暗中选择代表。
