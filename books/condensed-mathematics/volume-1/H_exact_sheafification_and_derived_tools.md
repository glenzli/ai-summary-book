# 附录 H：正合 sheafification 与派生工具

## H.0 目标

第一卷正文和附录 E 使用三类一般事实：

1. 阿贝尔群值 sheafification 是正合函子。
2. sheaf of modules 形成 Grothendieck 阿贝尔范畴。
3. 派生张量积可以用 K-flat 替换定义，并且与替换选择无关。

这些事实常被压缩成“标准理论”。本附录把第一项给出书内证明，把第二、三项拆成可核查的命题与明确输入定理。这样后续使用 $\operatorname{Ext}$、$\operatorname{Tor}$、solidification 和 analyticization 时，读者能看清哪些步骤是形式同调代数，哪些步骤仍引用一般定理。

本附录固定附录 A 工作层级内的小站点 $(\mathcal C,J)$。涉及模时再固定一个
交换环 sheaf \(R\)，从而得到 ringed site \((\mathcal C,R)\)。记

$$
\operatorname{PSh}(\mathcal C;\mathbf{Ab})
=
\operatorname{Fun}(\mathcal C^{op},\mathbf{Ab}),
\qquad
\operatorname{Sh}(\mathcal C;\mathbf{Ab})
\subset
\operatorname{PSh}(\mathcal C;\mathbf{Ab}).
$$

所有覆盖均指 $J$-覆盖。覆盖族、复形的次数集和所有直和指标都属于固定工作
universe；\(D(R)\) 表示无界导出范畴，不暗含有界上或有界下条件。

## H.1 匹配族与 plus 构造

设 $F$ 是阿贝尔群值预层，$\mathcal U=\{U_i\to U\}_{i\in I}$ 是覆盖。定义该覆盖上的匹配族群为等化子

$$
\operatorname{Match}(\mathcal U,F)
=
\ker\left(
\prod_i F(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_UU_j)
\right),
$$

两支箭头分别由 $U_i\times_UU_j\to U_i$ 与 $U_i\times_UU_j\to U_j$ 诱导。

**定义 H.1.** $F$ 的 plus 预层 $F^+$ 定义为

$$
F^+(U)
=
\varinjlim_{\mathcal U\in \operatorname{Cov}(U)}
\operatorname{Match}(\mathcal U,F),
$$

其中余极限沿覆盖细化取。若 $\mathcal V$ 细化 $\mathcal U$，则匹配族限制给出

$$
\operatorname{Match}(\mathcal U,F)\to
\operatorname{Match}(\mathcal V,F).
$$

**引理 H.2.** 对每个预层 $F$，$F^+$ 是 separated presheaf，即若 $s,t\in F^+(U)$ 在某个覆盖 $\{U_i\to U\}$ 上限制相同，则 $s=t$。

**证明.** 元素 $s,t$ 分别由覆盖 $\mathcal V,\mathcal W$ 上的匹配族表示。把 $\mathcal U,\mathcal V,\mathcal W$ 取共同细化 $\mathcal R$。在 $\mathcal R$ 上，$s,t$ 的限制逐项相同；由于 $F^+(U)$ 是按覆盖细化的余极限，两个代表在共同细化中相等即表示同一个余极限元素。证毕。

**引理 H.3.** 若 $F$ separated，则 $F^+$ 是 sheaf。

**证明.** 设 $\mathcal U=\{U_i\to U\}$ 是覆盖，给定 $F^+$ 中关于 $\mathcal U$ 的匹配族 $(s_i)$。每个 $s_i$ 可由覆盖 $\mathcal V_i=\{V_{ia}\to U_i\}$ 上的 $F$-匹配族 $(x_{ia})$ 表示。复合族 $\{V_{ia}\to U\}_{i,a}$ 覆盖 $U$。

需要证明 $(x_{ia})$ 是 $F$ 的匹配族。任取

$$
V_{ia}\times_UV_{jb}.
$$

它映到 $U_i\times_UU_j$。由于 $(s_i)$ 在 $F^+$ 中匹配，$s_i$ 与 $s_j$ 在 $U_i\times_UU_j$ 上的限制相同。按 $F^+$ 的定义，存在该纤维积的覆盖，使得 $x_{ia}$ 与 $x_{jb}$ 的限制在覆盖上相同。因为 $F$ separated，覆盖上相同推出它们在整个 $V_{ia}\times_UV_{jb}$ 上相同。故 $(x_{ia})$ 是匹配族，定义出 $s\in F^+(U)$。

唯一性由引理 H.2 给出。证毕。

**推论 H.4.** 阿贝尔群值 sheafification 可取为

$$
aF=(F^+)^+.
$$

自然映射 $F\to aF$ 对任意 sheaf $A$ 诱导双射

$$
\operatorname{Hom}_{\operatorname{Sh}}(aF,A)
\cong
\operatorname{Hom}_{\operatorname{PSh}}(F,A).
$$

**证明.** 由引理 H.2，$F^+$ separated；由引理 H.3，$(F^+)^+$ 是 sheaf。给定 $F\to A$，若 $A$ 是 sheaf，则每个覆盖上的匹配族可唯一粘合到 $A(U)$，于是得到 $F^+\to A$，再得到 $(F^+)^+\to A$。反向由复合 $F\to aF\to A$ 给出。唯一性来自 sheaf 粘合的唯一性。证毕。

## H.2 Sheafification 的正合性

**引理 H.5（局部零判据）.** 设 $F$ 是阿贝尔群值预层，$s\in F(U)$。则 $s$ 在 $aF(U)$ 中为零，当且仅当存在覆盖 $\{U_i\to U\}$ 使得

$$
s|_{U_i}=0\in F(U_i)
$$

对所有 $i$ 成立。

**证明.** 若 $s$ 局部为零，则在 plus 构造中它与零匹配族在该覆盖上有相同限制，故在 $F^+(U)$ 中为零，从而在 $aF(U)$ 中为零。

反过来，若 $s$ 在 $aF(U)=(F^+)^+(U)$ 中为零，则按 $F^+$ 的 separated 性和 plus 构造，存在覆盖 $\mathcal U$ 使得 $s$ 在 $F^+(U_i)$ 中为零。再对每个 $U_i$ 展开 $F^+$ 中为零的含义，得到覆盖 $\mathcal V_i$ 使得 $s$ 在每个 $V_{ij}$ 上为零。复合覆盖给出结论。证毕。

**定理 H.6（正合 sheafification）.** 函子

$$
a:\operatorname{PSh}(\mathcal C;\mathbf{Ab})
\to
\operatorname{Sh}(\mathcal C;\mathbf{Ab})
$$

是正合函子。

**证明.** 因为 $a$ 是左伴随，它保持余核和所有余极限。只需证明它保持 kernel，等价地证明若

$$
0\to F'\xrightarrow{\alpha}F\xrightarrow{\beta}F''
$$

在预层范畴逐对象正合，则

$$
0\to aF'\xrightarrow{a\alpha}aF\xrightarrow{a\beta}aF''
$$

在 sheaf 范畴正合。

先证 $a\alpha$ 单。设 $s\in aF'(U)$ 映到 $0\in aF(U)$。用 plus 构造表示 $s$：存在覆盖 $\{U_i\to U\}$ 和截面 $s_i\in F'(U_i)$ 形成匹配族，表示 $s$。$a\alpha(s)=0$ 表示在某个共同细化 $V_{ij}\to U_i$ 上，$\alpha(s_i)|_{V_{ij}}=0$。由于预层层面 $\alpha$ 单，$s_i|_{V_{ij}}=0$。由局部零判据，$s=0$。

再证中间正合。复合 $(a\beta)(a\alpha)$ 为零，因为预层层面的复合 $\beta\alpha$ 为零，sheafification 保持复合。反过来，设 $t\in aF(U)$ 且 $a\beta(t)=0$。取覆盖 $\{U_i\to U\}$ 和 $t_i\in F(U_i)$ 表示 $t$。条件 $a\beta(t)=0$ 意味着经细化后 $\beta(t_i)$ 局部为零。替换为共同细化，可假设 $\beta(t_i)=0$。预层正合给出唯一 $s_i\in F'(U_i)$ 使 $\alpha(s_i)=t_i$。由于 $(t_i)$ 是匹配族，且 $\alpha$ 单，$(s_i)$ 也是匹配族。它粘合为 $s\in aF'(U)$，并满足 $a\alpha(s)=t$。

于是 $a$ 保持 kernel；结合保持余核，$a$ 正合。证毕。

**推论 H.7.** $\operatorname{Sh}(\mathcal C;\mathbf{Ab})$ 是阿贝尔范畴，kernel 逐对象计算，cokernel 为预层 cokernel 的 sheafification。

**证明.** 预层阿贝尔群范畴是阿贝尔范畴，且有限极限、有限余极限逐对象计算。kernel 是有限极限，sheaf 条件对极限稳定，故 kernel 仍是 sheaf。cokernel 用 sheafification 送回 sheaf 范畴。由定理 H.6，coimage 到 image 的比较映射为同构。证毕。

## H.3 Grothendieck 阿贝尔范畴结构

**命题 H.8.** $\operatorname{Sh}(\mathcal C;\mathbf{Ab})$ 满足 AB5：filtered colimits 正合。

**证明.** filtered colimit 在预层范畴逐对象计算，而 $\mathbf{Ab}$ 中 filtered colimits 正合。sheaf 范畴中的 filtered colimit 可由预层 filtered colimit 后 sheafification 得到。由定理 H.6，sheafification 正合。因此 filtered colimit 把短正合列送到短正合列。证毕。

**命题 H.9.** 若 $\mathcal C$ 小，则 $\operatorname{Sh}(\mathcal C;\mathbf{Ab})$ 有生成元。

**证明.** 对每个 $U\in\mathcal C$，令

$$
\mathbb Z[h_U]=a(\mathbb Z[\operatorname{Hom}_{\mathcal C}(-,U)])
$$

为可表预层生成的自由阿贝尔 sheaf。取

$$
G=\bigoplus_{[U]\in \operatorname{Ob}(\mathcal C)/\cong}\mathbb Z[h_U].
$$

若 $A\ne0$，存在 $U$ 和 $s\in A(U)$ 非零。由 Yoneda 和自由性，$s$ 给出非零态射 $\mathbb Z[h_U]\to A$，从而给出非零态射 $G\to A$。故 $G$ 是生成元。证毕。

**推论 H.10.** $\operatorname{Sh}(\mathcal C;\mathbf{Ab})$ 是 Grothendieck 阿贝尔范畴。

**证明.** 由推论 H.7 得阿贝尔性；由命题 H.8 得 AB5；由命题 H.9 得生成元。证毕。

## H.4 Sheaf 模与投射/平坦分解

令 $R$ 是 $\operatorname{Sh}(\mathcal C;\mathbf{Ab})$ 中的交换环对象。

**命题 H.11.** $R$-模范畴 $R\text{-}\mathbf{Mod}$ 是 Grothendieck 阿贝尔范畴。

**证明.** 遗忘函子

$$
U:R\text{-}\mathbf{Mod}\to \operatorname{Sh}(\mathcal C;\mathbf{Ab})
$$

有左伴随 $M\mapsto R\otimes M$，并且 $R$-模中的极限由遗忘函子反映。余极限可在 sheaf 阿贝尔群中取余极限后赋予诱导的 $R$-作用。kernel、cokernel 和 filtered colimit 在底层 sheaf 中计算；由推论 H.10 的 AB5 得正合性。生成元可取 $R\otimes\mathbb Z[h_U]$ 的直和。证毕。

**定义 H.12.** 复形 $P^\bullet$ 称为 K-flat，如果函子

$$
\operatorname{Tot}(P^\bullet\otimes_R-):K(R)\to K(R)
$$

把 quasi-isomorphism 送到 quasi-isomorphism。等价地，对任意 acyclic 复形 $A^\bullet$，复形

$$
\operatorname{Tot}(P^\bullet\otimes_R A^\bullet)
$$

仍 acyclic。复形 $I^\bullet$ 称为 K-injective，如果对任意 acyclic $A^\bullet$，

$$
\operatorname{Hom}^\bullet_R(A^\bullet,I^\bullet)
$$

acyclic。

**外部输入定理 H.13（无界替换的精确范围）.** 设 \((\mathcal C,R)\) 是上述
小 ringed site。

1. 因 \(R\text{-}\mathbf{Mod}\) 是 Grothendieck 阿贝尔范畴，每个无界复形
   \(M^\bullet\) 有函子性 quasi-isomorphism
   $$
   M^\bullet\longrightarrow I^\bullet,
   $$
   其中 \(I^\bullet\) K-injective，且每个 \(I^n\) 都是 injective \(R\)-模。
2. 每个无界复形 \(M^\bullet\) 有 termwise surjective quasi-isomorphism
   $$
   P^\bullet\longrightarrow M^\bullet,
   $$
   其中 \(P^\bullet\) K-flat，且每个 \(P^n\) 都是 flat \(R\)-模。

**来源定位.** 第 1 项是 Stacks Project Tag `079P`（Theorem 19.12.6）；第 2 项是
Tag `06YL`（Lemma 21.17.11）。后者专门针对 ringed site，因此本书不再声称“任意
右正合闭幺半 Grothendieck 范畴”都自动有 K-flat 替换。

**本书不证明的边界.** 两个构造都使用无界复形的超限或滤过替换。本书使用它们保证
派生 Hom 与派生张量存在，但不重做该构造。

**定义 H.14.** 对 $M^\bullet,N^\bullet\in D(R)$，取 H.13(2) 的 termwise
surjective K-flat 替换 $P^\bullet\to M^\bullet$，定义

$$
M^\bullet\otimes_R^L N^\bullet
=
\operatorname{Tot}(P^\bullet\otimes_R N^\bullet).
$$

总复形使用直和 totalization 与 Koszul 号差。

**命题 H.15（定义独立性）.** 上述定义与 K-flat 替换选择无关。

**证明.** 先记录标准比较引理：若 $u:P^\bullet\to Q^\bullet$ 是 K-flat 复形之间的 quasi-isomorphism，则对任意 $N^\bullet$，

$$
u\otimes\operatorname{id}_{N^\bullet}:
P^\bullet\otimes_RN^\bullet\to Q^\bullet\otimes_RN^\bullet
$$

是 quasi-isomorphism。证明如下。取 $N^\bullet$ 的 K-flat 替换 $Q_N^\bullet\to N^\bullet$。由于 $P^\bullet$ 和 $Q^\bullet$ K-flat，映射

$$
P^\bullet\otimes_RQ_N^\bullet\to P^\bullet\otimes_RN^\bullet,
\qquad
Q^\bullet\otimes_RQ_N^\bullet\to Q^\bullet\otimes_RN^\bullet
$$

都是 quasi-isomorphism。又因为 $Q_N^\bullet$ K-flat，$u\otimes\operatorname{id}_{Q_N^\bullet}$ 是 quasi-isomorphism。由二出三性质，$u\otimes\operatorname{id}_{N^\bullet}$ 是 quasi-isomorphism。

现在设 $P^\bullet\to M^\bullet$ 与 $Q^\bullet\to M^\bullet$ 是 H.13(2)
给出的两个 termwise surjective K-flat 替换。取复形范畴中的逐次拉回

$$
R^\bullet=P^\bullet\times_{M^\bullet}Q^\bullet.
$$

投影 \(R^\bullet\to P^\bullet\) 的 kernel 等于
\(\ker(Q^\bullet\to M^\bullet)\)，后者 acyclic；该投影又逐次满射，所以是
quasi-isomorphism。同理 \(R^\bullet\to Q^\bullet\) 是 quasi-isomorphism。
再取 H.13(2) 的 K-flat 替换 \(T^\bullet\to R^\bullet\)。复合给出 K-flat 复形间的
quasi-isomorphisms

$$
T^\bullet\longrightarrow P^\bullet,
\qquad
T^\bullet\longrightarrow Q^\bullet.
$$

由比较引理，

$$
T^\bullet\otimes_RN^\bullet\to P^\bullet\otimes_RN^\bullet,
\qquad
T^\bullet\otimes_RN^\bullet\to Q^\bullet\otimes_RN^\bullet
$$

都是 quasi-isomorphism。故两个 totalized tensor complexes 在导出范畴中由 zigzag
同构。这个论证同时说明独立性依赖 termwise surjective 替换，不需要指定一个
homotopy pullback 模型。证毕。

**推论 H.16.** 若 $F$ 是平坦 $R$-模，视为次数 $0$ 复形，则

$$
F\otimes_R^L N^\bullet\simeq F\otimes_RN^\bullet.
$$

特别地，对 $N$ 也在次数 $0$ 的情形，

$$
\operatorname{Tor}_i^R(F,N)=0,\qquad i>0.
$$

**证明.** 平坦性说明 $F\otimes_R-$ 正合，因此 $F$ 作为次数 $0$ 复形 K-flat。代入定义 H.14。证毕。

## H.5 对第一卷正文的回填

1. 附录 C 中“sheafification 正合”由定理 H.6 给出。
2. 附录 E 中“$R$-模范畴是 Grothendieck 阿贝尔范畴”由命题 H.11 给出。
3. 第十一章和附录 G 中的派生张量积由定义 H.14 与命题 H.15 支撑。
4. solid 和 analytic 章节中出现的派生范畴仍需要 Scholze 的额外结构定理；本附录只提供一般 sheaf 模的同调代数基础。

## 练习

**练习 H.1.** 证明若 $F$ 已经是 sheaf，则自然映射 $F\to F^+$ 是同构。

**练习 H.2.** 在定理 H.6 的证明中，写出“共同细化”如何同时细化表示 $t$ 的覆盖和使 $\beta(t_i)$ 局部为零的覆盖。

**练习 H.3.** 证明 $G=\bigoplus_U\mathbb Z[h_U]$ 是生成元时，不需要选择每个同构类以外的对象。

**练习 H.4.** 设 $P^\bullet$ K-flat。证明 $P^\bullet[n]$ 仍 K-flat。
