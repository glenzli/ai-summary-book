# 附录 A：集合论宇宙、小骨架、presentability 与 accessible localization

## 本附录目标

本附录补足全书使用的大小约定和局部化工具。Motivic homotopy theory 同时使用几何范畴、presheaf categories、sheaf categories、presentable infinity-categories 和 stable categories；若不固定宇宙和小骨架，许多“由所有光滑概形生成”的表述会变成 proper-class 级别的伪命题。

## 依赖前置知识

需要 Grothendieck universe、小范畴、presheaf category、accessible category、presentable infinity-category、localization 和 adjoint functor theorem。

## A.1 宇宙和小骨架

**约定 A.1.** 固定 Grothendieck universes

$$
\mathbb U\in\mathbb V.
$$

本书中“集合”“小余极限”和“小范畴”分别指 `\mathbb U`-小集合、
`\mathbb U`-小余极限和 `\mathbb U`-小范畴。记 `\mathcal S_{\mathbb U}`
为 `\mathbb U`-小 spaces 的 infinity-范畴；它作为一个 infinity-范畴是
`\mathbb V`-小的。一个范畴称为 presentable，意指它是
`\mathbb U`-accessible、具有全部 `\mathbb U`-小余极限，并作为
`\operatorname{Cat}_{\mathbb V}` 的对象讨论。相应地，
`\operatorname{Pr}^{L}` 的态射保持全部 `\mathbb U`-小余极限。

全部概形都取为 `\mathbb U`-小概形。为使后文的基变换仍落在默认范围内，
固定一个 `\mathbb U`-小有限维 Noetherian 概形 `B`；默认变化的基概形是
有限型 `B`-概形。它们的纤维积仍有限型于 `B`，因而仍为有限维
Noetherian 概形。单独只固定一个基概形 `S` 的第一至第三章只需假设
`S` 为 `\mathbb U`-小有限维 Noetherian 概形。

**定义 A.2.** 若一个范畴 `C` 与某个 `\mathbb U`-小范畴等价，则称
`C` **本质 `\mathbb U`-小**。一个 `\mathbb U`-小骨架是 full
subcategory `C_0\subset C`，它与 `C` 等价且每个同构类恰含一个
`C_0` 中的对象。

**命题 A.3.** 若 `S` 是 `\mathbb U`-小 Noetherian 概形，则 `S` 上有限型
`\mathbb U`-小概形的范畴本质 `\mathbb U`-小。因此
`\operatorname{Sm}_S` 可取 `\mathbb U`-小骨架。

**证明.** 因 `S` Noetherian，可固定有限仿射开覆盖
`S=\bigcup_{a=1}^m\operatorname{Spec}R_a`。若 `X\to S` 有限型，则 `X`
Noetherian 且 quasi-compact。把各个 `X\times_S\operatorname{Spec}R_a`
取有限仿射开覆盖，并把两两交叠再取有限个主开集细分，可把 `X\to S`
编码为下列有限数据：有限生成 `R_a`-代数、有限个元素的局部化、这些
局部化之间的环同构，以及有限个 cocycle 等式。

每个 `R_a` 属于 `\mathbb U`。有限生成 `R_a`-代数都是某个
`R_a[t_1,\ldots,t_n]` 的商；理想、有限元组、局部化同态和有限交换图
各自组成 `\mathbb U`-小集合。对有限指标集合取并仍为 `\mathbb U`-小。
故上述全部有限粘合数据组成 `\mathbb U`-小集合。每个有限型
`S`-概形同构于其中一组数据粘合出的概形，所以同构类为
`\mathbb U`-小集合。态射同样由有限个相容环同态编码，故每两个对象间的
Hom 集为 `\mathbb U`-小。选取每个同构类的一个代表即得
`\mathbb U`-小骨架；光滑对象构成 full subcategory，故也可取小骨架。
`\square`

**注 A.4（骨架独立性）.** 本书所有 `\operatorname{Sm}_S` 都指
`\mathbb U`-小骨架。若 `e:C_0\simeq C_1` 是两个骨架间的等价，则预合成
给出 `\operatorname{Fun}(C_1^{op},\mathcal S_{\mathbb U})\simeq
\operatorname{Fun}(C_0^{op},\mathcal S_{\mathbb U})`。该等价把表示对象、
Nisnevich 覆盖筛和 `X\times\mathbb A^1\to X` 送到对应数据；所以它限制为
sheaf 范畴的等价，并由局部化泛性质进一步诱导 motivic 范畴的等价。
这里得到的是范畴等价，不是两个不同骨架的字面相等。六操作的高阶相干
由外部系数系统定理提供，不靠逐个骨架的任意选择来定义。

## A.2 Presheaf categories

**外部输入定理 A.5（presheaf 自由余完备化）.** 若 `C` 是
`\mathbb U`-小 infinity-范畴，则

$$
\mathcal P_{\mathbb U}(C)=
\operatorname{Fun}(C^{op},\mathcal S_{\mathbb U})
$$

是 presentable infinity-category。对任意具有 `\mathbb U`-小余极限的
`\mathcal D`，沿 Yoneda 嵌入 `y:C\to\mathcal P_{\mathbb U}(C)` 预合成给出

$$
\operatorname{Fun}^{L}(\mathcal P_{\mathbb U}(C),\mathcal D)
\simeq \operatorname{Fun}(C,\mathcal D).
$$

特别地，Yoneda 像在 `\mathbb U`-小余极限下生成 presheaf 范畴。
不发生宇宙混淆时，下文简写 `\mathcal P(C)=\mathcal P_{\mathbb U}(C)`。

**外部来源与边界.** Lurie, *Higher Topos Theory*, Theorem 5.1.5.6
与 Corollary 5.1.5.8；presentability 见 Remark 5.5.3.7。本书使用其泛性质，
不重建 straightening、Kan extension 与 accessibility 的基础理论。

**命题 A.6.** 若 `F:\mathcal P(C)\to\mathcal D` 保持小余极限，则 `F` 由其在 representables 上的值唯一确定。

**证明.** 由 A.5，限制函子

$$
\operatorname{Fun}^{L}(\mathcal P_{\mathbb U}(C),\mathcal D)
\longrightarrow\operatorname{Fun}(C,\mathcal D)
$$

是 infinity-范畴等价，因而 fully faithful。故两个保持小余极限的函子若在
representables 上等价，则该等价以可缩选择空间唯一延拓到全部 presheaves；
自然变换的高阶相干也包含在 fully faithful 性中。`\square`

## A.3 Sheafification as localization

**定义 A.7.** 设 `C` 为 `\mathbb U`-小站点。本书的 sheaf 指满足覆盖筛
descent（等价地，满足覆盖的 Cech descent）的 space-valued sheaf。
Sheafification 是左伴随

$$
L:\mathcal P(C)\rightleftarrows\operatorname{Shv}(C):i
$$

其中 `i` 是包含函子。

本书不默认施加 hyperdescent。Hypercomplete objects 组成进一步的 full
subcategory `\operatorname{Shv}(C)^\wedge`；除非正文明确写出
hypercompletion，不把 `\operatorname{Shv}(C)` 与它识别。

**外部输入定理 A.8（高阶 sheafification）.** `\operatorname{Shv}(C)` 是
`\mathcal P_{\mathbb U}(C)` 的 accessible left exact localization，故
presentable。其局部对象条件由覆盖筛 `R\hookrightarrow yX` 给出的
`\mathbb U`-小态射集合

$$
R\to yX
$$

表达：`F` 为 sheaf 当且仅当每个
`\operatorname{Map}(yX,F)\to\operatorname{Map}(R,F)` 为等价。

**外部来源与边界.** Lurie, *Higher Topos Theory*, Proposition 6.2.2.7。
这一定理同时承担 sheafification 的存在性、accessibility 与 left exactness；
本书不以一句“覆盖与有限极限相容”替代其证明。

## A.4 Accessible localization

**定义 A.9.** 设 `\mathcal C` presentable，`W` 为一集合态射。对象 `Z\in\mathcal C` 称为 `W`-local，若对每个 `w:A\to B` 属于 `W`，映射

$$
\operatorname{Map}_{\mathcal C}(B,Z)\to\operatorname{Map}_{\mathcal C}(A,Z)
$$

为等价。

**外部输入定理 A.10（小生成 accessible localization）.** `W`-local objects
组成 presentable reflective subcategory

$$
L_W:\mathcal C\rightleftarrows\mathcal C[W^{-1}]:i.
$$

局部化函子反演由 `W` 生成的 strongly saturated class，且以此为泛性质。

**外部来源与边界.** Lurie, *Higher Topos Theory*, Proposition 5.5.4.15
（存在性、presentability 与局部对象刻画）和 Proposition 5.5.4.20
（保持小余极限函子的泛性质）。本书不重证小对象论证和 accessibility。

**命题 A.11.** 若 `F:\mathcal C\to\mathcal D` 保持小余极限，则 `F` 通过 `L_W` 因子化，当且仅当 `F(w)` 对所有 `w\in W` 为等价。

**证明.** A.10 所引 HTT Proposition 5.5.4.20 给出 fully faithful 函子

$$
\operatorname{Fun}^L(\mathcal C[W^{-1}],\mathcal D)
\hookrightarrow
\operatorname{Fun}^L(\mathcal C,\mathcal D)
$$

其本质像正是反演 `W` 的保持小余极限函子。因此因子化存在当且仅当
`F(w)` 对每个 `w\in W` 为等价；fully faithful 性还说明因子化的选择空间
若非空则可缩。`\square`

## A.5 应用于 A1-局部化

**定义 A.12.** 对 `\mathbb U`-小有限维 Noetherian 基概形 `S`，令

$$
W_{\mathbb A^1}=
\{X\times_S\mathbb A^1_S\to X\mid X\in\operatorname{Sm}_S\}.
$$

其中每个几何对象先经 Yoneda 与 Nisnevich sheafification 视为 sheaf。
由于 `\operatorname{Sm}_S` 已取 `\mathbb U`-小骨架，
`W_{\mathbb A^1}` 是 `\mathbb U`-小集合。

**命题 A.13.** `\mathbf H(S)` 是 presentable。

**证明.** `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 由外部输入
A.8 presentable。A.12 验证了局部化生成态射确为 `\mathbb U`-小集合；对它
应用外部输入 A.10，得到 `\mathbf H(S)` presentable。`\square`

**命题 A.14.** `\mathbb A^1`-local objects 可由值上的条件检测。

**证明.** 这是第二章命题 2.6 的大小闭合版本。对 `p_X:X\times\mathbb A^1\to X`，局部对象条件是

$$
\operatorname{Map}(X,F)\to\operatorname{Map}(X\times\mathbb A^1,F)
$$

为等价。由 Yoneda lemma 分别识别为 `F(X)` 与 `F(X\times\mathbb A^1)`。`\square`

## A.6 本附录小结

全书的局部化构造都依赖同一模式：先固定宇宙并把几何输入降到
`\mathbb U`-小骨架，再在 `\mathbb V` 中的 presentable presheaf/sheaf
范畴中对 `\mathbb U`-小态射集做 accessible localization。这个模式保证
`\mathbf H(S)` 的大小与 presentability 合法；`\mathbf H_*(S)` 的
presentability 另由 under-category 定理得到，而对称幺半稳定化仍需附录 C
记录的外部输入。默认 sheaf 不含未声明的 hypercompletion。

## 练习

**练习 A.1.** 解释为什么 proper class 级别的生成集合会破坏 presentability 论证。

**练习 A.2.** 证明等价小范畴的 presheaf categories 等价。

**练习 A.3.** 写出 sheaf condition 如何成为局部对象条件。

**练习 A.4.** 证明 `W_{\mathbb A^1}` 是集合。

**练习 A.5.** 用命题 A.11 重新证明 realization functor 的 `\mathbb A^1`-不变性要求。
