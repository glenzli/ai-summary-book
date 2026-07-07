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

`\mathbb U`-小集合和范畴称为小；presentable infinity-categories 默认在 `\mathbb V` 中讨论。

**定义 A.2.** 若一个范畴 `C` 与某个小范畴等价，则称 `C` essentially small。一个小骨架是 full subcategory `C_0\subset C`，使每个对象与唯一一个 `C_0` 中对象同构。

**命题 A.3.** 若 `S` 是 Noetherian 概形，则 `S` 上有限型概形的同构类形成集合。因此 `\operatorname{Sm}_S` 可取小骨架。

**证明.** 有限型 `S`-概形可由有限仿射开覆盖和有限生成 `\mathcal O_S`-代数的粘合数据描述。固定 `S` 后，这些有限数据的同构类落在某个集合中，而不是 proper class。取每个同构类一个代表，即得小骨架。`\square`

**注 A.4.** 本书所有 `\operatorname{Sm}_S` 都指小骨架。不同小骨架之间由等价范畴连接，presheaf、sheaf 和 motivic localization 对该等价不敏感。

## A.2 Presheaf categories

**定理 A.5.** 若 `C` 是小 infinity-范畴，则

$$
\mathcal P(C)=\operatorname{Fun}(C^{op},\mathcal S)
$$

是 presentable infinity-category，并由 representables 在小余极限下生成。

**证明.** Presheaf category 是 `C` 的自由 cocompletion。Yoneda 嵌入 `y:C\to\mathcal P(C)` fully faithful；任意 presheaf 可写成 over-category `C_{/F}` 上 representables 的 colimit。这给出生成性。Presentability 是自由 cocompletion 的标准结论。`\square`

**命题 A.6.** 若 `F:\mathcal P(C)\to\mathcal D` 保持小余极限，则 `F` 由其在 representables 上的值唯一确定。

**证明.** 每个 presheaf `P` 是 representables 的 colimit。保持小余极限的 `F` 把该表达送到 `F(yc)` 的同形 colimit。因此 `F(P)` 由 `F` 在 `yc` 上的值和 colimit 相干唯一确定。`\square`

## A.3 Sheafification as localization

**定义 A.7.** 设 `C` 为小站点。Sheafification 是左伴随

$$
L:\mathcal P(C)\rightleftarrows\operatorname{Shv}(C):i
$$

其中 `i` 是包含函子。

**定理 A.8.** `\operatorname{Shv}(C)` 是 `\mathcal P(C)` 的 accessible left exact localization，故 presentable。

**证明.** Sheaf 条件由覆盖筛 `R\hookrightarrow yX` 给出的映射集合

$$
R\to yX
$$

的局部对象条件表达：`F` 为 sheaf 当且仅当 `\operatorname{Map}(yX,F)\to\operatorname{Map}(R,F)` 为等价。因为 `C` 小，覆盖筛数据构成集合。关于一集合态射的局部化是 accessible localization。Grothendieck topology 的有限极限相容性保证该局部化 left exact。`\square`

## A.4 Accessible localization

**定义 A.9.** 设 `\mathcal C` presentable，`W` 为一集合态射。对象 `Z\in\mathcal C` 称为 `W`-local，若对每个 `w:A\to B` 属于 `W`，映射

$$
\operatorname{Map}_{\mathcal C}(B,Z)\to\operatorname{Map}_{\mathcal C}(A,Z)
$$

为等价。

**定理 A.10.** `W`-local objects 组成 presentable reflective subcategory

$$
L_W:\mathcal C\rightleftarrows\mathcal C[W^{-1}]:i.
$$

**证明.** Presentable infinity-category 关于一集合态射的 Bousfield localization 存在，且为 accessible localization。局部对象 full subcategory 是反射子范畴，左伴随为 localization functor。`\square`

**命题 A.11.** 若 `F:\mathcal C\to\mathcal D` 保持小余极限，则 `F` 通过 `L_W` 因子化，当且仅当 `F(w)` 对所有 `w\in W` 为等价。

**证明.** 由 localization 的泛性质，预合成 `L_W` 给出

$$
\operatorname{Fun}^L(\mathcal C[W^{-1}],\mathcal D)
\hookrightarrow
\operatorname{Fun}^L(\mathcal C,\mathcal D)
$$

其本质像正是反演 `W` 的保持小余极限函子。`\square`

## A.5 应用于 A1-局部化

**定义 A.12.** 对基概形 `S`，令

$$
W_{\mathbb A^1}=
\{X\times_S\mathbb A^1_S\to X\mid X\in\operatorname{Sm}_S\}.
$$

由于 `\operatorname{Sm}_S` 已取小骨架，`W_{\mathbb A^1}` 是集合。

**命题 A.13.** `\mathbf H(S)` 是 presentable。

**证明.** `\operatorname{Shv}_{Nis}(\operatorname{Sm}_S)` 由定理 A.8 presentable。对集合 `W_{\mathbb A^1}` 应用定理 A.10，得到 `\mathbf H(S)` presentable。`\square`

**命题 A.14.** `\mathbb A^1`-local objects 可由值上的条件检测。

**证明.** 这是第二章命题 2.6 的大小闭合版本。对 `p_X:X\times\mathbb A^1\to X`，局部对象条件是

$$
\operatorname{Map}(X,F)\to\operatorname{Map}(X\times\mathbb A^1,F)
$$

为等价。由 Yoneda lemma 分别识别为 `F(X)` 与 `F(X\times\mathbb A^1)`。`\square`

## A.6 本附录小结

全书的局部化构造都依赖同一模式：先把几何输入降到小骨架，再在 presentable presheaf/sheaf 范畴中对一集合态射做 accessible localization。这个模式保证 `\mathbf H(S)`、`\mathbf H_*(S)` 和后续稳定化都有合法的范畴论基础。

## 练习

**练习 A.1.** 解释为什么 proper class 级别的生成集合会破坏 presentability 论证。

**练习 A.2.** 证明等价小范畴的 presheaf categories 等价。

**练习 A.3.** 写出 sheaf condition 如何成为局部对象条件。

**练习 A.4.** 证明 `W_{\mathbb A^1}` 是集合。

**练习 A.5.** 用命题 A.11 重新证明 realization functor 的 `\mathbb A^1`-不变性要求。
