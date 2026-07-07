# 第三章：Tate sphere、P1-稳定化与 SH(S)

## 本章目标

本章从 pointed motivic spaces 进入稳定 motivic homotopy theory。关键步骤是引入 Tate sphere `T`，并对 `T` 做稳定化，得到 `\mathbf{SH}(S)`。本章证明可以由 pointed cofiber calculus 得到的基本等价，并把稳定化的存在性和模型比较标记为外部输入。

## 依赖前置知识

需要第二章的 `\mathbf H_*(S)`、pointed colimits、smash product、cofiber、presentable symmetric monoidal infinity-categories 和谱对象稳定化。

## 3.1 Pointed smash product

**外部输入定理 3.1.** `\mathbf H(S)` 的 Cartesian monoidal structure 诱导 `\mathbf H_*(S)` 上的 smash product

$$
X\wedge Y=(X\times Y)/(X\vee Y),
$$

并使 `\mathbf H_*(S)` 成为 presentable symmetric monoidal infinity-category，且张量积分别保持小余极限。

**依赖源.** 这是 pointed presentable infinity-categories 和 motivic localization 与乘积相容的标准结论；后续需在附录 C 中定位。

**定义 3.2.** Simplicial circle 记为

$$
S^{1,0}=\Delta^1/\partial\Delta^1
$$

视为常值 pointed motivic space。

**定义 3.3.** `\mathbb G_m` 以单位截面 `1:S\to\mathbb G_m` 为基点。定义

$$
S^{1,1}=S^{1,0}\wedge\mathbb G_m.
$$

**定义 3.4.** Tate sphere 定义为

$$
T=\mathbb A^1/(\mathbb A^1\setminus0)=\mathbb A^1/\mathbb G_m
$$

在 `\mathbf H_*(S)` 中的对象。

## 3.2 Tate sphere 的基本等价

**命题 3.5.** 在 `\mathbf H_*(S)` 中有自然等价

$$
T\simeq S^{1,0}\wedge\mathbb G_m.
$$

**证明.** 在 pointed motivic spaces 中，`T` 是包含映射

$$
\mathbb G_m\hookrightarrow\mathbb A^1
$$

的 cofiber。第二章推论 2.10 给出 `\mathbb A^1\simeq *`。因此该 cofiber 等价于映射 `\mathbb G_m\to *` 的 cofiber。对任意 pointed object `Y`，映射 `Y\to *` 的 cofiber 是 suspension `\Sigma Y=S^{1,0}\wedge Y`。取 `Y=\mathbb G_m`，得到 `T\simeq S^{1,0}\wedge\mathbb G_m`。`\square`

**注 3.6.** 等价 `T\simeq\mathbb P^1/\infty` 需要把 `\mathbb P^1` 分解为 `\mathbb A^1` 与无穷远点，并验证对应商对象。该证明将在纯性和 Thom spaces 章节中给出坐标级版本。

**定义 3.7.** 对 `p\ge q\ge0`，定义 motivic sphere

$$
S^{p,q}=(S^{1,0})^{\wedge(p-q)}\wedge(\mathbb G_m)^{\wedge q}.
$$

稳定化后允许 `p,q` 为任意整数，此时负指数表示相应可逆对象的逆。

## 3.3 稳定 motivic homotopy category

**外部输入定理 3.8.** 存在稳定 presentable symmetric monoidal infinity-category `\mathbf{SH}(S)` 和保持小余极限的 symmetric monoidal functor

$$
\Sigma_T^\infty:\mathbf H_*(S)\longrightarrow\mathbf{SH}(S)
$$

满足以下泛性质：对任意 stable presentable symmetric monoidal infinity-category `\mathcal C`，若 `F:\mathbf H_*(S)\to\mathcal C` 保持小余极限并把 `T` 送为可逆对象，则 `F` 通过 `\Sigma_T^\infty` 唯一因子化。

**依赖源.** Morel-Voevodsky、Jardine、Ayoub、Cisinski-Deglise、Hoyois 及稳定化的一般理论。模型范畴版本和 infinity-categorical 版本的比较后续单独处理。

**定义 3.9.** `\mathbf{SH}(S)` 称为 `S` 上的 stable motivic homotopy category，或稳定 motivic homotopy infinity-范畴。单位对象写作

$$
\mathbb 1_S=\Sigma_T^\infty S_+.
$$

**定义 3.10.** 对 `X\in\operatorname{Sm}_S`，其 suspension spectrum 定义为

$$
\Sigma_T^\infty X_+\in\mathbf{SH}(S).
$$

**命题 3.11.** `T` 在 `\mathbf{SH}(S)` 中可逆。

**证明.** 这是定理 3.8 泛性质中稳定化的组成部分：`\mathbf{SH}(S)` 是通过迫使 `T` 可逆得到的 symmetric monoidal stabilization。因此 `\Sigma_T^\infty T` 有张量逆。`\square`

**命题 3.12.** `\Sigma_T^\infty` 保持小余极限。

**证明.** 定理 3.8 已把 `\Sigma_T^\infty` 指定为保持小余极限的左伴随型稳定化函子。若用泛性质重述，`\mathbf{SH}(S)` 是 presentable localization/stabilization 的目标，结构函子为左伴随，因此保持小余极限。`\square`

## 3.4 与经典稳定同伦论的差异

**命题 3.13.** `\mathbf{SH}(S)` 至少含有两个相互独立的 suspension 方向：simplicial suspension `S^{1,0}` 和 Tate 方向 `\mathbb G_m`。

**证明.** 命题 3.5 给出被稳定化反演的对象 `T` 分解为 `S^{1,0}\wedge\mathbb G_m`。经典稳定同伦论只需反演 simplicial circle 的 suspension 坐标；motivic 情形中 `\mathbb G_m` 作为代数几何对象参与稳定坐标。除非进一步施加 realization 或特殊基域上的额外等价，`\mathbb G_m` 不能被当作 simplicial circle。`\square`

**例子 3.14.** 当 `S=\operatorname{Spec}\mathbb C` 时，复 Betti realization 会把光滑复概形送到复解析空间，再送到拓扑 spaces。该 functor 与 `\mathbb A^1`-局部化相容，因为 `\mathbb C` 的仿射直线拓扑上可缩；稳定后给出到 classical spectra 的 realization functor。其完整构造和对称幺半性是外部输入，不在此例中证明。

**命题 3.15.** 不能仅由 `\mathbf{SH}(S)` 的定义推出某个 realization functor 是保守的。

**证明.** 保守性要求若 `R(E)\simeq0` 则 `E\simeq0`。稳定化的定义只给出 `R` 若存在必须反演 `T` 并保持相关结构；它不控制 `R` 的核。一个保持余极限的 functor 可以有非零核，因此保守性必须由额外几何或同伦论结果证明。`\square`

## 3.5 紧生成子口径

**外部输入定理 3.16.** 在常用有限性假设下，`\mathbf{SH}(S)` 由对象

$$
\Sigma^{p,q}\Sigma_T^\infty X_+,\qquad X\in\operatorname{Sm}_S,\ p,q\in\mathbb Z
$$

在小余极限和稳定操作下生成；在更精细版本中可取紧生成子。

**依赖源.** Ayoub、Cisinski-Deglise、Hoyois 等资料中的 compact generation 结果。不同基概形假设下表述不同，后续需 locator。

**命题 3.17.** 若 `\mathcal G` 是 stable presentable infinity-category `\mathcal C` 的一组生成子，则态射 `u:E\to F` 是等价，当且仅当对所有 `G\in\mathcal G` 和 `n\in\mathbb Z`，

$$
\operatorname{Map}_{\mathcal C}(\Sigma^nG,u)
$$

为等价。

**证明.** 令 `K` 为 `u` 的 fiber。`u` 为等价当且仅当 `K\simeq0`。若所有映射空间条件成立，则 `\operatorname{Map}(\Sigma^nG,K)` 对所有 `G,n` 可缩。令

$$
\mathcal L=\{A\in\mathcal C\mid \operatorname{Map}_{\mathcal C}(A,K)\simeq *\}.
$$

该 full subcategory 对平移、cofiber 和小余极限封闭，因为 `\operatorname{Map}(-,K)` 把 colimit 变为 limit。它包含所有生成子及其平移，故由生成性得 `\mathcal L=\mathcal C`。取 `A=K`，得 `\operatorname{Map}(K,K)` 可缩，于是恒等态射同伦于零，故 `K\simeq0`。反向由等价态射的 fiber 为零直接得到。`\square`

## 3.6 本章小结

稳定 motivic homotopy theory 通过反演 Tate sphere `T=\mathbb A^1/(\mathbb A^1-0)` 得到。`T` 与 `S^{1,0}\wedge\mathbb G_m` 的等价说明 motivic 稳定化含有拓扑和代数两个 suspension 坐标。`\mathbf{SH}(S)` 的存在性、对称幺半结构、模型比较和紧生成性是深外部输入；本书后续将在此基础上展开六操作。

## 练习

**练习 3.1.** 在 pointed spaces 中证明 `\operatorname{cofib}(Y\to *)\simeq\Sigma Y`。

**练习 3.2.** 说明 `\mathbb G_m` 的基点为什么取单位截面。

**练习 3.3.** 用命题 3.5 写出 `S^{2,1}` 与 `T` 的关系。

**练习 3.4.** 解释为什么 `T` 可逆不自动推出 `S^{1,0}` 与 `\mathbb G_m` 分别可逆，除非在相应稳定化中额外说明。

**练习 3.5.** 证明命题 3.17 中的 full subcategory `\mathcal L` 对 cofiber 封闭。
