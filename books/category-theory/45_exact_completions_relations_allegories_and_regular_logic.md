# 第四十五章：正合完成、关系、Allegory 与 Regular 逻辑

## 本章目标

本章补足普通范畴论中常被略过但对逻辑、topos 和同调代数很关键的一层：从有限极限范畴到 regular/exact 范畴的完成，以及关系范畴和 allegory 的形式化。正合完成把“等价关系有商”作为泛性质加入；关系演算则把态射推广为子对象，从而连接 regular 逻辑和范畴结构。

## 依赖前置知识

需要有限极限、regular category、effective equivalence relation、子对象、image factorization、bicategory、poset-enriched category、逻辑中的存在量词。

## 45.1 内部关系与复合

**定义 45.1.** 设 $\mathcal C$ 有有限极限。对象 $X,Y$ 之间的关系是乘积 $X\times Y$ 的子对象

$$
R\hookrightarrow X\times Y.
$$

记作 $R:X\nrightarrow Y$。

**定义 45.2.** 若 $\mathcal C$ regular，关系 $R:X\nrightarrow Y$ 与 $S:Y\nrightarrow Z$ 的复合 $S\circ R:X\nrightarrow Z$ 定义为子对象

$$
\exists_{\pi_{XZ}}\bigl(\pi_{XY}^*R\wedge\pi_{YZ}^*S\bigr)\hookrightarrow X\times Z
$$

其中所有投影来自 $X\times Y\times Z$。

**命题 45.3.** 在 $\mathbf{Set}$ 中，上述复合就是通常关系复合。

**证明.** 对 $R\subseteq X\times Y$ 与 $S\subseteq Y\times Z$，拉回到 $X\times Y\times Z$ 后取交得到三元组 $(x,y,z)$，满足 $xRy$ 且 $ySz$。沿 $\pi_{XZ}$ 取 image 即取存在量词，得到 $(x,z)$ 满足存在 $y$ 使 $xRy$ 且 $ySz$。这正是通常关系复合。$\square$

## 45.2 函数作为关系

**定义 45.4.** 态射 $f:X\to Y$ 的图像关系 $\Gamma_f:X\nrightarrow Y$ 是单态

$$
X\xrightarrow{(\operatorname{id}_X,f)}X\times Y.
$$

**命题 45.5.** 对态射 $f:X\to Y$ 与 $g:Y\to Z$，有

$$
\Gamma_g\circ\Gamma_f=\Gamma_{gf}.
$$

**证明.** 复合关系由存在 $y$ 使 $y=f(x)$ 且 $z=g(y)$ 给出。由于 $y$ 被唯一确定为 $f(x)$，该条件等价于 $z=g(f(x))$。因此复合子对象正是 $(\operatorname{id}_X,gf):X\to X\times Z$ 的 image，即 $\Gamma_{gf}$。$\square$

**命题 45.6.** 对任意对象 $X$，$\Gamma_{\operatorname{id}_X}$ 是由对角线 $\Delta_X:X\to X\times X$ 给出的恒等关系；并且对任意关系 $R$ 有 $(R^\circ)^\circ=R$。

**证明.** 由定义，$\Gamma_{\operatorname{id}_X}$ 是

$$
X\xrightarrow{(\operatorname{id}_X,\operatorname{id}_X)}X\times X,
$$

即对角线子对象，所以正是关系范畴中的恒等关系。若 $R\hookrightarrow X\times Y$，则 $R^\circ$ 是沿交换同构 $X\times Y\cong Y\times X$ 得到的子对象；交换同构自反，故再取一次反向回到原来的子对象 $R\hookrightarrow X\times Y$。$\square$

**命题 45.7.** Regular category 中关系复合以对角关系为单位，并且满足结合律；此外若 $R\le R'$ 且 $S\le S'$，则 $S\circ R\le S'\circ R'$。

**证明.** 单位律来自公式

$$
\exists x'\,(x=x'\wedge R(x',y))\;\Longleftrightarrow\;R(x,y)
$$

及其右侧类似式子。范畴上，这是沿对角线拉回后再取 image，所得子对象由 pullback 的泛性质与 image 的唯一性同构于 $R$。

结合律中，$(T\circ S)\circ R$ 与 $T\circ(S\circ R)$ 都解释同一个 regular 公式

$$
\exists y\,\exists z\,(R(x,y)\wedge S(y,z)\wedge T(z,w)).
$$

有限积的结合同构、pullback 的 Beck-Chevalley 性和 regular image 对复合的相容性给出两侧 image 子对象相同。单调性来自 pullback、有限交和 image 运算都保持子对象序。$\square$

## 45.3 Regular completion

**定义 45.8.** 有限极限范畴 $\mathcal C$ 的 regular completion 是 regular category $\mathcal C_{\mathrm{reg}}$ 与保持有限极限的函子

$$
i:\mathcal C\to\mathcal C_{\mathrm{reg}}
$$

满足对任意 regular category $\mathcal R$，预合成给出等价

$$
\operatorname{Reg}(\mathcal C_{\mathrm{reg}},\mathcal R)\simeq\operatorname{Lex}(\mathcal C,\mathcal R),
$$

其中左侧为 regular functors。

**外部输入定理 45.9.** 每个小有限极限范畴有 regular completion，可通过关系、覆盖和有限极限语法构造。

**命题 45.10.** 若 $\mathcal C$ 已 regular，则其 regular completion 等价于 $\mathcal C$。

**证明.** 令 $i=\operatorname{id}_{\mathcal C}$。对任意 regular $\mathcal R$，regular functors $\mathcal C\to\mathcal R$ 是保持有限极限且保持 regular epis 的函子。若 completion 的泛性质要求仅从 lex functors 出发，则已 regular 情形中标准反射把 lex functor 的 regular 保持性作为结构约束。对 $\mathcal C$ 作为 regular 范畴自身，恒等函子满足同一表示性质，因此由表示对象唯一性得到等价。$\square$

## 45.4 Exact category 与正合完成

**定义 45.11.** Regular category $\mathcal E$ 称为 exact category，若每个内部等价关系都是某个态射的 kernel pair。

**定义 45.12.** Regular category $\mathcal C$ 的 exact completion 是 exact category $\mathcal C_{\mathrm{ex}}$ 与 regular functor

$$
j:\mathcal C\to\mathcal C_{\mathrm{ex}}
$$

满足对任意 exact category $\mathcal E$，

$$
\operatorname{Ex}(\mathcal C_{\mathrm{ex}},\mathcal E)\simeq\operatorname{Reg}(\mathcal C,\mathcal E).
$$

**外部输入定理 45.13.** 每个小 regular category 有 exact completion。对象可由 $\mathcal C$ 中的内部等价关系表示，态射由相容关系表示。

**命题 45.14.** 若 $\mathcal C$ 已 exact，则 $\mathcal C_{\mathrm{ex}}\simeq\mathcal C$。

**证明.** Exact completion 由一个反射性泛性质刻画。若 $\mathcal C$ 已 exact，则恒等函子 $\mathcal C\to\mathcal C$ 对任意 exact $\mathcal E$ 给出

$$
\operatorname{Ex}(\mathcal C,\mathcal E)\simeq\operatorname{Reg}(\mathcal C,\mathcal E)
$$

因为 exact functor 正是保持 exact 结构的 regular functor。于是 $\mathcal C$ 自身满足 exact completion 的泛性质，唯一性给出等价。$\square$

## 45.5 Effective equivalence relations

**定义 45.15.** 内部等价关系 $R\rightrightarrows X$ 称为 effective，若存在 $q:X\to Q$，使 $R$ 是 $q$ 的 kernel pair。

**命题 45.16.** 在 exact category 中，每个内部等价关系有稳定商。

**证明.** Exact category 定义保证每个内部等价关系 $R\rightrightarrows X$ 是某个 $q:X\to Q$ 的 kernel pair。Regular 范畴中 regular epi/image factorization 对 pullback 稳定，kernel pair 的构造也由 pullback 给出。因此沿任意 $Y\to Q$ 拉回 $q$ 得到的商仍以拉回后的 $R$ 为 kernel pair，故商稳定。$\square$

## 45.6 Allegory

**定义 45.17.** Allegory 是一个 locally posetal 2-category，带反变恒等于对象的 involution

$$
(-)^\circ:\mathcal A^{op}\to\mathcal A
$$

和满足关系演算公理的交结构。其对象可理解为类型，1-态射为关系。

**外部输入定理 45.18.** Regular category 的关系范畴 $\operatorname{Rel}(\mathcal C)$ 形成 allegory。Exact categories 可由满足额外 tabulation/effectivity 条件的 allegories 表征。

**命题 45.19.** 在 $\operatorname{Rel}(\mathcal C)$ 中，关系的反向由交换乘积因子给出。

**证明.** 关系 $R\hookrightarrow X\times Y$ 的反向 $R^\circ:Y\nrightarrow X$ 是复合

$$
R\hookrightarrow X\times Y\xrightarrow{\tau}Y\times X
$$

的子对象，其中 $\tau$ 为对称交换。同一子对象 $R$ 的元素 $(x,y)$ 被读作 $(y,x)$，故反向两次回到原关系。$\square$

## 45.7 Regular 逻辑的关系演算

**命题 45.20.** Regular 逻辑中的公式

$$
\exists y\,(R(x,y)\wedge S(y,z))
$$

正由关系复合 $S\circ R$ 解释。

**证明.** 定义 45.2 中复合先在 $X\times Y\times Z$ 上取 $\pi_{XY}^*R$ 与 $\pi_{YZ}^*S$ 的交，这解释合取 $R(x,y)\wedge S(y,z)$。再沿 $\pi_{XZ}$ 取 image，即 regular category 中的存在量词 $\exists_y$。因此所得子对象正是该公式的解释。$\square$

## 45.8 本章小结

关系范畴把 regular 逻辑的存在-合取片段几何化；regular completion 自由加入 image 和 regular epi 结构；exact completion 自由加入等价关系的有效商；allegory 把关系演算抽象为 poset-enriched 2-范畴。它们共同说明：正合性不是附加技术条件，而是逻辑商、关系和存在量词稳定性的范畴表达。

## 练习

**练习 45.1.** 定义 $\mathcal C$ 中的关系 $R:X\nrightarrow Y$。

**练习 45.2.** 写出 regular category 中关系复合的公式。

**练习 45.3.** 证明集合中该公式恢复通常关系复合。

**练习 45.4.** 定义态射的图像关系。

**练习 45.5.** 证明 $\Gamma_g\circ\Gamma_f=\Gamma_{gf}$。

**练习 45.6.** 定义 regular completion 的泛性质。

**练习 45.7.** 定义 exact category。

**练习 45.8.** 定义 exact completion。

**练习 45.9.** 证明已 exact 范畴的 exact completion 等价于自身。

**练习 45.10.** 定义 effective equivalence relation。

**练习 45.11.** 说明 exact category 中等价关系商为何稳定。

**练习 45.12.** 定义 allegory。

**练习 45.13.** 描述关系的反向 $R^\circ$。

**练习 45.14.** 用关系复合解释 regular 公式 $\exists y(R(x,y)\wedge S(y,z))$。

**练习 45.15.** 证明对角关系是关系复合的单位。

**练习 45.16.** 说明关系复合的结合律为何等价于两个存在量词次序给出同一 regular 公式。
