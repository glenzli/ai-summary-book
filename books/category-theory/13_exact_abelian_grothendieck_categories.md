# 第十三章：正规、正合、阿贝尔和 Grothendieck 范畴

“正合”在不同语境中并非同一个定义。正规范畴用稳定的像分解控制内部等价关系，Barr 正合范畴再要求这些关系有效；加性与阿贝尔范畴则以零对象、核和余核描述同调代数中的短正合列。Grothendieck 范畴进一步加入 AB5 与生成元，使滤过余极限和导出函子具有良好性质。本章把这些层次放在同一条蕴含链上，同时给出反例说明它们不能随意互换。

所需工具是有限极限与余极限、核/余核的泛性质和生成元。各类正合性的假设会在定义处重新列出；凡调用嵌入定理、足够内射对象或导出范畴结论，都明确标记为外部输入而不从有限极限形式主义中误推。

## 13.1 零对象、核与余核

**定义 13.1.** 范畴 $\mathcal C$ 的零对象是既始又终的对象，记作 $0$。若 $\mathcal C$ 有零对象，则任意 $X,Y$ 之间有零态射

$$
0_{X,Y}:X\to 0\to Y.
$$

**定义 13.2.** 在有零对象的范畴中，态射 $f:X\to Y$ 的核是等化子

$$
\ker(f)\to X
$$

等化 $f$ 与零态射 $0_{X,Y}$。余核是对偶概念，即 $f$ 与零态射的余等化子

$$
Y\to\operatorname{coker}(f).
$$

## 13.2 加性与阿贝尔范畴

**定义 13.3.** 范畴 $\mathcal A$ 称为预加性范畴，若每个 Hom 集是阿贝尔群且复合双线性。若它还有有限 biproduct，则称为加性范畴。

**定义 13.4.** 加性范畴 $\mathcal A$ 称为阿贝尔范畴，若：

1. 每个态射有核和余核；
2. 每个单态射是某个态射的核；
3. 每个满态射是某个态射的余核；
4. 对任意态射 $f$，典范态射
   $$
   \operatorname{coim}(f)\to\operatorname{im}(f)
   $$
   是同构。

**例子 13.5.** $\mathbf{Ab}$、环 $R$ 上左模范畴 $R\text{-}\mathbf{Mod}$、以及小范畴上的阿贝尔群值函子范畴都是阿贝尔范畴。

**命题 13.6.** 在阿贝尔范畴中，短正合列

$$
0\to A\xrightarrow{f}B\xrightarrow{g}C\to0
$$

表示 $f$ 是 $g$ 的核且 $g$ 是 $f$ 的余核。

**证明.** 这是阿贝尔范畴中正合性的定义展开：在 $B$ 处正合意为 $\operatorname{im}(f)\cong\ker(g)$；左端 $0\to A$ 正合给出 $f$ 单，右端 $C\to0$ 正合给出 $g$ 满。结合核-余核刻画得到陈述。$\square$

**定义 13.7.** 对阿贝尔范畴中的态射 $f:A\to B$，定义

$$
\operatorname{coim}(f)=\operatorname{coker}(\ker f),
\qquad
\operatorname{im}(f)=\ker(\operatorname{coker}f).
$$

阿贝尔范畴公理要求典范态射 $\operatorname{coim}(f)\to\operatorname{im}(f)$ 为同构。

**命题 13.8.** 在阿贝尔范畴中，态射 $f:A\to B$ 可分解为

$$
A\twoheadrightarrow\operatorname{im}(f)\hookrightarrow B,
$$

其中第一箭头为满态射，第二箭头为单态射。

**证明.** 由定义 13.7 和阿贝尔范畴公理，

$$
\operatorname{coim}(f)\cong\operatorname{im}(f).
$$

商映射 $A\to\operatorname{coim}(f)=\operatorname{coker}(\ker f)$ 是余核，故为满态射；包含 $\operatorname{im}(f)=\ker(\operatorname{coker}f)\to B$ 是核，故为单态射。典范态射使 $f$ 等于这两个态射的复合。$\square$

**定义 13.9.** 阿贝尔范畴之间的加性函子 $F:\mathcal A\to\mathcal B$ 称为左正合，若它保持有限极限，等价地保持核和左正合列；称为右正合，若它保持有限余极限，等价地保持余核和右正合列；称为正合，若它既左正合又右正合。

## 13.3 正规与正合范畴

**定义 13.10.** 有有限极限的范畴称为正规范畴（regular category），若每个态射可分解为正规满射后接单射，且正规满射在拉回下稳定。

**定义 13.11.** 正规范畴称为 Barr-正合范畴，若每个等价关系都是某个态射的核偶。

**例子 13.12.** $\mathbf{Set}$ 是 Barr-正合范畴；任意阿贝尔范畴的底层有限极限结构给出正合性良好的环境。

## 13.4 Grothendieck 范畴

**定义 13.13.** Grothendieck 范畴是阿贝尔范畴 $\mathcal A$，满足：

1. $\mathcal A$ 有所有小余极限；
2. 滤过余极限正合，即满足 AB5；
3. $\mathcal A$ 有生成元。

**例子 13.14.** $R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴；Grothendieck 站点上的阿贝尔群值 sheaf 范畴也是 Grothendieck 范畴。

**命题 13.15.** 对任意环 $R$，$R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴。

**证明.** 模范畴是阿贝尔范畴，核、余核由底层阿贝尔群中的核、商模给出。它有所有小余极限，逐底层集合构造后加上诱导的 $R$-作用即可。滤过余极限在模范畴中由底层集合滤过余极限计算；滤过余极限在 $\mathbf{Set}$ 中与有限极限相容，并且模中的加法与 $R$-作用逐元素定义，因此短正合列的滤过余极限仍短正合。这给出 AB5。

最后，$R$ 作为左 $R$-模是生成元：若 $f,g:M\rightrightarrows N$ 不同，取 $m\in M$ 使 $f(m)\ne g(m)$。由 $R\to M,\ r\mapsto rm$ 得到一个从生成元出发的态射检测 $f\ne g$。故 $R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴。$\square$

**外部输入定理 13.16（Gabriel-Popescu）.** 若 $\mathcal A$ 是 Grothendieck 范畴且 $G$ 是生成元，则 $\mathcal A$ 等价于某个模范畴的局部化。更精确地，存在环 $R=\operatorname{End}(G)$ 及正合左伴随局部化

$$
R\text{-}\mathbf{Mod}\to\mathcal A.
$$

该定理是 Grothendieck 范畴结构理论的核心输入，本书不在此证明。

## 13.5 判别式、稳定性与边界例子

**命题 13.17.** 在有零对象和核的范畴中，任意核态射都是单态射。对偶地，在有余核的范畴中，任意余核态射都是满态射。

**证明.** 核是 $f$ 与零态射的等化子。任意等化子都是单态射：若 $e:E\to X$ 是等化子且 $eu=ev$，则 $u$ 与 $v$ 都是从同一对象到 $E$ 的态射，其经 $e$ 后相等；由等化子的唯一性得 $u=v$。余核情形对偶。$\square$

**命题 13.18.** 在阿贝尔范畴中，态射 $f:A\to B$ 为单态射，当且仅当 $\ker f\cong0$；为满态射，当且仅当 $\operatorname{coker}f\cong0$。

**证明.** 若 $f$ 单，则 $0\to A$ 等化 $f$ 与零态射并满足核的泛性质，因此 $\ker f\cong0$。反过来，设 $\ker f=0$，且 $u,v:T\rightrightarrows A$ 满足 $fu=fv$。由于 Hom 集为阿贝尔群，$f(u-v)=0$。由核的泛性质，$u-v$ 唯一分解过 $\ker f=0$，故 $u-v=0$，即 $u=v$。所以 $f$ 为单态射。满态射判别为对偶命题。$\square$

**命题 13.19.** 正合函子 $F:\mathcal A\to\mathcal B$ 保持 image 与 coimage：对任意 $f:A\to B$，有自然同构

$$
F(\operatorname{im}f)\cong\operatorname{im}(Ff),
\qquad
F(\operatorname{coim}f)\cong\operatorname{coim}(Ff).
$$

**证明.** 正合函子保持核和余核。于是

$$
F(\operatorname{im}f)
=F(\ker(\operatorname{coker}f))
\cong
\ker(\operatorname{coker}(Ff))
=\operatorname{im}(Ff).
$$

coimage 的公式

$$
\operatorname{coim}f=\operatorname{coker}(\ker f)
$$

同理。$\square$

**例子 13.20（阿贝尔但非 Grothendieck）.** 有限阿贝尔群范畴 $\mathbf{Ab}_{\mathrm{fin}}$ 是阿贝尔范畴：有限阿贝尔群同态的核、余核、image 和 coimage 仍为有限阿贝尔群。但它不是 Grothendieck 范畴，因为它没有所有小余积。例如可数多个 $\mathbb Z/2$ 的余积在 $\mathbf{Ab}$ 中是无限直和，不是有限阿贝尔群。

## 13.6 正合性的层次

阿贝尔范畴把模范畴中的核、余核、image/coimage 和正合列抽象化；其中单满态射可由核和余核判别。Grothendieck 范畴进一步加入余完备性、滤过余极限正合性和生成元，是 sheaf 同调和导出范畴的基本环境。正规和 Barr-正合范畴则在非加性环境中保留正合商的控制。

## 练习

**练习 13.1.** 证明有零对象的范畴中零态射唯一。

**练习 13.2.** 在 $\mathbf{Ab}$ 中写出群同态 $f:A\to B$ 的核和余核。

**练习 13.3.** 证明加性范畴中的二元 biproduct 同时是积和余积。

**练习 13.4.** 给出一个非阿贝尔的加性范畴例子。

**练习 13.5.** 查阅 AB3、AB4、AB5 条件，并说明 Grothendieck 范畴使用哪一个。

**练习 13.6.** 在 $\mathbf{Ab}$ 中计算 $\operatorname{coim}(f)$ 与 $\operatorname{im}(f)$，并说明二者为何同构。

**练习 13.7.** 证明任意核态射都是单态射。

**练习 13.8.** 说明正合函子为什么保持短正合列。

**练习 13.9.** 证明 $R$ 作为左 $R$-模是 $R\text{-}\mathbf{Mod}$ 的生成元。

**练习 13.10.** 解释 Gabriel-Popescu 定理中“局部化”为什么可看作从模范畴到 Grothendieck 范畴的表示。

**练习 13.11.** 证明任意等化子都是单态射，任意余等化子都是满态射。

**练习 13.12.** 在 $\mathbf{Ab}$ 中验证命题 13.18：群同态单当且仅当核为零，满当且仅当余核为零。

**练习 13.13.** 证明正合函子保持 image。

**练习 13.14.** 证明有限阿贝尔群范畴没有可数余积。
