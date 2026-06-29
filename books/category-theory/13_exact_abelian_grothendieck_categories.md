# 第十三章：正规、正合、阿贝尔和 Grothendieck 范畴

## 本章目标

本章概述范畴论中用于代数和同调代数的正合性语言：正规范畴、正合范畴、加性范畴、阿贝尔范畴和 Grothendieck 范畴。

## 依赖前置知识

需要有限极限、余极限、核、余核的泛性质和生成元概念。

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

**定义 13.A.** 对阿贝尔范畴中的态射 $f:A\to B$，定义

$$
\operatorname{coim}(f)=\operatorname{coker}(\ker f),
\qquad
\operatorname{im}(f)=\ker(\operatorname{coker}f).
$$

阿贝尔范畴公理要求典范态射 $\operatorname{coim}(f)\to\operatorname{im}(f)$ 为同构。

**命题 13.B.** 在阿贝尔范畴中，态射 $f:A\to B$ 可分解为

$$
A\twoheadrightarrow\operatorname{im}(f)\hookrightarrow B,
$$

其中第一箭头为满态射，第二箭头为单态射。

**证明.** 由定义 13.A 和阿贝尔范畴公理，

$$
\operatorname{coim}(f)\cong\operatorname{im}(f).
$$

商映射 $A\to\operatorname{coim}(f)=\operatorname{coker}(\ker f)$ 是余核，故为满态射；包含 $\operatorname{im}(f)=\ker(\operatorname{coker}f)\to B$ 是核，故为单态射。典范态射使 $f$ 等于这两个态射的复合。$\square$

**定义 13.C.** 阿贝尔范畴之间的加性函子 $F:\mathcal A\to\mathcal B$ 称为左正合，若它保持有限极限，等价地保持核和左正合列；称为右正合，若它保持有限余极限，等价地保持余核和右正合列；称为正合，若它既左正合又右正合。

## 13.3 正规与正合范畴

**定义 13.7.** 有有限极限的范畴称为正规范畴（regular category），若每个态射可分解为正规满射后接单射，且正规满射在拉回下稳定。

**定义 13.8.** 正规范畴称为 Barr-正合范畴，若每个等价关系都是某个态射的核偶。

**例子 13.9.** $\mathbf{Set}$ 是 Barr-正合范畴；任意阿贝尔范畴的底层有限极限结构给出正合性良好的环境。

## 13.4 Grothendieck 范畴

**定义 13.10.** Grothendieck 范畴是阿贝尔范畴 $\mathcal A$，满足：

1. $\mathcal A$ 有所有小余极限；
2. 滤过余极限正合，即满足 AB5；
3. $\mathcal A$ 有生成元。

**例子 13.11.** $R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴；Grothendieck 站点上的阿贝尔群值 sheaf 范畴也是 Grothendieck 范畴。

**命题 13.D.** 对任意环 $R$，$R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴。

**证明.** 模范畴是阿贝尔范畴，核、余核由底层阿贝尔群中的核、商模给出。它有所有小余极限，逐底层集合构造后加上诱导的 $R$-作用即可。滤过余极限在模范畴中由底层集合滤过余极限计算；滤过余极限在 $\mathbf{Set}$ 中与有限极限相容，并且模中的加法与 $R$-作用逐元素定义，因此短正合列的滤过余极限仍短正合。这给出 AB5。

最后，$R$ 作为左 $R$-模是生成元：若 $f,g:M\rightrightarrows N$ 不同，取 $m\in M$ 使 $f(m)\ne g(m)$。由 $R\to M,\ r\mapsto rm$ 得到一个从生成元出发的态射检测 $f\ne g$。故 $R\text{-}\mathbf{Mod}$ 是 Grothendieck 范畴。$\square$

**外部输入定理 13.12（Gabriel-Popescu）.** 若 $\mathcal A$ 是 Grothendieck 范畴且 $G$ 是生成元，则 $\mathcal A$ 等价于某个模范畴的局部化。更精确地，存在环 $R=\operatorname{End}(G)$ 及正合左伴随局部化

$$
R\text{-}\mathbf{Mod}\to\mathcal A.
$$

该定理是 Grothendieck 范畴结构理论的核心输入，本书不在此证明。

## 13.5 本章小结

阿贝尔范畴把模范畴中的核、余核、正合列抽象化；Grothendieck 范畴进一步加入余完备性、滤过余极限正合性和生成元，是 sheaf 同调和导出范畴的基本环境。正规和 Barr-正合范畴则在非加性环境中保留正合商的控制。

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
