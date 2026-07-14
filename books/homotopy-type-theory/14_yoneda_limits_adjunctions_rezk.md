# 第十四章：Yoneda、极限、伴随与 Rezk 完备化

一个对象 $c$ 可以由所有射入它的态射共同识别：Yoneda 引理把预层 $P$ 在 $c$ 处的元素，与从可表预层 $\mathcal C(-,c)$ 到 $P$ 的自然变换等同。这个观察一方面把对象嵌入预层范畴，另一方面提供 Rezk 完备化的具体构造：只保留那些“仅仅可表”的预层，就得到对象同构能够转成路径的单值范畴。

本章先完成 Yoneda 的双向构造，再用可收缩 Hom 表述极限、用 Hom 等价表述伴随，最后区分 Rezk 完备化的书内构造与其外部泛性质。全章使用函数外延性、命题截断和第十三章的单值范畴。固定的预范畴 $\mathcal C$ 假设 locally small；$\mathsf{PSh}(\mathcal C)$ 的对象类型通常位于比 Hom 所在小宇宙更高的层级，本章不使用 resizing，也不声称完备化保持对象小性。

## 14.1 预层与 Yoneda

**定义 14.1.** 设 $\mathcal C$ 为预范畴。一个集合值预层是反变函子
$$
P:\mathcal C^{op}\to\mathsf{Set}.
$$

**定义 14.2.** Yoneda 嵌入把对象 $c:\mathcal C$ 送到可表预层
$$
y(c)\coloneqq\mathcal C(-,c).
$$

**定理 14.3（Yoneda 引理）.** 对预层 $P$ 和对象 $c$，有自然等价
$$
\mathsf{Nat}(y(c),P)\simeq P(c).
$$

**证明（书内证明核）.** 见附录 Q.9。正向映射把自然变换 $\alpha:y(c)\Rightarrow P$ 送到 $\alpha_c(\mathsf{id}_c)$；反向映射把 $u:P(c)$ 送到分量
$$
g:x\to c\quad\mapsto\quad P(g)(u).
$$
两侧互逆分别由预层恒等律和自然性条件推出；自然变换相等使用函数外延性和自然性证明的命题性。$\square$

**推论 14.3.1（Yoneda 嵌入 fully faithful）.** Yoneda 嵌入
$$
y:\mathcal C\to\mathsf{PSh}(\mathcal C)
$$
fully faithful。

**证明（书内证明核）.** 见附录 U.11-U.12。Hom 映射把 $h:c\to d$ 送到后复合自然变换
$$
g:x\to c\mapsto h\circ g.
$$
它是附录 Q.10 中 Yoneda Hom 等价的逆方向。$\square$

## 14.2 极限与余极限

**定义 14.4.** 图 $D:J\to\mathcal C$ 的锥由对象 $c$ 与态射族 $c\to D(j)$ 组成，满足自然性条件。极限是锥范畴中的终对象。

**定义 14.5.** 在 HoTT 中，“是终对象”应为命题：
$$
\mathsf{isTerminal}(t)\coloneqq\prod_{x}\mathsf{isContr}(\mathcal C(x,t)).
$$

**命题 14.6（极限唯一性）.** 若极限存在，则极限对象在单值范畴中唯一到相等。

**证明.** 见附录 AF.3-AF.4。两个终对象之间有唯一同构；单值范畴把该同构转为对象路径。$\square$

## 14.3 伴随

**定义 14.7.** 函子 $F:\mathcal C\to\mathcal D$ 与 $G:\mathcal D\to\mathcal C$ 构成伴随，若存在自然等价
$$
\mathcal D(Fc,d)\simeq\mathcal C(c,Gd)
$$
自然于 $c$ 与 $d$。

**命题 14.8（单位余单位形式）.** 上述 Hom 等价形式等价于单位 $\eta:\mathsf{Id}\Rightarrow GF$、余单位 $\epsilon:FG\Rightarrow\mathsf{Id}$ 和三角恒等式。

**证明.** 见附录 AF.5-AF.11。由 Hom 等价取单位为 $\Phi(\mathsf{id})$，余单位为 $\Phi^{-1}(\mathsf{id})$；反向由
$$
f\mapsto G(f)\circ\eta_c,\qquad
g\mapsto\epsilon_d\circ F(g)
$$
给出 Hom 映射，三角恒等式证明它们互逆。自然性和高阶证明分量由 Hom 集合性与自然变换路径原则处理。$\square$

## 14.4 Rezk 完备化

**定义 14.9.** Rezk 完备化把预范畴 $\mathcal C$ 映到单值范畴 $\widehat{\mathcal C}$，并配备 fully faithful 且 essentially surjective 的函子
$$
\mathcal C\to\widehat{\mathcal C}.
$$

**构造 14.10（Yoneda 本质像）.** 本书采用附录 R 的 Yoneda 本质像构造：
$$
\widehat{\mathcal C}_0
\coloneqq
\sum_{P:\mathsf{PSh}(\mathcal C)}
\left\|
\sum_{c:\mathcal C_0}(y(c)=P)
\right\|.
$$
Hom 由预层范畴继承，嵌入函子为
$$
\eta_{\mathcal C}(c)=
\bigl(y(c), |(c,\mathsf{refl}_{y(c)})|\bigr).
$$
附录 R.7-R.10 在书内证明 $\widehat{\mathcal C}$ 单值、$\eta_{\mathcal C}$ fully faithful 且 essentially surjective；其中对本质像见证的截断消去只进入另一个命题截断目标。

**外部输入定理 14.11（Rezk 完备化泛性质）.** 若 $\mathcal D$ 是单值范畴，则预合成函子
$$
\eta_{\mathcal C}^{*}:
[\widehat{\mathcal C},\mathcal D]
\longrightarrow
[\mathcal C,\mathcal D]
$$
是预范畴同构；特别地，其对象函数
$$
\mathsf{Fun}(\widehat{\mathcal C},\mathcal D)
\longrightarrow
\mathsf{Fun}(\mathcal C,\mathcal D)
$$
是类型等价。

**来源与未重证边界.** Ahrens--Kapulkin--Shulman, *Univalent categories and the Rezk completion*, Mathematical Structures in Computer Science 25 (2015), Theorem 8.4，DOI `10.1017/S0960129514000486`。该定理的精确版本是：若 $H:\mathcal A\to\mathcal B$ fully faithful 且 essentially surjective，目标 $\mathcal D$ 单值，则
$(-\circ H):[\mathcal B,\mathcal D]\to[\mathcal A,\mathcal D]$
是预范畴同构。附录 R.9-R.10 已书内证明 $\eta_{\mathcal C}$ 满足这两个假设；附录 AA 解释来源证明中如何用 contractible types 避免从命题截断选择代表。本书不重证 Theorem 8.4 中对象扩张、Hom transport、函子律和代表元独立性的逐项计算，因此它们不再标成书内证明。

## 14.5 表示、唯一性与完备化

Yoneda 把元素恢复为自然变换，极限把唯一性写成 Hom 的可收缩性，伴随把构造间的对应写成自然 Hom 等价。Rezk 完备化则把 Yoneda 的本质像组织成单值范畴；其对象、Hom、嵌入及 weak-equivalence 性质已在书内构造，而“对所有单值目标的限制函子为同构”采用 Theorem 14.11 的精确外部输入。这里始终是 Hom 为集合的一范畴层理论；附录 BB 的 Rezk object 属于合成 $\infty$-范畴语言，名称相近但对象和规则不同。

## 练习

**练习 14.1.** 写出自然变换的类型，并说明自然性条件为什么是命题。

**练习 14.2.** 证明两个终对象同构。

**练习 14.3.** 对集合范畴，写出二元积的泛性质。

**练习 14.4.** 解释 Rezk 完备化与“把等价对象识别为相等”的关系。
