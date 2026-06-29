# 第十四章：Yoneda、极限、伴随与 Rezk 完备化

## 本章目标

本章给出 HoTT 中单值范畴论的核心工具：Yoneda 引理、极限、伴随和 Rezk 完备化。长证明在附录 Q、U、X、AA、AF 中展开为证明核或证明架构，并标注形式化来源。

## 依赖前置知识

本章依赖单值范畴、函数外延性、命题截断和集合层数学。Yoneda 引理的证明核见附录 Q；预层范畴和 Yoneda 嵌入 fully faithful 版本见附录 U；Rezk 完备化的构造输入和外部边界见附录 R。

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

**构造 14.10（Rezk 完备化蓝图）.** 本书采用附录 R 的 Yoneda 本质像构造：
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
附录 R.7-R.10 给出 $\widehat{\mathcal C}$ 单值、$\eta_{\mathcal C}$ fully faithful 且 essentially surjective 的证明路线。

**定理 14.11（Rezk 完备化泛性质，证明架构）.** 若 $\mathcal D$ 是单值范畴，则预合成
$$
(-)\circ\eta_{\mathcal C}:
\mathsf{Fun}(\widehat{\mathcal C},\mathcal D)
\to
\mathsf{Fun}(\mathcal C,\mathcal D)
$$
是等价。

**验证状态：证明架构 / 外部机器化义务。** 见附录 R.11 和附录 AA。函子范畴、自然同构和预层范畴单值性已在附录 X 展开；Rezk 泛性质已在附录 AA 降为 weak equivalence 限制函子的等价证明架构，剩余为 AA.8-AA.10 的逐行 transport 和代表元相容计算。

## 本章小结

HoTT 中的范畴论要求把“唯一到唯一同构”改写成“唯一到路径”，这正是单值范畴的作用。Yoneda、极限、伴随和 Rezk 完备化构成单值范畴论的基础工具箱。

## 练习

**练习 14.1.** 写出自然变换的类型，并说明自然性条件为什么是命题。

**练习 14.2.** 证明两个终对象同构。

**练习 14.3.** 对集合范畴，写出二元积的泛性质。

**练习 14.4.** 解释 Rezk 完备化与“把等价对象识别为相等”的关系。
