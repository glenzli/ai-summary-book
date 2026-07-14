# 第五十三章：$\infty$-Cosmos 与模型无关的高阶范畴论

Quasi-categories、complete Segal spaces 和单纯范畴各自实现高阶范畴，但许多伴随、极限与 Kan 延拓定理只依赖它们共有的 enriched 与 isofibration 结构。$\infty$-cosmos 把这些共同公理抽取出来，并在其 homotopy 2-category 中发展模型无关的高阶范畴论。Modules、comma objects 与 weighted limits 由此能一次证明，再应用到不同模型。

本章使用 simplicial enrichment、isofibrations、2-categories 与 profunctors。Cosmos 公理不会被简化为“任意富范畴”；每个模型是否满足 completeness、cotensor 和 flexible weighted limit 条件会明确作为外部验证。

## 53.1 $\infty$-Cosmos 的对象

**定义 53.1.** 一个 $\infty$-cosmos 是带有对象、映射 quasi-categories、isofibrations、equivalences、products、cotensors 和 pullbacks of fibrations 的 simplicially enriched category $\mathcal K$，满足一组稳定性和完备性公理。

**外部输入定理 53.2.** Quasi-categories、complete Segal spaces、marked simplicial sets 的合适 fibrant objects 和某些 simplicial categories 形成 $\infty$-cosmoi。

**定义 53.3.** $\infty$-cosmos 中的对象称为 $\infty$-categories，映射对象记作

$$
\operatorname{map}_{\mathcal K}(A,B).
$$

## 53.2 Homotopy 2-category

**定义 53.4.** $\infty$-cosmos $\mathcal K$ 的 homotopy 2-category $\mathcal K_2$ 有同样对象，Hom category 为

$$
\mathcal K_2(A,B)=h\operatorname{map}_{\mathcal K}(A,B).
$$

**命题 53.5.** 若 $\operatorname{map}_{\mathcal K}(A,B)$ 是 quasi-category，则 $h\operatorname{map}_{\mathcal K}(A,B)$ 是普通范畴。

**证明.** Quasi-category 的同伦范畴由对象为 0-单纯形、态射为 1-单纯形按 2-单纯形生成的同伦关系取商得到。内角填充保证复合存在且结合到同伦；取商后结合律和单位律严格成立，因此为普通范畴。$\square$

## 53.3 等价与 Isofibration

**定义 53.6.** $\infty$-cosmos 中态射 $f:A\to B$ 称为 equivalence，若它在 homotopy 2-category $\mathcal K_2$ 中为等价。

**定义 53.7.** Isofibration 是 $\infty$-cosmos 公理指定的一类 fibration-like maps，要求对等价提升稳定，并在 pullback 下稳定。

**命题 53.8.** Isofibration 的 pullback 仍为 isofibration。

**证明.** 这是 $\infty$-cosmos 公理之一。它确保 slice、comma object 和变基构造保持在允许的 fibrational 类中。$\square$

## 53.4 伴随

**定义 53.9.** $\infty$-cosmos 中的 adjunction 是 homotopy 2-category $\mathcal K_2$ 中的 adjunction，即 1-态射

$$
f:A\rightleftarrows B:u
$$

配单位和余单位 2-态射，满足三角恒等式。

**外部输入定理 53.10.** 在 $\infty$-cosmos 中，homotopy 2-category 中的 adjunction 等价于 quasi-categorical adjunction 的多种模型定义。

**命题 53.11.** 设 $f:A\rightleftarrows B:u$ 是 $\infty$-cosmos 所表示的 $\infty$-categories 之间的伴随。若 $D:J\to A$ 有 colimit $c$，则 $fc$ 是 $fD$ 的 colimit。

**证明.** 对每个 $y\in B$，伴随与 colimit 的映射空间刻画给出

$$
\operatorname{Map}_B(fc,y)
\simeq
\operatorname{Map}_A(c,uy)
\simeq
\lim_{j\in J^{op}}\operatorname{Map}_A(Dj,uy)
\simeq
\lim_{j\in J^{op}}\operatorname{Map}_B(fDj,y).
$$

这正是 $fc$ 的 colimit 泛性质。若只在 Hom categories $\mathcal K_2(-,-)$ 中写普通集合式同构，会丢失映射空间层的高阶相干；这里的等价发生在 spaces 中。$\square$

## 53.5 Modules 与 weighted limits

**定义 53.12.** 在 $\infty$-cosmos 中，module 或 profunctor $M:A\nrightarrow B$ 可由 span 型或 comma 型对象表示，抽象地扮演从 $A$ 到 $B$ 的 bimodule。

**外部输入定理 53.13.** $\infty$-cosmos 中可发展 modules、collages、weighted limits 和 Kan extensions 的模型无关理论，并恢复 quasi-category 口径下的相应构造。

**命题 53.14.** Representable module 由态射 $f:A\to B$ 诱导。

**证明.** 普通范畴中函子 $f:A\to B$ 诱导 profunctor $B(f-, -)$。在 $\infty$-cosmos 中，用映射 quasi-category 或 comma object 替代 Hom 集，得到对象对 $(a,b)$ 的映射空间 $\operatorname{map}_B(fa,b)$。该构造对 $A$ 反变、对 $B$ 协变，并满足 module 的相干性，因此给 representable module。$\square$

## 53.6 模型无关性的意义

**命题 53.15.** 若两个 $\infty$-category 模型之间有保持 simplicial enrichment、isofibrations、equivalences、cotensors 与指定 flexible limits 的 cosmos equivalence，则在这些结构中表述的伴随、极限和 Kan 延拓定理可跨模型转移。

**证明.** $\infty$-cosmos 等价保持对象、映射 quasi-categories、equivalences、isofibrations 以及由这些结构定义的 comma、slice、adjunction 和 limit。若某定理的陈述和证明只使用这些保持结构，则沿等价运输后仍成立。$\square$

**命题 53.16.** $\infty$-cosmos 中的 equivalences 满足 $2$-out-of-$3$。

**证明.** 按定义，$f$ 是 equivalence 当且仅当它在 homotopy 2-category $\mathcal K_2$ 中为等价。任意 2-category 中的等价 1-态射满足 $2$-out-of-$3$：若 $f,g$ 为等价，则 $gf$ 的拟逆由拟逆反向复合给出；若 $gf$ 与 $f$ 为等价，则 $g\simeq (gf)f^{-1}$ 为等价；另一种情形同理。因此 $\mathcal K$ 中 equivalences 也满足 $2$-out-of-$3$。$\square$

## 53.7 跨模型的高阶范畴演算

$\infty$-cosmos 为高阶范畴论提供模型无关的公理框架。它保留足以讨论伴随、极限、isofibrations、modules 和 weighted limits 的结构，同时避免在每个定理中重新选择 quasi-category、Segal space 或 simplicial category 模型。

## 练习

**练习 53.1.** 定义 $\infty$-cosmos。

**练习 53.2.** 举出 $\infty$-cosmos 的模型来源。

**练习 53.3.** 定义 homotopy 2-category $\mathcal K_2$。

**练习 53.4.** 证明 quasi-category 的同伦范畴是普通范畴。

**练习 53.5.** 定义 $\infty$-cosmos 中的 equivalence。

**练习 53.6.** 定义 isofibration。

**练习 53.7.** 说明 isofibration pullback 稳定。

**练习 53.8.** 定义 $\infty$-cosmos 中的 adjunction。

**练习 53.9.** 证明左伴随保持表示性 colimit。

**练习 53.10.** 定义 module/profunctor。

**练习 53.11.** 说明态射如何诱导 representable module。

**练习 53.12.** 解释 $\infty$-cosmos 的模型无关意义。

**练习 53.13.** 证明 $\infty$-cosmos 中 equivalences 满足 $2$-out-of-$3$。
