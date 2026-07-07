# 附录 C：dg quotient、Morita localization 与 perfect modules

## C.1 dg quotient

**定义 C.1.** 设 $\mathcal A$ 是 dg category，$\mathcal B\subset\mathcal A$ 是 full dg subcategory。dg quotient
$$
\mathcal A/\mathcal B
$$
是把 $\mathcal B$ 中对象在同伦意义下强制变成零对象的 dg category。其同伦范畴应模型化 Verdier quotient
$$
H^0(\mathcal A)/H^0(\mathcal B)
$$
在合适 pretriangulated 假设下的增强。

**外部输入定理 C.2（Drinfeld dg quotient）.** Drinfeld 构造的 dg quotient 满足上述泛性质，并与 Verdier quotient 兼容到适当增强层级。

## C.2 Morita localization

**定义 C.3.** dg functor $F:\mathcal A\to\mathcal B$ 是 Morita equivalence，若
$$
\operatorname{Perf}(\mathcal A)\to\operatorname{Perf}(\mathcal B)
$$
是 quasi-equivalence。

**命题 C.4.** quasi-equivalence 蕴含 Morita equivalence。

**证明.** quasi-equivalence 诱导 representable modules 的 quasi-equivalence，并保持 shifts、cones 和 direct summands 生成的 perfect subcategories。因此在 perfect module categories 上得到 quasi-equivalence。证毕。

## C.3 Perfect modules

**定义 C.5.** 右 $\mathcal A$-module $M$ 称为 perfect，若它属于 representable modules 在 shifts、cones、有限直和和 direct summands 下生成的厚子范畴。

**命题 C.6.** 若 $\mathcal A$ 由 full subcategory $\mathcal G$ split-generate，则 inclusion $\mathcal G\hookrightarrow\mathcal A$ 是 Morita equivalence。

**证明.** split-generation 表示 $\mathcal A$ 的所有 representables 属于 $\mathcal G$ 的 representables 的厚闭包。因 perfect modules 由 representables 厚生成，两个 perfect module categories 相同到 quasi-equivalence。证毕。

## C.4 HMS 用途

HMS 中的 localization 主要出现在：

- stop removal：quotient by linking disks；
- singularity category：$\mathrm D^b\operatorname{Coh}/\operatorname{Perf}$；
- Orlov functors：连接 quotient、matrix factorization 和 coherent sheaf categories；
- Viterbo functor：在 module categories 上表现为 localization 或 homological epimorphism。

## 本附录小结

dg quotient 和 Morita localization 是把几何“移除”“限制”“塌缩”翻译成范畴语言的基础。正式 HMS 陈述应优先在 Morita 层面处理 quotient，而不是只写三角范畴商。
