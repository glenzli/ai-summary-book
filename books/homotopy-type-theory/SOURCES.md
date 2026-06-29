# 资料源与引用规则

本书不是泛泛综述。每个核心定义、定理和证明路线都应能追溯到正式数学资料或公开形式化库。涉及“最新”状态的条目按 2026-06-29 联网核查；后续扩写必须重新核查可能变化的版本信息。

## 核心来源

1.  The Univalent Foundations Program, *Homotopy Type Theory: Univalent Foundations of Mathematics*, Institute for Advanced Study, 2013.  
    链接：<https://homotopytypetheory.org/book/>  
    用途：本书基础主线来源。包括 intensional type theory、identity types、equivalences、univalence、higher inductive types、homotopy levels、sets、categories 和 synthetic homotopy theory 的经典教材口径。

2.  Egbert Rijke, *Introduction to Homotopy Type Theory*, arXiv:2212.11082.  
    链接：<https://arxiv.org/abs/2212.11082>  
    用途：基础规则、等价、单值性、同伦层级和教材化组织的重要参考。正文若采用其具体引理，应核对章节和命题口径。

3.  HoTT/Coq-HoTT, public Coq library for Homotopy Type Theory.  
    链接：<https://github.com/HoTT/Coq-HoTT>  
    2026-06-29 核查：GitHub 页面显示该库为 HoTT 的 Coq 库，latest release 为 V9.1，日期为 2026-06-24。  
    用途：HoTT 的 Coq 形式化库；用于核查基础路径代数、等价、同伦层级、HIT 相关公理化结构和许多标准定理的机器形式化状态。

4.  Andrej Bauer, Jason Gross, Peter LeFanu Lumsdaine, Michael Shulman, Matthieu Sozeau, Bas Spitters, *The HoTT Library: A formalization of homotopy type theory in Coq*, CPP 2017 / arXiv.  
    链接：<https://arxiv.org/abs/1610.04591>  
    用途：Coq-HoTT 库的设计、形式化范围和技术取舍；用于理解 Coq 中 universe、typeclass、路径代数和等价库的口径。

5.  UniMath project, public Coq library for univalent mathematics.  
    链接：<https://github.com/UniMath/UniMath>  
    用途：单值基础下的代数、范畴论、同伦层级和形式化数学主资料源之一。正文引用 UniMath 时必须说明其采用的基础口径与 Coq-HoTT 的差异。

6.  Cyril Cohen, Thierry Coquand, Simon Huber, Anders Mörtberg, *Cubical Type Theory: a constructive interpretation of the univalence axiom*, arXiv:1611.02108.  
    链接：<https://arxiv.org/abs/1611.02108>  
    用途：cubical type theory 和计算单值性的基础来源；本书在 cubical 专章中作为外部元理论输入。

7.  Agda documentation, Cubical mode.  
    链接：<https://agda.readthedocs.io/en/latest/language/cubical.html>  
    2026-06-29 核查：官方文档版本为 Agda 2.9.0，Cubical 章节说明 interval、path types、Glue types、higher inductive types 和 cubical variants。  
    用途：Agda 中 cubical features、path types 和 univalence 相关语言支持的官方文档。涉及软件行为时必须以官方文档为准。

8.  agda/cubical, Cubical Agda library.  
    链接：<https://github.com/agda/cubical>  
    2026-06-29 核查：GitHub 页面称其为 Cubical Agda 的 experimental library。  
    用途：Cubical Agda 下单值性、高阶归纳类型、合成同伦论、代数和范畴论的机器形式化资料源。

9.  Chris Kapulkin, Peter LeFanu Lumsdaine, Vladimir Voevodsky, *The Simplicial Model of Univalent Foundations*, arXiv:1211.2851.  
    链接：<https://arxiv.org/abs/1211.2851>  
    用途：单值性的一致性和 simplicial set 模型背景；在本书中作为外部模型论输入，不作为内部证明。

10.  Michael Shulman, *Univalence for inverse diagrams and homotopy canonicity*, Mathematical Structures in Computer Science / arXiv.  
     链接：<https://arxiv.org/abs/1203.3253>  
     用途：单值性的模型论和同伦规范性背景；用于高级元理论讨论。

11.  1Lab, Agda-formalized reference for univalent mathematics.  
     链接：<https://1lab.dev/>  
     用途：可浏览的 Agda 形式化数学参考；适合核查定义网络和形式化口径。正文引用时必须区分 1Lab 的 formalized exposition 与传统论文来源。

12.  nLab, Homotopy Type Theory and related entries.  
     链接：<https://ncatlab.org/nlab/show/homotopy+type+theory>  
     用途：发现术语和交叉引用；不得作为核心定理唯一来源。

13.  Ian Orton, Andrew M. Pitts, *Decomposing the Univalence Axiom*, arXiv:1712.04890.  
     链接：<https://arxiv.org/abs/1712.04890>  
     用途：把单值性分解为更易在 cubical/set 模型中验证的原则；用于第十六章的模型论边界。

14.  Axel Ljungström, Anders Mörtberg, *Computational Synthetic Cohomology Theory in Homotopy Type Theory*, arXiv:2401.16336.  
     链接：<https://arxiv.org/abs/2401.16336>  
     用途：Cubical Agda 中合成上同调和上同调环的机器形式化案例；用于第十二章近期形式化入口。

15.  Felix Cherubini, Thierry Coquand, Matthias Hutzler, *A Foundation for Synthetic Algebraic Geometry*, arXiv:2307.00073.  
     链接：<https://arxiv.org/abs/2307.00073>  
     用途：合成代数几何与类型论基础的近期研究边界；不得作为基础 HoTT 定理使用。

16.  Peter LeFanu Lumsdaine, Michael A. Warren, *The local universes model: an overlooked coherence construction for dependent type theories*, arXiv:1411.1736.  
     链接：<https://arxiv.org/abs/1411.1736>  
     用途：依赖类型论模型中的 coherence 和 strictification 背景；用于第十六章讨论模型与语法严格化。

17.  Brunerie number formalization resources, HoTT and Cubical Agda libraries.  
     链接：<https://github.com/HoTT/Coq-HoTT>，<https://github.com/agda/cubical>  
     用途：合成同伦论深层计算的形式化背景；引用时必须核查具体库路径和公理口径。

18.  Daniel Gratzer, Michael Shulman, Jonathan Sterling, *Strict universes for Grothendieck topoi*, arXiv:2202.12012.  
     链接：<https://arxiv.org/abs/2202.12012>  
     用途：univalent/cubical 模型中 universe hierarchy、realignment 和语义稳定性背景；用于第十六章高级模型讨论。

## 版本化形式化库快照

以下快照按 2026-06-29 核查，用于附录 S 的模块索引。后续引用若依赖具体 theorem name，应固定到这些 commit 或更新本节。

1.  HoTT/Coq-HoTT commit `a030184c0bfc9d61f3bcd33c67660b800e106427`。  
    链接：<https://github.com/HoTT/Coq-HoTT/tree/a030184c0bfc9d61f3bcd33c67660b800e106427>  
    用途：路径代数、等价、单值性、截断、HIT 接口、整数、圆的基本群和部分范畴论入口。

2.  UniMath commit `9ed7661d3ad33c74e35824efccf861b4fdc17323`。  
    链接：<https://github.com/UniMath/UniMath/tree/9ed7661d3ad33c74e35824efccf861b4fdc17323>  
    用途：h-level、单值范畴、Yoneda、SIP、displayed categories 和代数结构范畴入口。

3.  agda/cubical commit `92166033326aa59800a580b428125f3c654b5e45`。  
    链接：<https://github.com/agda/cubical/tree/92166033326aa59800a580b428125f3c654b5e45>  
    用途：cubical univalence、HIT、截断、集合商、单值范畴论、Rezk 完备化、群结构路径和 Eilenberg-Mac Lane cohomology 入口。

## 章节依赖映射

- 第 1-4 章主要依赖 HoTT Book、Rijke 教材和 Coq-HoTT/UniMath 中的基础路径代数口径。
- 第 5-8 章主要依赖 HoTT Book、Rijke 教材、Coq-HoTT 和 UniMath 中等价、同伦层级与单值性的标准发展。
- 第 9-12 章主要依赖 HoTT Book 的 HIT 章节、Cubical Agda、Agda 官方文档、cubical type theory 论文和合成上同调形式化论文。
- 第 13-15 章主要依赖 UniMath、Coq-HoTT、1Lab 和 HoTT Book 的范畴论章节。
- 第 16-17 章主要依赖 cubical type theory、simplicial model、Agda 官方文档、cubical model 元理论和最新公开形式化库状态。

## 引用纪律

- 正文不大量转述来源原文；只重写数学内容。
- 若某个定理来自来源但本书暂不证明，必须标注“来源”和“后续依赖程度”。
- 若本书给出自己的证明，需要检查证明是否只使用本书已建立的引理。
- 若不同来源采用不同基础口径，例如 Coq-HoTT 的 HoTT 模式、UniMath 的 univalent foundations、Cubical Agda 的 cubical primitives，必须在正文说明口径差异。
- 涉及软件库和近期研究时，必须写核查日期；当前初始核查日期为 2026-06-29。
