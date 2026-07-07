# 第十七章：stop removal、Viterbo functor 与 functorial HMS

## 本章目标

本章把第七章的 stopped categories 用于 functorial HMS。重点是 Orlov functor、Viterbo transfer、stop removal、localization 和 B-side functors 的对应关系。

## 依赖前置知识

需要第七章 stops、第十五章 descent、第十六章 microlocal sheaves。

## 17.1 Functoriality 的基本问题

**定义 17.1.** HMS 的 functorial enhancement 不只要求对象层面的等价
$$
\mathcal A\simeq\mathcal B,
$$
还要求几何操作诱导的 functors 在镜像下对应。形式上，它要求一个由几何对象组成的 diagram
$$
A_\bullet
$$
和 B-side diagram
$$
B_\bullet
$$
之间存在 categories-valued functor 的自然等价。

**例 17.2.** 移除 stop、包含 Liouville sector、改变 Landau-Ginzburg fiber、开嵌入和 divisor complement 都可能产生 functorial HMS square。

## 17.2 Orlov functor

**定义 17.3.** 在 Landau-Ginzburg/Calabi-Yau 语境中，Orlov functor 通常指连接 matrix factorization/singularity category 与 coherent sheaf category 的 functor。A-side partially wrapped 语境中，也有从 stop 的 Fukaya category 到 ambient stopped/sector category 的 Orlov 型 functor。

**警告 17.4.** “Orlov functor”不是唯一固定方向的函子。本书使用时必须说明源、靶和几何构造。

**外部输入定理 17.5（partially wrapped Orlov functor）.** 在合适 stop/sector 假设下，stop 诱导的 Orlov functor 可具有 spherical functor 性质，且与 Landau-Ginzburg 模型中的范畴结构相匹配。  
来源：Sylvan, *Orlov and Viterbo functors in partially wrapped Fukaya categories*。

## 17.3 Viterbo transfer

**定义 17.6.** 对 Liouville inclusion $U\subset M$，Viterbo transfer 是从大空间 invariants 到小空间 invariants 的映射或 functorial 结构。在 wrapped categories 中，它可表示为
$$
\mathcal W(M)\to\mathcal W(U)
$$
的某种 restriction/localization 型 functor，方向依具体模型固定。

**外部输入定理 17.7（Viterbo localization/homological epimorphism）.** 对合适 Weinstein domain/subdomain，Viterbo functor 在 module categories 上可表现为 localization 或 homological epimorphism 型性质。  
来源：GPS stop removal 结果和 Sylvan 的 Viterbo functor 工作。

## 17.4 Functorial HMS square 的验证

**定义 17.8.** 一个 functorial HMS square 称为严格验证的，若满足：

1. 四个顶点的 categories 均为明确增强范畴；
2. 两个竖直箭头为已证明 HMS 等价；
3. 两个水平箭头由明确几何操作诱导；
4. 存在增强自然变换或同伦，使方块交换；
5. 所有 localization kernels、adjoints 或 spherical twists 均在两侧匹配。

**命题 17.9.** 若 functorial HMS square 严格验证，则其诱导的 Grothendieck group、Hochschild homology 和 Euler pairings 的方块也交换。

**证明.** 增强 functors 诱导 $K_0$、Hochschild homology 和 Euler pairing 上的映射。自然同伦保证函子诱导的这些映射相等。HMS 等价保持对应不变量，因此得到交换方块。证毕。

## 17.5 研究边界

Functorial HMS 仍是活跃方向。2025 年之后的工作尝试把 BPS categories、topological field theory、wall-crossing 和物理 functorial structures 组织进 HMS。除非完成精确 theorem locator，本书只把这些作为研究边界，不写成基础定理。

## 本章小结

Functorial HMS 要求镜像等价与几何操作相容。Stop removal、Orlov functor 和 Viterbo transfer 是 A-side 的核心操作；B-side 对应 coherent sheaf restriction、matrix factorization functors、singularity quotient 和 localization。严格验证需要增强自然变换，而不只是对象层面的字典。

## 练习

**练习 17.1.** 给出一个 functorial HMS square，并说明四个顶点和四条边。

**练习 17.2.** 解释为什么 Orlov functor 的方向必须在每次使用时重新声明。

**练习 17.3.** 证明命题 17.9。

**练习 17.4.** 比较 stop removal 和 Verdier quotient 的形式相似性。
