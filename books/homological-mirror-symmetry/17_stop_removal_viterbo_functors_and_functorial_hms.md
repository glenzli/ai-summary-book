# 第十七章：stop removal、Viterbo functor 与 functorial HMS

两个增强范畴分别等价，并不保证几何操作在等价下相容。Stop removal 可能对应 B-side 商，sector inclusion 可能对应 extension，而 Viterbo transfer 的方向又与 sector 的协变 pushforward 不同；若只比较四个顶点，水平函子的核、伴随和 twist 都可能错位。函子化 HMS 要求的是两个范畴值图之间的增强自然等价。本章把第七章的局部化、第十五章的 descent 与第十六章的层模型放在同一个交换方块中，逐一固定 Orlov 与 Viterbo 函子的源、靶和已知外部定理范围。

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

**定义 17.4A（spherical functor 的稳定范畴口径）.** 设
$S:\mathcal C\to\mathcal D$ 是小 stable categories 之间有左右伴随的 exact
functor，右伴随记为 $R$。由 counit 和 unit 构造 twist 与 cotwist
$$
T_S=\operatorname{cofib}(SR\to\operatorname{id}_{\mathcal D}),\qquad
C_S=\operatorname{fib}(\operatorname{id}_{\mathcal C}\to RS).
\tag{17.1}
$$
若 $T_S,C_S$ 均为 autoequivalences，且左右伴随之间的标准比较映射满足
spherical compatibility，则称 $S$ 为 spherical。只验证其中一个 cone
functor 可逆，在没有 compatibility theorem 时不够。

**外部输入定理 17.5（Sylvan 的 partially wrapped Orlov criterion）.** 对
由合适 stop 得到的 Fukaya category，存在从 stop 的 Fukaya category 到
ambient Liouville sector 的 partially wrapped category 的 Orlov functor。
Sylvan 给出几何判据；满足该判据时，此函子在定义 17.4A 的意义下 spherical。
Landau--Ginzburg stop 是该判据的基本来源之一。
来源：Sylvan, *Orlov and Viterbo functors in partially wrapped Fukaya
categories*, arXiv:1908.02317。

## 17.3 Viterbo transfer

**定义 17.6.** 对 Liouville inclusion $U\subset M$，Viterbo transfer 是从大空间 invariants 到小空间 invariants 的映射或 functorial 结构。在 wrapped categories 中，它可表示为
$$
\mathcal W(M)\to\mathcal W(U)
$$
的某种 restriction/localization 型 functor，方向依具体模型固定。

**外部输入定理 17.7（Viterbo localization/homological epimorphism）.** 设
$U\subset M$ 是 Sylvan 模型中的 Liouville subdomain inclusion，且 $U,M$
分别为 Weinstein。则 Viterbo functor 是 homological epimorphism，即传到
module categories 后为 localization。若 $M\setminus U$ 的相关 cobordism
满足 GPS 的 Weinstein 假设，则 GPS 结果给出 genuine localization 的更强
版本。
来源：Sylvan, arXiv:1908.02317；GPS stop removal theorem。

## 17.4 Functorial HMS square 的验证

**定义 17.8.** 一个 functorial HMS square 称为严格验证的，若满足：

1. 四个顶点的 categories 均为明确增强范畴；
2. 两个竖直箭头为已证明 HMS 等价；
3. 两个水平箭头由明确几何操作诱导；
4. 存在增强自然变换或同伦，使方块交换；
5. 所有 localization kernels、adjoints 或 spherical twists 均在两侧匹配。

**命题 17.9.** 若 functorial HMS square 严格验证，则其诱导的 Grothendieck group、Hochschild homology 和 Euler pairings 的方块也交换。

**证明.** 增强 functors 诱导 $K_0$、Hochschild homology 和 Euler pairing 上的映射。自然同伦保证函子诱导的这些映射相等。HMS 等价保持对应不变量，因此得到交换方块。证毕。

局部化方块还会强制两侧的核相互对应，而这比不变量方块交换更接近几何内容。

**命题 17.10（交换局部化方块保持核）.** 设有 stable categories 的方块
$$
\begin{array}{ccc}
\mathcal A_1 & \xrightarrow{q_A} & \mathcal A_2\\
\downarrow E_1 & \Downarrow\eta & \downarrow E_2\\
\mathcal B_1 & \xrightarrow{q_B} & \mathcal B_2,
\end{array}
\tag{17.2}
$$
其中 $E_1,E_2$ 是 equivalences，$q_A,q_B$ 是 exact localizations，且
$\eta:E_2q_A\simeq q_BE_1$。则 $E_1$ 限制为
$$
\ker(q_A)\xrightarrow{\sim}\ker(q_B).
$$

**证明.** 若 $X\in\ker(q_A)$，则由 $\eta$，
$q_BE_1(X)\simeq E_2q_A(X)\simeq0$，故 $E_1(X)\in\ker(q_B)$。对
$E_1$ 的 quasi-inverse 使用逆自然同构，得到反向包含和本质满；映射复形的
全忠实性由 $E_1$ 保持。证毕。

在 stop-removal 方块中，命题 17.10 要求 B-side localization 的核恰对应
linking disks 的厚闭包；只知道四个顶点分别等价，无法得到这一结论。

## 17.5 函子化比较仍需保持的结构

一个交换方块可以进一步带 adjunctions、spherical twists、monoidal products
或 Calabi--Yau traces。顶点的 Morita 等价和水平箭头的自然相容性并不自动
保持这些附加结构；每一种结构都需要相应的 mate transformation 或 trace
compatibility。例如 Orlov functor 的 spherical twist 要在镜像下对应某个
明确的 B-side twist，而不能仅由 $K_0$ 上的反射公式猜定。第二十章会把
wall-crossing 与 BPS 术语也按这种“先固定模型、再比较结构”的原则分开。

函子化 HMS 的基本单位不是四个彼此孤立的等价，而是带指定自然同伦的交换方块。Stop removal 的核由 linking disks 控制，Viterbo functor 在适当 Weinstein 假设下成为模范畴局部化，Orlov functor 的 spherical 性又记录相应 twist；镜像一侧必须逐项保持这些结构。$K_0$ 或 Hochschild 方块的交换只是这种增强相容性的推论，不能反过来恢复原方块。

## 练习

**练习 17.1.** 给出一个 functorial HMS square，并说明四个顶点和四条边。

**练习 17.2.** 解释为什么 Orlov functor 的方向必须在每次使用时重新声明。

**练习 17.3.** 证明命题 17.9。

**练习 17.4.** 比较 stop removal 和 Verdier quotient 的形式相似性。
