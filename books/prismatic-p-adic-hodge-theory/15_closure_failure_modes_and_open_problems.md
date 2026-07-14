# 第十五章：错误模式、理论边界与开放问题

棱柱理论把许多比较对象放进同一框架，也因此产生一类特别隐蔽的错误：不同 prism 上的 specialization 被写成相等，反演后成立的 Frobenius 同构被提升到积分层，或一个前沿预印本的接口被当作无条件分类定理。辨认这些错误本身是一项数学能力，因为每一种都能追溯到缺失的底环、完备化、有限性或 descent 假设。本章以具体反例模式整理前十四章的边界，随后把仍未解决的问题按“系数与非阿贝尔化、几何化、torsion 与有效性”组织起来。它不再统计写作完成度，而是说明现有理论在哪一步停止，以及继续推进必须增加什么结构。

## 15.1 比较箭头不是对象等同

固定 bounded prism $(A,I)$、适当光滑的 $p$-adic formal scheme
$X/(A/I)$，并记
$$
C_{X/A}=R\Gamma_\Delta(X/A).
$$
第三至十一章出现的 Hodge--Tate、de Rham、crystalline 与 étale 输出，
都由 $C_{X/A}$ 经过不同的换基、Frobenius pullback、反演或 derived fixed
point 构造获得。它们共享源对象，却不因此属于同一系数范畴。

**命题 15.1（比较路径的类型约束）.** 一个从 $C_{X/A}$ 到经典上同调的
比较陈述，至少要给出下列数据：

1. base prism $(A,I)$ 与 $X$ 的几何假设；
2. 施加在 $C_{X/A}$ 上的 derived base change 或 completion；
3. 是否先作 $\varphi_A^*$，以及 Frobenius 是 semilinear 还是已线性化；
4. 是否反演 $I$、模 $p^r$ 或取 derived inverse limit；
5. 目标复形所在的系数环与导出范畴。

缺少任一项时，“prismatic cohomology 等于某经典上同调”都不是类型完整的
等式。

**证明.** Derived tensor product 改变系数环，Frobenius pullback 改变
$A$-module structure，反演 $I$ 与模 $p^r$ 又改变对象所在的局部化或有限层
范畴。Derived fixed point 还是一个 fiber，而非原对象的子模。若不记录这些
操作，源与靶甚至不必属于同一范畴，因而不存在可供判断的同构。$\square$

## 15.2 形式推导与深比较定理的分界

从一个已经给定的比较同构出发，换基、取 fiber、传递 cup product 等形式后果
可以在导出范畴中证明；比较同构本身的存在却不是这些形式操作的结果。Perfectoid
rings 与 perfect prisms 的对应、prismatic comparison、BMS integral
comparison、Fontaine--Faltings--Tsuji comparison、$F$-crystals 与 crystalline
lattices 的范畴等价，以及 prismatization 的 crystals-as-QCoh 解释，分别需要
almost/pro-étale descent、torsion 控制或 stacky descent 等深层输入。

**判据 15.2.** 若一条论证只使用导出张量、fiber triangle 和 Frobenius 的形式
恒等式，它最多能传递一个已知比较定理，不能据此建立上述任一比较定理的存在性。

## 15.3 常见错误模式

**错误模式 15.3.** 把 $R\Gamma_\Delta(X/A)$ 直接等同于 etale cohomology。

**修正 15.4.** Etale comparison 需要 perfect prism、invert $I$、Frobenius fixed construction 和 modulo $p^n$ 或 inverse limit。

**错误模式 15.5.** 把 Hodge-Tate specialization 的 conjugate filtration 当成 de Rham Hodge filtration。

**修正 15.6.** 前者位于 $\overline\Delta_{X/A}$，后者位于
$\phi_A^\ast R\Gamma_\Delta(X/A)\widehat\otimes_A^L A/I\simeq
R\Gamma_{\mathrm{dR}}^{\wedge,L}_p$。这个 unfiltered comparison 本身不把
conjugate filtration 识别为 Hodge filtration。

**错误模式 15.7.** 把 Breuil-Kisin module、Breuil-Kisin-Fargues module、filtered $\varphi$-module 和 prismatic $F$-crystal 混用。

**修正 15.8.** 这些对象处于不同底环和范畴中，由 comparison 或 classification theorem 连接。

**错误模式 15.9.** 在 Nygaard/syntomic 公式中省略 twist convention。

**修正 15.10.** 必须说明 $\{i\}$ 是 $(I/I^2)^{\otimes i}$、其 dual，还是经 orientation 后的 $d^i$ 表示。

## 15.4 四条彼此独立的假设轴

比较定理的适用范围可以按四条轴记录：

- **底对象轴**：perfect、Breuil--Kisin、crystalline 或一般 bounded prism；
- **几何轴**：smooth、proper、quasisyntomic、lci、stacky 或 derived；
- **系数轴**：常系数、vector-bundle crystal、torsion coefficient 或非阿贝尔对象；
- **积分轴**：integral、模 $p^r$、反演 $pI$ 后或 rational。

**定义 15.11.** 把一条比较陈述的假设型记为
$$
\mathsf T=(\mathsf B,\mathsf G,\mathsf C,\mathsf L),
$$
其中四个分量依次记录上述底对象、几何、系数与积分层级。

**命题 15.12.** 若两条定理的假设型在某一轴上不同，且没有沿该轴的 base-change、
descent、devissage 或反演比较定理，则二者不能无条件合并。

**证明.** 四条轴分别改变 site、允许的覆盖、系数范畴或目标环。没有连接该轴的
函子与相容性定理时，两条结论的源或靶类型不同，合并陈述没有定义。$\square$

## 15.5 按结构障碍组织的开放问题

**研究边界 15.13.** 当前扩展方向可按缺失结构而非论文清单分成四组：

1. **系数与非阿贝尔化**：为 Hodge--Tate crystals、$q$-Higgs modules 与
   非阿贝尔对象构造共同的系数范畴，并证明 comparison 与 descent；
2. **几何化**：在 Artin/derived stacks 上比较 prismatization、QCoh 与原始
   prismatic site，同时控制 pushforward；
3. **torsion 与有效性**：刻画 $F$-gauge、display 和 $F$-crystal 之间何时保留
   integral lattice、height 与 Frobenius/Verschiebung；
4. **上同调运算**：构造 spectral syntomic operations，并证明它们与 cup product、
   Tate twist 和 classical realizations 相容。

Shimura varieties、Brauer groups 与 finite flat group schemes 分别为这些障碍提供
测试场景，但不能替代一般比较定理。

## 15.6 可复合比较定理的最低格式

**定义 15.14.** 一条比较定理称为可复合的，如果它明确给出：源与靶、假设型
$\mathsf T$、比较态射的构造或外部来源、对 base change/Frobenius/filtration 的
自然性，以及结论发生在 integral、finite-level 还是 rational 层。

**命题 15.15.** 两条可复合比较定理只有在中间对象、系数环、Frobenius
线性化和过滤约定逐项一致时，才能通过态射复合产生第三条比较定理。

**证明.** 态射复合首先要求前一靶等于后一源。对导出 $A$-modules，这包括
$A$-作用、completion 与 localization；对 filtered Frobenius objects，还包括
semilinear structure 和过滤指标。逐项一致时，导出范畴中的复合存在且自然性可传递；
任一项不一致时必须先提供相应换基或重指标同构。$\square$

## 15.7 统一语言不能抹去的差异

Prismatic language 的力量在于让多个 realization 共享一个源对象；它的风险也来自
同一处，因为共享源不等于目标相同。可靠的论证必须沿比较路径保留 base prism、
几何假设、系数和积分层级。未来的系数、stacky 与非阿贝尔理论若要真正纳入这套
框架，也必须给出这些轴上的函子和相容性，而不能只沿用已有对象的名称。

## 练习

**练习 15.1.** 从错误模式 15.3、15.5、15.7、15.9 中任选一个，写出一个错误证明并修正。

**练习 15.2.** 任取第十四章的一条应用结果，写出其假设型
$\mathsf T=(\mathsf B,\mathsf G,\mathsf C,\mathsf L)$，并指出与 smooth proper
scheme 情形相比改变了哪一轴。

**练习 15.3.** 选择研究边界 15.13 的一个方向，写出一条类型完整但尚未声称为真的
目标比较命题，并列出使其可复合所需的自然性条件。
