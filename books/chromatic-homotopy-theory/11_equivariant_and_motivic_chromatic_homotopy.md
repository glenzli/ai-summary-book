# 第十一章：Equivariant 和 motivic chromatic homotopy

非等变色层理论以一个素数和一维形式群高度组织谱；加入群作用或代数几何基底后，检测数据会同时依赖子群、几何点和 realization functor，原有 type 不再能原样搬运。本章比较 genuine equivariant 与 motivic 两个扩展方向：哪些 Bousfield、thick ideal 和形式群结构仍有平行版本，哪些结论需要额外的保守性或基底假设，哪些仍处在研究前沿。有限群作用、genuine spectra、motivic spectra 与形式群律的基本语言作为入口；深层分类与重构结果精确标为外部输入，使“类比”不被误写为定理。

## 11.1 Genuine equivariant spectra

**定义 11.1.** 设 $G$ 是有限群。$G$-equivariant stable homotopy theory 的对象是 genuine $G$-spectra，记其稳定 infinity-范畴为
$$
\mathbf{Sp}^G.
$$
它不同于带 $G$-作用的普通谱范畴 $\operatorname{Fun}(BG,\mathbf{Sp})$，因为 genuine theory 包含 representation spheres、fixed point functors 和 transfers。

**警告 11.2.** Naive $G$-spectra 和 genuine $G$-spectra 不可混用。许多 equivariant chromatic 现象依赖 genuine fixed points 和 representation grading。

**定义 11.3.** 对 genuine $G$-spectrum $X$ 和子群 $H\le G$，geometric fixed points 记作
$$
\Phi^H X.
$$
它是从 equivariant theory 到 ordinary spectra 的对称幺半函子之一。

**命题 11.4.** 若 $X\simeq0$ 于 $\mathbf{Sp}^G$，则对所有 $H\le G$ 有 $\Phi^H X\simeq0$。

**证明.** $\Phi^H$ 是函子，保持等价和零对象。若 $X$ 等价于零对象，则其像等价于零对象。证毕。

**警告 11.5.** 反向检测需要额外定理。对所有 geometric fixed points 为零是否推出 $X\simeq0$，依赖 genuine equivariant stable homotopy 的检测结果，不能只由定义推出。

## 11.2 Equivariant chromatic type

**定义 11.6.** 对有限群 $G$ 和 genuine $G$-spectrum $X$，一个粗略的 equivariant chromatic profile 是函数
$$
H\longmapsto \operatorname{type}(\Phi^H X),
$$
其中 $H$ 遍历 $G$ 的子群，右侧 type 在普通 $p$-局部谱中计算。

**警告 11.7.** 定义 11.6 只是入口。真正的 equivariant thick tensor ideals、Balmer spectrum、equivariant formal group laws 和 equivariant $v_n$ self-map 需要更精细结构。

**外部输入 11.8.** Balmer spectrum of finite genuine $G$-spectra、Hausmann-Meier equivariant formal group laws、Bhattacharya-Guillou-Li 和后续工作给出 equivariant chromatic theory 的结构基础。

**前沿输入 11.9.** Behrens-Carlisle 2024 对有限 abelian $p$-group $A$ 建立了与经典 chromatic picture 平行的 $A$-equivariant chromatic 框架，包括 equivariant analogs of $v_n$ self-maps、chromatic tower、smash product theorem 和 chromatic convergence theorem。

## 11.3 Motivic chromatic homotopy

**定义 11.10.** 对基域或基 scheme $S$，motivic spectra 的稳定 infinity-范畴记作
$$
\mathbf{SH}(S).
$$
其对象带有拓扑悬挂和 Tate twist 两类方向，因而同伦群常写成双分次。

**警告 11.11.** Motivic chromatic theory 不等于把普通 chromatic theory 的每个谱加一个权重分次。基域、real/complex realization、$\eta$-completion、motivic Steenrod algebra 和 algebraic cobordism 都会改变问题。

**定义 11.12.** Algebraic cobordism spectrum $MGL$ 是 motivic theory 中与 $MU$ 平行的对象。其定向和形式群律给出 motivic chromatic theory 的入口。

**外部输入 11.13.** Hopkins-Morel、Voevodsky、Hoyois、Spitzweck 等结果提供 $MGL$、orientation 和 slice filtration 的基础。Motivic Morava K-theories 和 motivic Morava E-theories 的构造与性质依赖基域和 completion 条件。

## 11.4 Synthetic 和重构结果

**定义 11.14.** Synthetic spectra 是介于拓扑谱和代数模型之间的范畴，用于把 Adams-Novikov filtration 或 even filtration categorify。Equivariant synthetic spectra 是其 genuine equivariant 版本。

**前沿输入 11.15.** Allen-Piessevaux 2025 证明，对有限 abelian group $A$，复数上的 cellular $A$-equivariant motivic spectra 在素数完备后可由 synthetic $A$-equivariant spectra 重构。其计算输入涉及 equivariant algebraic cobordism 和 equivariant formal group laws。

**使用限制 11.16.** 记录 11.15 是前沿 motivic/equivariant 接口。进入正文定理链前需核查 completion、cellular、基域 $\mathbb C$、有限 abelian group 和 prime 的全部假设。

## 11.5 Equivariant thick ideals 的基本形状

**定义 11.17.** 对有限群 $G$，finite genuine $G$-spectra 的 thick tensor ideal 是对 cofiber、retract、悬挂和与任意 finite genuine $G$-spectrum 张量封闭的 thick 子范畴。

**外部输入 11.18.** 这些 thick tensor ideals 可通过子群 $H\le G$ 的 geometric fixed points 和普通 chromatic type 数据描述，精确形式属于 equivariant Balmer spectrum 定理。

**警告 11.19.** 即使所有 $\Phi^H X$ 的普通 type 已知，组装成 genuine equivariant 信息仍需要 restriction、transfer 和 representation sphere 数据。profile 是必要信息的一部分，不是完整分类。

## 11.6 Motivic realization 与保守性

**定义 11.20.** 若基域嵌入 $\mathbb C$，complex realization 是函子
$$
\operatorname{Re}_{\mathbb C}:\mathbf{SH}(k)\to\mathbf{Sp}.
$$

**警告 11.21.** Realization functor 通常不是保守的。motivic 谱在 realization 后为零，不必在 motivic category 中为零。任何从拓扑 chromatic 结论返回 motivic 结论的步骤都需要保守性或 completion 假设。

**例 11.22.** Motivic Hopf map $\eta$ 和拓扑 Hopf map 的关系依赖 realization，但 motivic $\eta$-completion 和拓扑 completion 是不同过程。motivic chromatic 章节必须独立记录 completion。

## 11.7 可以迁移与不可直接迁移的结构

Equivariant 和 motivic chromatic theory 都保留“高度、周期性、形式群”的核心思想，但对象范畴和检测工具发生变化。Genuine equivariant theory 需要 geometric fixed points 和 representation grading；motivic theory 需要基域、Tate twist 和 $MGL$。当前章节建立接口和风险边界，后续扩写需分成独立教材级章节。

## 练习

**练习 11.1.** 给出 naive $G$-spectrum 和 genuine $G$-spectrum 的一个结构差异。

**练习 11.2.** 若 $G=C_2$，列出其子群，并写出 equivariant chromatic profile 需要检查的 geometric fixed points。

**练习 11.3.** 解释 motivic homotopy groups 为什么通常是双分次。

**练习 11.4.** 查阅一个 motivic Morava K-theory 的定义，记录其基域和 completion 假设。
