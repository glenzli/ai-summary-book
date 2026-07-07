# 第二十章：string theory 与量子场论、几何和数论的接口

## 本章目标

本章总结 string theory 与 gauge theory、enumerative geometry、moduli spaces、dualities 和 arithmetic geometry 的接口，并明确本书不把这些方向无限展开。它是收束章，不是外部学科百科。

## 依赖前置知识

需要第十二章 D-branes、第十六章 topological strings、第十八章 AdS/CFT 和第十三章紧化。

## 20.1 与量子场论的接口

**例 20.1（D-branes and gauge theory）.** D-branes 上的 open-string low-energy theory 给出 gauge theory。Brane configurations 可工程化 supersymmetric gauge theories，其 Coulomb/Higgs branches 对应 brane positions、Wilson lines 或几何 moduli。

**例 20.2（AdS/CFT）.** AdS/CFT 将某些 large-$N$ gauge theories 与 quantum gravity/string backgrounds 联系起来。强耦合 gauge theory observable 可在 classical gravity 极限中近似计算。

**原则 20.3（QFT 接口边界）.** 本书只使用 string theory 主线自然产生的 QFT 结构：worldvolume gauge theory、duality-derived field theory statements、holographic dictionary 和 anomaly matching。一般 QFT 技术不在本书中无限展开。

## 20.2 与几何的接口

**例 20.4（enumerative geometry）.** A-model topological string 与 Gromov-Witten theory、Donaldson-Thomas theory 和 curve counting 相关。

**例 20.5（complex geometry）.** B-model 与 variation of Hodge structure、period integrals、derived categories 和 mirror symmetry 相关。

**原则 20.6（几何接口边界）.** 本书展开几何只到 string compactification、topological string 或 duality dictionary 必需的程度。Yau theorem、derived algebraic geometry、virtual fundamental class 等大型理论作为外部输入处理。

## 20.3 与数论的接口

**例 20.7（modular forms）.** Torus partition functions、elliptic genera、moonshine phenomena 和某些 BPS counting generating functions 都自然产生 modular 或 automorphic forms。

**例 20.8（periods and arithmetic）.** Calabi-Yau periods、Picard-Fuchs equations 和 mirror maps 与 arithmetic geometry 有深层联系。Flux vacua 的计数问题也可引出 lattice points、heights 和 arithmetic statistics。

**原则 20.9（数论接口边界）.** 数论接口不替代本书主线。正文只在 modular invariance、elliptic genus、periods 和 BPS generating functions 直接出现时引入相关对象。

## 20.3A 接口矩阵

| 接口方向 | 本书中出现的位置 | 允许展开到哪里 | 不在本书展开的部分 |
|---|---|---|---|
| Gauge theory | D-branes, AdS/CFT | worldvolume SYM, anomaly, large-$N$ dictionary | 一般非微扰 QFT 分类 |
| Enumerative geometry | A-model, mirror symmetry | GW invariants, mirror map, periods | virtual class 完整构造 |
| Complex geometry | CY compactification | Hodge numbers, moduli, DUY 接口 | derived algebraic geometry 细节 |
| Number theory | modular invariance, BPS counts | modular forms as partition functions | arithmetic geometry 系统理论 |
| Quantum gravity constraints | flux, swampland | conjecture 状态和控制条件 | swampland program 全貌 |

## 20.4 全书主线回顾

**定理状态回顾 20.10.** 本书主线可分为四类陈述：

1. 世界面 QFT 内可证明的命题：OPE、Virasoro、BRST、T-duality 的 CFT 形式、tree amplitudes。
2. 外部输入定理：Yau theorem、no-ghost theorem、modular invariance 的几何基础、anomaly polynomial factorization、Donaldson-Uhlenbeck-Yau。
3. 低能有效理论推导：beta functions、supergravity、DBI/WZ、dimensional reduction。
4. 物理猜想或非微扰对偶：S-duality、U-duality、M-theory、AdS/CFT、general mirror symmetry。

**原则 20.11（教材收口原则）.** 一个外部方向只有在满足以下条件之一时才进入正文：

1. 它是 string consistency 的必要条件；
2. 它直接计算 string spectrum、amplitude、effective action 或 protected quantity；
3. 它是后续 duality dictionary 的必要组成；
4. 它解释已在正文出现的数学对象。

否则只在资料源或后置专题中列出，不扩展为本书正文。

**命题 20.13（主线闭包）.** 按原则 20.11，本书的正文主线在第 20 章闭合：后续新增材料应归入例子、习题、附录公式或专题阅读，而不是新增主线章。

**证明.** 第 1 至 19 章已经覆盖 worldsheet definition、quantization、consistency、spectrum、amplitudes、superstrings、D-branes、compactification、duality、holography 和 flux。第 20 章给出外部接口边界。任何新增方向若不服务于这些结构，就不满足原则 20.11。$\square$

## 20.5 后续阅读路径

**定义 20.12（专题路径）.** 完成本书主线后，可按目的选择专题路径：

1. Scattering amplitudes：深入 pure spinor、amplituhedron、modern bootstrap。
2. Compactification：深入 algebraic geometry、vector bundles、moduli stabilization。
3. Holography：深入 large-$N$ QFT、black holes、quantum information。
4. Topological strings：深入 GW/DT theory、derived categories、homological mirror symmetry。
5. Arithmetic：深入 modularity、moonshine、automorphic BPS counting。

## 本章小结

String theory 的力量来自多方向接口，但教材主线必须收束在 worldsheet、spectrum、consistency、duality、geometry 和 holography。外部理论按必要性引入，而不是无界扩张。

## 练习

**练习 20.1.** 举例说明一个 string duality 如何产生非平凡的量子场论陈述。

**练习 20.2.** 从本书四类陈述中各举一例，并说明其证明状态。

**练习 20.3.** 选择一个外部接口方向，说明它为什么应作为专题阅读而不是新增主线章。

