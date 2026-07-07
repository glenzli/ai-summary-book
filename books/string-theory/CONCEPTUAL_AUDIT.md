# 最终概念审定

本文档列出本书最容易混淆的概念边界。后续扩写必须以此作为局部审定表。

## 1. Worldsheet 与 target spacetime

Worldsheet 是二维量子场论的定义域；target spacetime 是场 $X:\Sigma\to M$ 的取值空间。世界面 conformal invariance 不是 target spacetime conformal invariance。只有在特定背景和边界条件下，worldsheet statement 才能翻译为 target-space spectrum、equations of motion 或 scattering amplitude。

## 2. Gauge fixing 与物理自由度

Conformal gauge 不是物理假设，而是 gauge choice。Virasoro constraints、ghosts 和 BRST cohomology 负责移除 gauge redundancy。正文中所有“物理态”必须指 old covariant constraints 或 BRST cohomology，不得只指 Fock space 中的任意 oscillator state。

## 3. Critical dimension 的含义

玻色弦的 $D=26$ 和 RNS 超弦的 $D=10$ 是平坦背景中 worldsheet conformal anomaly 消失的条件。它们不是一般 curved background 中的独立假设；在 curved background 中，一致性由 sigma-model beta functions 和 Weyl invariance 条件表达。

## 4. 质量公式与 normal ordering

质量公式依赖 $\alpha'$ 归一化、开闭弦零模规范和 normal ordering constant $a$。闭弦必须分别满足左右 Virasoro 条件，因而有质量公式和 level matching 两个条件。任何谱计算都必须同时声明二者。

## 5. BRST closed、exact 与 physical equivalence

$Q_B|\psi\rangle=0$ 只表示 BRST closed；物理等价类是 cohomology
$$
\ker Q_B/\operatorname{im}Q_B.
$$
Gauge equivalent polarizations 对应 BRST exact states。顶点算子中的未积分与积分形式差异由 ghost number 和 conformal Killing group gauge fixing 控制。

## 6. Perturbative equality 与 nonperturbative duality

T-duality 在 perturbative worldsheet CFT 中有精确定义。S-duality、U-duality、M-theory 极限和 AdS/CFT 在当前教材中作为物理对偶或受检验框架陈述，除非某一受限命题已有独立数学证明。正文必须避免把这些陈述写成无条件定理。

## 7. D-branes 的双重身份

D-branes 首先是开弦端点的 Dirichlet 边界条件；在含 RR fields 的超弦中，它们也是带 RR charge 的动力学对象。二者的等价性需要 disk 振幅、RR coupling 和低能 supergravity 解相互校准。

## 8. 低能有效作用的边界

Supergravity、DBI 和 Wess-Zumino 作用量只描述低能、长波长或特定导数展开下的有效理论。它们不是完整 string theory。正文中每次使用有效作用都必须声明忽略的 $\alpha'$ 修正、string loop 修正或 massive string modes。

## 9. Compactification 与 vacuum selection

紧化给出从高维理论到低维有效理论的构造；它本身不自动选择真实真空。Moduli stabilization、flux、brane/orientifold 和 nonperturbative effects 属于额外结构，必须在第十九章单独标明假设。

## 10. 数学定理、外部输入和物理猜想

正文采用四类状态：`P` 已证，`S` 证明草图，`E` 外部输入，`C` 物理猜想或对偶性原则。任何非平凡陈述不得无状态地使用。

