# 附录 A：泛函分析背景

## 本章目标

本附录汇总正文所需的 Hilbert 空间泛函分析工具，尤其是稠密子空间、闭算子、自伴性和谱。

## 依赖前置知识

需要度量空间、线性代数和基本 Hilbert 空间理论。

## A.1 稠密性与闭算子

**定义 A.1.** 子空间 $D\subset\mathcal H$ 称为稠密，若其闭包等于 $\mathcal H$。算子 $A:D\to\mathcal H$ 称为闭算子，若从 $\psi_n\to\psi$ 且 $A\psi_n\to\eta$ 可推出 $\psi\in D$ 且 $A\psi=\eta$。

**命题 A.2.** 有界算子在整个 Hilbert 空间上是闭算子。

**证明.** 若 $\psi_n\to\psi$ 且 $A\psi_n\to\eta$，由有界性 $A\psi_n\to A\psi$，极限唯一给出 $\eta=A\psi$。$\square$

## A.2 二次型

**定义 A.3.** 半有界二次型是稠密子空间 $Q\subset\mathcal H$ 上的 sesquilinear 型 $q$，满足 $q(\psi,\psi)\ge c\|\psi\|^2$。

**外部输入定理 A.4（Friedrichs 扩张，QM-EXT-17）.** 半有界闭二次型唯一对应一个半有界自伴算子。

## A.3 谱

**定义 A.5.** 闭算子 $A$ 的 resolvent 集由所有 $z\in\mathbb C$ 组成，使 $A-z$ 双射且逆为有界算子。谱为其补集。

## 本章小结

无界量子 Hamiltonian 的严格处理离不开闭算子、二次型和 resolvent。正文只使用这些工具的必要结论。

## 练习

**练习 A.1.** 证明闭算子的图像在 $\mathcal H\oplus\mathcal H$ 中为闭子空间。

**练习 A.2.** 说明为什么有界自伴算子的谱包含在实轴上。
