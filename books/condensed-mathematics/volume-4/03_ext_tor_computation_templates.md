# 第三章：Ext 与 Tor 的计算模板

## 本章目标

本章整理第一卷附录 G 中的 Ext/Tor 计算为可操作模板。

## 3.1 Ext 模板

输入：

1. 对象 $M\in\mathbf{CondAb}$。
2. 投射分解
   $$
   \cdots\to P_1\to P_0\to M\to0.
   $$
3. 目标对象 $A$。

步骤：

1. 形成复形
   $$
   \operatorname{Hom}(P_0,A)\to
   \operatorname{Hom}(P_1,A)\to\cdots.
   $$
2. 取同调：
   $$
   \operatorname{Ext}^i(M,A)=H^i\operatorname{Hom}(P_\bullet,A).
   $$

## 3.2 ED 自由对象

若

$$
P=\mathbb Z[\underline E],
$$

其中 $E$ 极不连通，则 $P$ 投射，因此

$$
\operatorname{Ext}^i(P,A)=0,\qquad i>0.
$$

## 3.3 Tor 模板

输入：

1. 凝聚交换环 $R$。
2. $R$-模 $M,N$。
3. 平坦或投射分解 $P_\bullet\to M$。

步骤：

$$
\operatorname{Tor}_i^R(M,N)
=
H_i(P_\bullet\otimes_RN).
$$

## 3.4 长正合列模板

短正合列

$$
0\to A'\to A\to A''\to0
$$

给出 Ext 长正合列。计算时常用它做维数平移。

## 3.5 风险点

1. 取值在 ED 对象上容易，但不等于自动知道 Ext。
2. 投射分解必须在凝聚阿贝尔群或凝聚模范畴中。
3. 普通拓扑向量空间的 projective resolution 不能直接替代凝聚范畴中的分解。

## 练习

**练习 3.1.** 对投射对象 $P$ 证明 $\operatorname{Ext}^1(P,A)=0$。

**练习 3.2.** 写出两项投射分解计算 $\operatorname{Ext}^1$ 的公式。

**练习 3.3.** 说明平坦对象的高阶 Tor 为什么消失。
