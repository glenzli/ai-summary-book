# 第十三章：预范畴、单值范畴与结构等同

## 本章目标

本章在 HoTT 中定义预范畴（precategory）和单值范畴（univalent category），并说明对象相等与同构之间的关系。Displayed categories、displayed univalence 和 univalent bicategories 的高阶扩展见附录 BE。

## 依赖前置知识

本章依赖集合层、等价、单值性和结构等同性原则。默认 Hom 类型是集合，以避免高阶范畴复杂性。预范畴、同构、$\mathsf{idtoiso}$ 和集合范畴单值性的证明核见附录 P；结构附加和高阶范畴论接口的 displayed category 口径见附录 BE。

## 13.1 预范畴

**定义 13.1.** 一个预范畴 $\mathcal C$ 包含：

1.  对象类型 $\mathcal C_0:\mathcal U$；
2.  对每个 $x,y:\mathcal C_0$，Hom 类型 $\mathcal C(x,y):\mathcal U$；
3.  每个 Hom 类型是集合；
4.  恒等态射 $\mathsf{id}_x:\mathcal C(x,x)$；
5.  复合
    $$
    \mathcal C(y,z)\to\mathcal C(x,y)\to\mathcal C(x,z);
    $$
6.  左右单位律和结合律。

这些律由于 Hom 是集合，可作为命题性公理处理。

**定义 13.2.** 对象 $x,y:\mathcal C_0$ 的同构类型 $x\cong y$ 由态射 $f:\mathcal C(x,y)$、$g:\mathcal C(y,x)$ 和双向逆律组成。

## 13.2 从对象路径到同构

**定义 13.3.** 对任意预范畴，有规范映射
$$
\mathsf{idtoiso}_{x,y}:(x=y)\to(x\cong y)
$$
由路径归纳定义；反身路径送到恒等同构。

**定义 13.4.** 预范畴 $\mathcal C$ 是单值范畴，若对任意 $x,y$，$\mathsf{idtoiso}_{x,y}$ 是等价。

**解释 13.5.** 单值范畴中，对象相等与对象同构等价。这不是对所有预范畴自动成立的性质，而是范畴定义的一部分。

## 13.3 例子

**例 13.6（集合范畴）.** 以集合为对象、函数为态射得到预范畴 $\mathsf{Set}_{\mathcal U}$。在单值性下，它是单值范畴。

**证明（书内证明核）.** 见附录 P.8-P.10。对集合 $A,B$，集合范畴中的同构等价于类型等价 $A\simeq B$；单值性给出 $(A=B)\simeq(A\simeq B)$；$\mathsf{isSet}$ 的证明分量由推论 O.4 删除。$\square$

**例 13.7（群范畴）.** 群及群同态形成预范畴。在结构等同性原则下，它是单值范畴。

**证明（书内证明核）.** 见附录 P.11，并使用附录 J.11 的群对象 SIP。群对象路径等价于群同构；$\mathsf{idtoiso}$ 在反身路径上计算为恒等同构，因此群范畴满足单值范畴条件。$\square$

## 13.4 等价范畴与同构范畴

**定义 13.8.** 函子 $F:\mathcal C\to\mathcal D$ 是范畴等价，若它是 fully faithful 且 essentially surjective。这里 essentially surjective 通常使用命题截断：
$$
\prod_{d:\mathcal D}\left\|\sum_{c:\mathcal C}F(c)\cong d\right\|.
$$

**命题 13.9（单值范畴中的等价改进，条件形式）.** 假设附录 AA.11 的 weak-equivalence 限制函子定理已经在选定目标库中完成定义翻译。则在单值范畴之间，范畴等价可提升为合适意义下的范畴同构。

**验证状态。** 函子范畴单值性见附录 X；Rezk completion 泛性质的书内归约见附录 AA。具体地，weak equivalence $F:\mathcal C\to\mathcal D$ 对任意单值目标 $\mathcal E$ 诱导限制函子等价，取 $\mathcal E=\mathcal C,\mathcal D$ 可构造拟逆函子和双向自然同构。AA.8-AA.10 保留 transport 与代表元相容性的逐项证明义务，见 K.1.4 的文本收口说明。

## 本章小结

预范畴只要求 Hom 是集合；单值范畴进一步要求对象路径等价于对象同构。这样，范畴论中的“同构对象可替换”成为类型论中的 transport 原则。

Displayed categories 和 univalent bicategories 把这一原则扩展到“结构附加”和“2-维态射”场景；本章只给出一范畴核心。

## 练习

**练习 13.1.** 写出 $\mathsf{idtoiso}$ 的路径归纳定义。

**练习 13.2.** 证明在任意预范畴中，同构关系是自反、对称、传递的。

**练习 13.3.** 说明为什么 Hom 类型要求是集合。

**练习 13.4.** 解释 essentially surjective 为什么使用命题截断。
