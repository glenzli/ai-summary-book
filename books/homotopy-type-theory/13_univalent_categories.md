# 第十三章：预范畴、单值范畴与结构等同

在普通范畴论中，同构对象常被口头上当作“没有区别”，但替换时仍要携带所选同构及其相干。HoTT 允许对象类型本身具有路径，于是可以追问规范映射
$$
\mathsf{idtoiso}_{x,y}:(x=y)\to(x\cong y)
$$
是否为等价。肯定回答这一问题的预范畴称为单值范畴；其中对象同构能够被转换成真正的对象路径，普通 transport 因而承担替换工作。

本章在集合值 Hom 的一范畴层研究这一现象。集合性保证范畴律和自然性条件是命题，单值性与结构等同性原则则给出集合范畴、群范畴等主要例子。预范畴、$\mathsf{idtoiso}$ 和这些例子的证明核见附录 P；displayed categories 与双范畴的更高接口留在附录 BE，不混入本章的一范畴定义。

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

**定义 13.8（weak equivalence 与等价结构）.** 函子 $F:\mathcal C\to\mathcal D$ 是 weak equivalence，若它 fully faithful 且 essentially surjective；后者使用命题截断：
$$
\prod_{d:\mathcal D}\left\|\sum_{c:\mathcal C}F(c)\cong d\right\|.
$$
一个范畴等价结构还实际给出拟逆 $G:\mathcal D\to\mathcal C$、自然同构
$1_{\mathcal C}\cong GF$、$FG\cong1_{\mathcal D}$ 以及三角相容。因而 weak equivalence 的本质满性只是性质，等价结构则包含选择后的拟逆数据。

**外部输入定理 13.9（单值范畴中的 weak equivalence）.** 设 $\mathcal C,\mathcal D$ 都是单值范畴，且函子 $F:\mathcal C\to\mathcal D$ fully faithful 并 essentially surjective。则 $F$ 是范畴等价；进一步，对单值范畴，范畴等价等价于“范畴同构”，后者指 $F$ fully faithful 且对象函数 $F_0:\mathcal C_0\to\mathcal D_0$ 是类型等价。

**来源与未重证边界。** Ahrens--Kapulkin--Shulman, *Univalent categories and the Rezk completion*, Mathematical Structures in Computer Science 25 (2015), Lemma 6.8 与 Lemma 6.15，DOI `10.1017/S0960129514000486`。Lemma 6.8 用源范畴的单值性证明
$$
\sum_{c:\mathcal C}Fc\cong d
$$
是命题，从而可合法消去 essentially-surjective 的命题截断；Lemma 6.15 比较范畴等价与对象函数等价。本书不重抄其中函子记录相等和 transport 的逐项证明，后文只按上述精确版本调用。

## 13.5 同构何时足以替换对象

预范畴只把 Hom 限制为集合，单值范畴再要求对象路径与对象同构等价。集合范畴和群范畴说明这不是抽象口号：类型单值性与 SIP 分别把双射和群同构转成对象路径。外部输入定理 13.9 还解释了命题截断为何不妨碍单值范畴中的本质满性产生拟逆。下一章会用 Yoneda 嵌入把任意预范畴送进一个单值范畴，并讨论这一过程的泛性质。

## 练习

**练习 13.1.** 写出 $\mathsf{idtoiso}$ 的路径归纳定义。

**练习 13.2.** 证明在任意预范畴中，同构关系是自反、对称、传递的。

**练习 13.3.** 说明为什么 Hom 类型要求是集合。

**练习 13.4.** 解释 essentially surjective 为什么使用命题截断。
