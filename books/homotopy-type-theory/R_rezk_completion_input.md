# 附录 R：Rezk 完备化的构造输入

## 目标

本附录把第十四章的 Rezk 完备化从一句“外部输入”改写为精确构造蓝图。函子范畴与预层范畴单值性已在附录 U、X 展开；泛性质证明架构见附录 AA。本附录集中给出 Rezk 完备化构造及其与泛性质的接口：

1.  构造预层范畴 $\widehat{\mathcal C}_{\mathsf{PSh}}=[\mathcal C^{op},\mathsf{Set}]$；
2.  用 Yoneda 嵌入 $y:\mathcal C\to\widehat{\mathcal C}_{\mathsf{PSh}}$；
3.  取其本质像作为 Rezk 完备化；
4.  登记 fully faithful、essentially surjective 和泛性质所需输入。

## R.1 预层范畴输入

**输入 R.1（预层范畴）.** 对预范畴 $\mathcal C$，存在预范畴
$$
\mathsf{PSh}(\mathcal C)\coloneqq[\mathcal C^{op},\mathsf{Set}]
$$
其对象为附录 Q.1 的集合值预层，Hom 为自然变换，恒等和复合逐点定义。

**证明（书内证明核）。** 见附录 U.1-U.5。对象和自然变换来自附录 Q；Hom 集合性由命题 U.1 给出；恒等、复合和范畴律由 U.2-U.4 逐点证明。

**定理 R.2（预层范畴单值性）.** $\mathsf{PSh}(\mathcal C)$ 是单值范畴。

**证明（书内证明核）。** 附录 U 把 $\mathsf{PSh}(\mathcal C)$ 展开为集合值反变函子和自然变换的预范畴。附录 X.10 证明：若目标范畴 $\mathsf{Set}$ 单值，则函子范畴 $[\mathcal C^{\mathsf{op}},\mathsf{Set}]$ 单值。集合范畴单值性由 P.10 给出。由 X.11，附录 U 的手写预层范畴与该函子范畴定义等价，因此 $\mathsf{PSh}(\mathcal C)$ 单值。$\square$

## R.2 Yoneda 嵌入

**定义 R.3（Yoneda 嵌入）.** 定义函子
$$
y:\mathcal C\to\mathsf{PSh}(\mathcal C)
$$
在对象上为
$$
c\mapsto y(c)=\mathcal C(-,c),
$$
在态射 $f:c\to d$ 上给出自然变换
$$
y(f):y(c)\Rightarrow y(d)
$$
其 $x$ 分量为
$$
\mathcal C(x,c)\to\mathcal C(x,d),
\qquad
g\mapsto f\circ g.
$$
函子律见附录 U.8-U.10。

**命题 R.4（Yoneda 嵌入 fully faithful）.** 对任意 $c,d:\mathcal C_0$，映射
$$
\mathcal C(c,d)\to\mathsf{Nat}(y(c),y(d))
$$
是等价。

**证明.** 这是定理 U.11。附录 Q.10 给出反向等价
$$
\mathsf{Nat}(y(c),y(d))\simeq\mathcal C(c,d),
$$
其逆正是定义 R.3 的 $f\mapsto y(f)$。$\square$

## R.3 本质像作为 Rezk 完备化

**定义 R.5（Rezk 完备化对象）.** 定义
$$
\widehat{\mathcal C}_0
\coloneqq
\sum_{P:\mathsf{PSh}(\mathcal C)}
\left\|
\sum_{c:\mathcal C_0}(y(c)=P)
\right\|.
$$
直观上，它由“仅仅可表”的预层组成。第二分量是命题截断，保证对象属于本质像是性质而非额外选择。

**定义 R.6（Rezk 完备化的 Hom）.** 对
$$
(P,r),(Q,s):\widehat{\mathcal C}_0
$$
定义
$$
\widehat{\mathcal C}((P,r),(Q,s))
\coloneqq
\mathsf{Nat}(P,Q).
$$
恒等、复合和 Hom 集合性从预层范畴继承。

**命题 R.7（$\widehat{\mathcal C}$ 是单值范畴）.** $\widehat{\mathcal C}$ 是单值范畴。

**证明.** 见附录 AH.3 和 AH.9。$\widehat{\mathcal C}$ 是单值范畴 $\mathsf{PSh}(\mathcal C)$ 的 full subcategory，其对象由命题性谓词
$$
\left\|\sum_{c:\mathcal C_0}(y(c)=P)\right\|
$$
切出。命题性 full subcategory 保持单值性；这里使用预层范畴单值性 R.2，以及路径形式本质像与同构形式本质像在单值范畴中的等价。$\square$

**定义 R.8（Rezk 嵌入）.** 定义函子
$$
\eta_{\mathcal C}:\mathcal C\to\widehat{\mathcal C}
$$
在对象上为
$$
c\mapsto
\bigl(y(c),\, |(c,\mathsf{refl}_{y(c)})|\bigr),
$$
在态射上为 Yoneda 嵌入的态射作用。

**命题 R.9（Rezk 嵌入 fully faithful）.** $\eta_{\mathcal C}$ fully faithful。

**证明.** Hom 没有被 full subcategory 改变，因此由命题 R.4 直接得到。$\square$

**命题 R.10（Rezk 嵌入 essentially surjective）.** $\eta_{\mathcal C}$ essentially surjective：
$$
\prod_{X:\widehat{\mathcal C}_0}
\left\|
\sum_{c:\mathcal C_0}\eta_{\mathcal C}(c)\cong X
\right\|.
$$

**证明.** 见附录 AH.7-AH.9。令 $X=(P,r)$。由 $r$ 已有
$$
\left\|\sum_{c:\mathcal C_0}(y(c)=P)\right\|.
$$
对该命题截断消去，取代表 $(c,p)$。路径 $p:y(c)=P$ 在预层范畴中由 $\mathsf{idtoiso}$ 给出同构；由于 $\widehat{\mathcal C}$ 是 full subcategory，同一自然同构也是 $\eta_{\mathcal C}(c)\cong X$。再送入命题截断。目标是命题截断，因此该消去合法。$\square$

## R.4 泛性质

**定理 R.11（Rezk 完备化泛性质，证明架构）.** 若 $\mathcal D$ 是单值范畴，则预合成
$$
(-)\circ\eta_{\mathcal C}:
\mathsf{Fun}(\widehat{\mathcal C},\mathcal D)
\to
\mathsf{Fun}(\mathcal C,\mathcal D)
$$
是合适意义下的等价。

**证明架构。** 见附录 AA.12。附录 X 已补上函子范畴、自然同构和单值函子范畴；附录 AA 证明 weak equivalence 的限制函子对单值目标给出等价的标准架构，并把剩余逐行 transport 计算定位在 AA.8-AA.10。

## R.5 当前状态

本附录已经精确给出 Rezk 完备化的对象、Hom、嵌入函子和基本性质的证明路线。附录 AA 进一步给出泛性质的证明架构。仍未逐行书内展开的是：

1.  本质像对象的代表元截断消去与扩张构造的逐步相容性；
2.  fully faithful + essentially surjective 嵌入对单值目标的扩张唯一性；
3.  AA.8-AA.10 中对象路径 transport、自然性和函子律的全部机器级细节。

因此，第十四章可引用 R.5-R.10 作为构造蓝图，引用 R.11 和附录 AA 作为泛性质证明架构。
