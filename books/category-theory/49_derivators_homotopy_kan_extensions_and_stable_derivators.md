# 第四十九章：Derivator、同伦 Kan 延拓与稳定 Derivator

单个同伦范畴会忘记同伦余极限图的形状；derivator 不恢复全部高阶映射空间，而是同时记录每个小图形 $I$ 上的同伦范畴 $\mathbb D(I)$ 以及限制和同伦 Kan 延拓。这个 2-函子数据足以严格表达逐点同伦极限、base change 与稳定性，并在传统三角范畴和完整 $\infty$-范畴之间形成中间语言。本章从 derivator 公理出发，比较它由模型范畴或 $\infty$-范畴产生的方式。

所需背景是 2-范畴、模型范畴、Kan 延拓和稳定同伦论。Derivator 只捕获 diagram homotopy categories；正文会明确哪些结论可由它检测，哪些高阶相干信息仍然遗失。

## 49.1 预 Derivator

**定义 49.1.** 设 $\mathbf{Cat}$ 为小范畴的 2-范畴。一个预 derivator 是严格 2-函子

$$
\mathbb D:\mathbf{Cat}^{op}\to\mathbf{CAT}.
$$

对小范畴 $I$，$\mathbb D(I)$ 称为 $I$-形图的同伦范畴。

**定义 49.2.** 对函子 $u:I\to J$，记

$$
u^*=\mathbb D(u):\mathbb D(J)\to\mathbb D(I)
$$

为限制函子。

**命题 49.3.** 预 derivator 把范畴复合反向送为限制函子复合。

**证明.** 因为 $\mathbb D$ 是 $\mathbf{Cat}^{op}$ 上的严格 2-函子，若 $I\xrightarrow uJ\xrightarrow vK$，则在 $\mathbf{Cat}^{op}$ 中复合方向相反。因此

$$
(vu)^*=\mathbb D(vu)=\mathbb D(u)\mathbb D(v)=u^*v^*.
$$

恒等函子同理送为恒等限制。$\square$

## 49.2 Derivator 公理

**定义 49.4.** Derivator 是满足如下核心性质的预 derivator：

1. $\mathbb D$ 把小 coproducts of diagram categories 送为 products of categories。
2. 对每个 $I$，点值函子族 $i^*:\mathbb D(I)\to\mathbb D(*)$ 联合保守。
3. 每个 $u^*$ 有左、右伴随
   $$
   u_! \dashv u^*\dashv u_*.
   $$
4. 左右 Kan 延拓满足点态公式的同伦版本。

**外部输入定理 49.5.** 每个 combinatorial model category 由

$$
\mathbb D_{\mathcal M}(I)=\operatorname{Ho}(\mathcal M^I)
$$

（取逐点弱等价的导出图范畴）产生 derivator。每个有全部小极限和小余极限的 $\infty$-category $C$ 也由定义 49.13 产生 derivator。任意 relative category 未必有这些同伦 Kan 延拓，因而不自动产生 derivator。

## 49.3 同伦 Kan 延拓

**定义 49.6.** Derivator 中，$u_!$ 称为同伦左 Kan 延拓，$u_*$ 称为同伦右 Kan 延拓。

**命题 49.7.** 若 $u:I\to *$ 为唯一函子，则 $u_!$ 和 $u_*$ 分别给出 $I$-形同伦余极限和同伦极限。

**证明.** 对终范畴 $*$，限制 $u^*:\mathbb D(*)\to\mathbb D(I)$ 把对象送为常值图。其左伴随按定义是沿 $u$ 的同伦左 Kan 延拓，即把 $I$-图压缩为最普遍的余锥；这正是同伦余极限。右伴随同理给最普遍的锥，即同伦极限。$\square$

## 49.4 点态公式与 comma categories

**外部输入定理 49.8.** Derivator 的点态公式说，对 $u:I\to J$ 与 $j\in J$，

$$
j^*u_!X\simeq \operatorname{hocolim}_{(u/j)} X|_{(u/j)}
$$

且右 Kan 延拓有对偶公式

$$
j^*u_*X\simeq \operatorname{holim}_{(j/u)} X|_{(j/u)}.
$$

**命题 49.9.** 若 $u:I\to J$ 是 equivalence of categories，则 $u^*$ 为等价。

**证明.** 设 $v:J\to I$ 为拟逆，并有自然同构 $vu\cong\operatorname{id}_I$、$uv\cong\operatorname{id}_J$。预 derivator 是 2-函子，故自然同构被送为自然同构。因此

$$
u^*v^*\cong(uv)^*\cong\operatorname{id},\qquad
v^*u^*\cong(vu)^*\cong\operatorname{id}.
$$

故 $u^*$ 为等价。$\square$

## 49.5 稳定 Derivator

**定义 49.10.** Derivator $\mathbb D$ 称为 pointed，若 $\mathbb D(*)$ 有零对象且由限制保持。它称为 stable，若 cocartesian squares 与 cartesian squares 在 $\mathbb D([1]\times[1])$ 中一致。

**外部输入定理 49.11.** 稳定 derivator 的基础范畴 $\mathbb D(*)$ 具有典范三角范畴结构。

**命题 49.12.** 在 stable derivator 中，pushout square 同时是 pullback square。

**证明.** 这是 stable 的定义展开。Cocartesian square 即同伦 pushout square，cartesian square 即同伦 pullback square。稳定性要求二者类相同，因此任意 pushout square 同时为 pullback square。$\square$

## 49.6 与 $\infty$-范畴的比较

**定义 49.13.** 给定 $\infty$-category $C$，其同伦 derivator 定义为

$$
\mathbb D_C(I)=h\operatorname{Fun}(N(I),C)
$$

其中 $h$ 表示同伦范畴。

**外部输入定理 49.14.** 若 $C$ 有全部小极限和小余极限，则 $\mathbb D_C$ 是 derivator。若 $C$ 还 stable，则 $\mathbb D_C$ 是 stable derivator。若只研究某个固定 diagram 2-category，可把“小”相应限制为其中允许的形状。

**命题 49.15.** $\mathbb D_C(*)\simeq hC$。

**证明.** 因为 $N(*)=\Delta^0$，函子 $\infty$-范畴 $\operatorname{Fun}(\Delta^0,C)$ 等价于 $C$。取同伦范畴得到

$$
\mathbb D_C(*)=h\operatorname{Fun}(\Delta^0,C)\simeq hC.
$$

$\square$

**命题 49.16.** Derivator 的 coproduct-product 公理推出

$$
\mathbb D(\varnothing)\simeq *
$$

且对小范畴 $I,J$ 有

$$
\mathbb D(I\amalg J)\simeq\mathbb D(I)\times\mathbb D(J).
$$

**证明.** Derivator 公理要求 $\mathbb D$ 把图形范畴的小 coproduct 送为范畴的 product。空 coproduct 是初始小范畴 $\varnothing$，其像是空 product，即终范畴。二元 coproduct $I\amalg J$ 的像即二元 product，得到第二个等价。$\square$

## 49.7 同时保留所有图形的同伦范畴

Derivator 把同伦论从单个同伦范畴扩展为所有图形的同伦范畴系统。限制函子、同伦 Kan 延拓、点态公式和稳定性公理让三角范畴中的许多非函子性构造重新变得可控。它与 $\infty$-范畴互补：$\infty$-范畴保留全部高阶映射空间，derivator 保留足够多的图形同伦范畴以支撑同伦代数计算。

## 练习

**练习 49.1.** 定义预 derivator。

**练习 49.2.** 证明预 derivator 中 $(vu)^*=u^*v^*$。

**练习 49.3.** 列出 derivator 的核心公理。

**练习 49.4.** 定义同伦左、右 Kan 延拓。

**练习 49.5.** 说明唯一函子 $I\to *$ 的 Kan 延拓给出同伦极限和余极限。

**练习 49.6.** 写出左 Kan 延拓的点态公式。

**练习 49.7.** 证明 equivalence of categories 诱导 derivator 值的等价。

**练习 49.8.** 定义 stable derivator。

**练习 49.9.** 说明 stable derivator 中 pushout 与 pullback 的关系。

**练习 49.10.** 陈述 stable derivator 给出三角范畴结构。

**练习 49.11.** 从 $\infty$-category $C$ 定义 $\mathbb D_C$。

**练习 49.12.** 证明 $\mathbb D_C(*)\simeq hC$。

**练习 49.13.** 证明 derivator 的 coproduct-product 公理推出 $\mathbb D(\varnothing)\simeq *$ 与 $\mathbb D(I\amalg J)\simeq\mathbb D(I)\times\mathbb D(J)$。
