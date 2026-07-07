# 第十章 黑洞、视界与因果结构

## 10.1 坐标奇异和物理奇异

Schwarzschild 度规在 $r=2GM$ 处有

$$
g_{tt}=0,\qquad g_{rr}\to\infty.
$$

但曲率标量

$$
K=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}
=\frac{48G^2M^2}{r^6}
$$

在 $r=2GM$ 有限，在 $r=0$ 发散。因此 $r=2GM$ 是坐标奇异，$r=0$ 是真正曲率奇异。

## 10.2 Eddington-Finkelstein 坐标

定义 tortoise 坐标

$$
r_*=r+2GM\ln\left|\frac{r}{2GM}-1\right|.
$$

再定义先进时间

$$
v=t+r_*.
$$

Schwarzschild 度规变为

$$
ds^2
=-\left(1-\frac{2GM}{r}\right)dv^2
+2dv\,dr+r^2d\Omega^2.
$$

该形式在 $r=2GM$ 正则。进入的类光曲线可以穿过视界。

## 10.3 事件视界

非严格但物理清楚的定义是：黑洞区域是不能向未来无穷远发送信号的事件集合；其边界称为事件视界。

在 Schwarzschild 时空中，事件视界位于

$$
r=2GM.
$$

在视界内，未来方向不可避免地指向更小的 $r$。这不是“逃逸速度超过光速”的 Newton 说法，而是因果锥结构本身倾斜。

## 10.4 Kruskal 延拓

引入 null 坐标

$$
u=t-r_*,
\qquad
v=t+r_*.
$$

再作指数变换可得 Kruskal 坐标 $(U,V)$。在这些坐标中，Schwarzschild 最大解析延拓覆盖外部区域、黑洞区域、白洞区域和另一个外部区域。

本书不完整构造 Penrose 图，但要强调：坐标图不是时空本身。不同坐标可能覆盖不同区域，视界的正则性必须用坐标无关或可延拓的方式判断。

更具体地，对外部区域可取

$$
U=-e^{-u/(4GM)},\qquad V=e^{v/(4GM)}.
$$

于是

$$
UV=-\left(\frac{r}{2GM}-1\right)e^{r/(2GM)}.
$$

在 $(U,V)$ 中，径向部分成为

$$
ds^2_{\mathrm{radial}}
=-\frac{32G^3M^3}{r}e^{-r/(2GM)}\,dU\,dV.
$$

这里 $r$ 被隐式看作 $UV$ 的函数。关键点是：前面的系数在 $r=2GM$ 有限且非零，所以视界处的奇异性已被消去。真正的曲率奇异仍在 $r=0$，对应 $UV=1$。

## 10.5 径向类光曲线与视界方向

在先进 Eddington-Finkelstein 坐标中令 $d\Omega=0$ 且 $ds^2=0$，得到

$$
0=-\left(1-\frac{2GM}{r}\right)dv^2+2dv\,dr.
$$

径向类光曲线分为两族：

$$
dv=0,
\qquad
\frac{dr}{dv}
=\frac12\left(1-\frac{2GM}{r}\right).
$$

第一族是进入的类光线。第二族在 $r>2GM$ 时有 $dr/dv>0$，可向外传播；在 $r=2GM$ 时 $dr/dv=0$，贴着视界；在 $r<2GM$ 时即使是“向外”的类光线也满足 $dr/dv<0$，未来方向仍走向更小的 $r$。这说明视界不是物体表面，而是因果结构的边界。

## 10.6 困陷面

在球对称情形中，一个二维球面若两族未来指向的正交类光测地线膨胀率都为负，则称为困陷面。直观地说，从该球面向内和向外发出的光束，其横截面积都在未来方向减小。

Schwarzschild 黑洞内部的 $r=\mathrm{const}<2GM$ 球面具有这种性质。困陷面的出现是奇点定理中的核心几何假设之一：在合适能量条件和全局因果条件下，它会迫使未来类光测地线不完备。

## 10.7 表面引力和温度

Schwarzschild 黑洞的表面引力为

$$
\kappa=\frac{1}{4GM}.
$$

量子场论在曲时空中的外部输入给出 Hawking 温度

$$
T_H=\frac{\hbar\kappa}{2\pi k_B}
=\frac{\hbar}{8\pi GM k_B}.
$$

这是外部输入结论，本书不证明 Hawking 辐射。

## 10.8 Kerr 黑洞概览

旋转黑洞由 Kerr 度规描述，参数为质量 $M$ 和角动量 $J=Ma$。它具有：

- 外视界和内视界。
- 能层 ergosphere。
- 拖曳惯性系的 Lense-Thirring 效应。
- 更复杂的测地线和可分离结构。

**外部输入 D (Kerr 唯一性).** 在适当正则性、真空、渐近平直和定常轴对称假设下，四维无电荷定常黑洞由 Kerr 解描述。

该定理属于高级黑洞理论，不在本书内部证明。

## 10.9 黑洞章节的计算顺序

面对一个候选黑洞度规，基本检查顺序是：

1. 先计算坐标分量的奇异点。
2. 再计算曲率不变量，例如 $R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}$，判断是否为曲率奇异。
3. 对可疑坐标奇异点寻找正则坐标，例如 Eddington-Finkelstein 或 Kruskal 坐标。
4. 用径向类光曲线或因果锥判断未来方向。
5. 若讨论整体结构，再分析无穷远、Cauchy 超曲面、视界和最大延拓。

这个顺序可以避免把坐标失效误认为物理发散。

## 习题

1. 计算 Kretschmann 标量在 $r=2GM$ 和 $r=0$ 的行为。
2. 推导 Eddington-Finkelstein 度规形式。
3. 解释事件视界为何是全局概念。
4. 恢复单位，写出 Schwarzschild 半径 $r_s$。
5. 说明 Kerr 黑洞与 Schwarzschild 黑洞的主要差别。
