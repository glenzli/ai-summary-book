# 附录 X：函子范畴、自然同构与单值性

本附录补全第十三、十四章反复引用的一般函子范畴证明核。全篇沿用附录 P 的预范畴口径：Hom 类型为集合，范畴律的证明分量在路径比较中由命题性消去。

设 $\mathcal C,\mathcal D$ 为预范畴。

## X.1 函子

**定义 X.1（函子）.** 函子 $F:\mathcal C\to\mathcal D$ 由以下数据组成：

1.  对象函数
    $$
    F_0:\mathcal C_0\to\mathcal D_0;
    $$
2.  态射函数
    $$
    F_1:\mathcal C(x,y)\to\mathcal D(F_0x,F_0y);
    $$
3.  恒等保持
    $$
    F_1(\mathsf{id}_x)=\mathsf{id}_{F_0x};
    $$
4.  复合保持
    $$
    F_1(g\circ f)=F_1(g)\circ F_1(f).
    $$

由于 $\mathcal D$ 的 Hom 是集合，第三、四项的路径目标是命题。因此函子相等时只需比较对象函数和态射函数；函子律证明分量由 Hom 集合性消去。

**定义 X.2（函子复合与恒等函子）.** 恒等函子 $\mathsf{Id}_{\mathcal C}$ 在对象和态射上均为恒等函数。若
$$
F:\mathcal C\to\mathcal D,\qquad
G:\mathcal D\to\mathcal E,
$$
则复合 $G\circ F:\mathcal C\to\mathcal E$ 在对象和态射上分别为普通复合：
$$
(G\circ F)_0(x)\coloneqq G_0(F_0x),
$$
$$
(G\circ F)_1(f)\coloneqq G_1(F_1(f)).
$$
函子律由 $F$ 与 $G$ 的函子律和 $\mathcal E$ 的路径代数给出。

## X.2 自然变换

**定义 X.3（自然变换）.** 对函子 $F,G:\mathcal C\to\mathcal D$，自然变换
$$
\alpha:F\Rightarrow G
$$
由分量族
$$
\alpha_x:\mathcal D(Fx,Gx)
$$
和自然性路径
$$
G(f)\circ \alpha_x=\alpha_y\circ F(f)
$$
组成，其中 $f:\mathcal C(x,y)$。

**命题 X.4（自然变换相等）.** 若 $\alpha,\beta:F\Rightarrow G$，则路径
$$
\alpha=\beta
$$
等价于逐点路径族
$$
\prod_{x:\mathcal C_0}\alpha_x=\beta_x.
$$

**证明.** 自然变换类型是
$$
\sum_{\alpha_0:\prod_x\mathcal D(Fx,Gx)}
\prod_{x,y}\prod_{f:\mathcal C(x,y)}
  (G(f)\circ\alpha_0(x)=\alpha_0(y)\circ F(f)).
$$
第二分量是路径族，其每个目标位于 Hom 集合
$\mathcal D(Fx,Gy)$ 的路径空间中，因此是命题。由 $\Sigma$ 路径刻画和子类型路径原则，自然变换相等只需比较分量族。再由函数外延性，分量族相等等价于逐点相等。$\square$

**推论 X.5（自然变换类型是集合）.** 对任意 $F,G$，
$$
\mathsf{isSet}(F\Rightarrow G).
$$

**证明.** 由 X.4，自然变换的路径空间等价于逐点路径族。每个分量路径位于 Hom 集合 $\mathcal D(Fx,Gx)$ 中，故是命题；函数外延性保持命题性。于是自然变换路径空间是命题，自然变换类型为集合。$\square$

## X.3 函子范畴

**定义 X.6（函子范畴）.** 定义预范畴
$$
[\mathcal C,\mathcal D]
$$
如下：

1.  对象为函子 $F:\mathcal C\to\mathcal D$；
2.  Hom 为自然变换：
    $$
    \mathsf{Hom}_{\left[\mathcal C,\mathcal D\right]}(F,G)\coloneqq(F\Rightarrow G);
    $$
3.  Hom 集合性由 X.5 给出；
4.  恒等自然变换 $\mathsf{id}_F$ 的分量为
    $$
    (\mathsf{id}_F)_x\coloneqq\mathsf{id}_{Fx};
    $$
5.  垂直复合 $\beta\cdot\alpha:F\Rightarrow H$ 的分量为
    $$
    (\beta\cdot\alpha)_x\coloneqq\beta_x\circ\alpha_x.
    $$

自然性证明由 $\alpha,\beta$ 的自然性和 $\mathcal D$ 的结合律给出：
$$
H(f)\circ(\beta_x\circ\alpha_x)
=(H(f)\circ\beta_x)\circ\alpha_x
=(\beta_y\circ G(f))\circ\alpha_x
=\beta_y\circ(G(f)\circ\alpha_x)
=\beta_y\circ(\alpha_y\circ F(f))
=(\beta_y\circ\alpha_y)\circ F(f).
$$
单位律和结合律逐分量化为 $\mathcal D$ 中的单位律和结合律；由 X.4，它们给出自然变换路径。因此 $[\mathcal C,\mathcal D]$ 是预范畴。

## X.4 自然同构

**定义 X.7（自然同构）.** 自然变换 $\alpha:F\Rightarrow G$ 是自然同构，若每个分量
$$
\alpha_x:\mathcal D(Fx,Gx)
$$
都是 $\mathcal D$ 中的同构。记
$$
F\cong_{\mathsf{nat}}G
\coloneqq
\sum_{\alpha:F\Rightarrow G}\prod_{x:\mathcal C_0}\mathsf{isIso}(\alpha_x).
$$

**命题 X.8（函子范畴同构等价于自然同构）.** 对函子
$F,G:\mathcal C\to\mathcal D$，有等价
$$
(F\cong_{[\mathcal C,\mathcal D]}G)\simeq(F\cong_{\mathsf{nat}}G).
$$

**证明.** 从函子范畴同构出发，设底层自然变换为
$\alpha:F\Rightarrow G$，其逆自然变换为 $\beta:G\Rightarrow F$，并有
$$
\beta\cdot\alpha=\mathsf{id}_F,\qquad
\alpha\cdot\beta=\mathsf{id}_G.
$$
由 X.4 对这些路径取分量，得到对每个 $x$：
$$
\beta_x\circ\alpha_x=\mathsf{id}_{Fx},
\qquad
\alpha_x\circ\beta_x=\mathsf{id}_{Gx}.
$$
故 $\alpha_x$ 为同构，得到自然同构。

反向地，给定自然变换 $\alpha$ 且每个 $\alpha_x$ 是同构，令
$$
\beta_x\coloneqq(\alpha_x)^{-1}.
$$
需要证明 $\beta$ 自然。对 $f:x\to y$，要证
$$
F(f)\circ\beta_x=\beta_y\circ G(f).
$$
从 $\alpha$ 的自然性
$$
G(f)\circ\alpha_x=\alpha_y\circ F(f)
$$
左复合 $\beta_y$、右复合 $\beta_x$，并使用 $\beta_y\circ\alpha_y=\mathsf{id}$ 与
$\alpha_x\circ\beta_x=\mathsf{id}$，得到所需等式。于是 $\beta$ 是自然变换，并且 $\beta\cdot\alpha$ 与 $\alpha\cdot\beta$ 逐分量为恒等；由 X.4 得到函子范畴中的逆律。

两向互逆只需检查底层自然变换，因为 isIso 是命题（P.3），自然性证明也由 Hom 集合性消去。$\square$

## X.5 函子范畴的单值性

**定理 X.9（目标单值推出函子范畴单值）.** 若 $\mathcal D$ 是单值范畴，则
$$
[\mathcal C,\mathcal D]
$$
是单值范畴。

**证明.** 固定函子 $F,G:\mathcal C\to\mathcal D$。需证明
$$
\mathsf{idtoiso}_{F,G}:(F=G)\to(F\cong_{[\mathcal C,\mathcal D]}G)
$$
是等价。由 X.8，只需把函子路径与自然同构等价起来。

从函子路径 $p:F=G$ 到自然同构：对 $p$ 作路径归纳。反身情形给出恒等自然变换，其每个分量为恒等同构。这与 $\mathsf{idtoiso}$ 的定义一致。

反向地，给定自然同构 $(\alpha,i)$。由于 $\mathcal D$ 单值，每个分量同构
$$
\alpha_x:Fx\cong Gx
$$
给出对象路径
$$
p_x:Fx=Gx.
$$
由函数外延性得到对象函数路径
$$
p_0:F_0=G_0.
$$
沿 $p_0$ transport $F$ 的态射函数后，需证明它等于 $G$ 的态射函数。对 $f:\mathcal C(x,y)$，这个目标在 Hom 集合
$$
\mathcal D(Gx,Gy)
$$
中。展开 transport 后，它正是自然性方程
$$
G(f)\circ\alpha_x=\alpha_y\circ F(f)
$$
在单值性给出的对象路径 $p_x,p_y$ 下的重写形式。由于目标 Hom 是集合，任何由同一自然性方程得到的路径相容性证明相等；再用函数外延性得到态射函数路径 $p_1$。最后函子律证明分量位于 Hom 集合的路径类型中，因此由命题性消去。得到函子路径
$$
F=G.
$$

现在检查两向互逆。自然同构到路径再到自然同构时，逐分量化为 $\mathcal D$ 的单值性等价的三角恒等；由 X.4，自然变换路径由逐分量路径决定。路径到自然同构再到路径时，对函子路径作路径归纳，反身情形为 reflexivity；函子律证明分量仍由命题性消去。故 $\mathsf{idtoiso}_{F,G}$ 是等价。$\square$

## X.6 预层范畴的单值性

**推论 X.10（集合值预层范畴单值）.** 若 $\mathsf{Set}_{\mathcal U}$ 是单值范畴，则
$$
[\mathcal C^{\mathsf{op}},\mathsf{Set}_{\mathcal U}]
$$
是单值范畴。

**证明.** 由定理 X.9，取 $\mathcal D\coloneqq\mathsf{Set}_{\mathcal U}$。集合范畴单值性由 P.10 给出。$\square$

**推论 X.11（附录 U 的预层范畴）.** 附录 U 中显式构造的
$$
\mathsf{PSh}(\mathcal C)
$$
与函子范畴
$$
[\mathcal C^{\mathsf{op}},\mathsf{Set}_{\mathcal U}]
$$
定义等价；其自然变换、恒等与复合分别对应 X.3 与 X.6 的 Hom、恒等和垂直复合。因此 U.1-U.6 是 X.6 在集合值预层上的展开实例，X.10 给出其单值性。

**证明.** 对象方面，附录 U 的预层即反变集合值函子；态射方面，U.1 的自然变换正是 X.3 的自然变换，其中 $\mathcal C^{\mathsf{op}}$ 把反变自然性改写为协变自然性。恒等与复合的分量定义与 X.6 相同。由 Hom 集合性，范畴律证明分量相等自动成立。$\square$

## X.7 对 Rezk 完备化泛性质的作用

附录 R 把 Rezk 完备化构造为 Yoneda 嵌入在预层范畴中的本质像。本附录补上其中两个一般范畴论输入：

1.  函子范畴 $[\mathcal C,\mathcal D]$ 的预范畴结构；
2.  当目标 $\mathcal D$ 单值时，函子范畴单值。

对任意单值范畴 $\mathcal E$，沿
$$
\eta:\mathcal C\to\widehat{\mathcal C}
$$
预合成给出
$$
[\widehat{\mathcal C},\mathcal E]\to[\mathcal C,\mathcal E]
$$
得到预范畴同构，是 Ahrens--Kapulkin--Shulman, *Univalent categories and the Rezk completion*, Theorem 8.4 的外部输入。附录 R 在书内验证 $\eta$ 是 weak equivalence，附录 AA 则精确记录该外部定理的假设、结论和消去边界。尤其是命题截断代表元的处理、fully faithful 扩张以及对象路径引起的 Hom transport 相容性均由来源定理承担；本附录的函子范畴单值性不能替代那些计算。
