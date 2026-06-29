# 附录 AR：Cauchy 实数的环、序与完备有序域接口

附录 AK 已给出 Cauchy 实数 HIIT、完备性和加法证明核。本附录继续把乘法、序、倒数和完备有序域结构的证明义务展开到教材可审查层。它不把每个有理数误差估计逐项展开，但说明每个定理依赖的估计形态。

## AR.1 负元与距离

**定义 AR.1（负元）.** 在有理数生成元上定义
$$
-\mathsf{rat}(q)\coloneqq\mathsf{rat}(-q).
$$
在极限元上定义
$$
-\mathsf{lim}(a)\coloneqq \mathsf{lim}(\lambda\varepsilon.\,-a(\varepsilon)).
$$

**命题 AR.2（负元保持 Cauchy 近似）.** 若 $a$ 是 Cauchy 近似，则
$$
\lambda\varepsilon.\,-a(\varepsilon)
$$
是 Cauchy 近似。

**证明（证明核）.** 由距离对负元的不变性
$$
d(-x,-y)<\varepsilon \simeq d(x,y)<\varepsilon.
$$
该性质在有理数生成元上是有序域等式，在极限情形由 HIIT 归纳和极限唯一性传递。$\square$

**命题 AR.3（加法逆元律）.** 对任意 $x:\mathbb R_C$，
$$
x+(-x)=0.
$$

**证明状态.** 对 $x$ 作 HIIT 归纳。有理数情形归约为 $\mathbb Q$ 的 $q+(-q)=0$；极限情形把等式逐点应用于近似 $a(\varepsilon)$，再用极限唯一性。误差预算只需加法的 Lipschitz 性。$\square$

## AR.2 局部有界性

**定义 AR.4（有界实数）.** 实数 $x$ 称为有界，若存在自然数 $N$ 使
$$
|x|<N.
$$
在构造性口径下，这通常记为命题截断：
$$
\|\,\sum_{N:\mathbb N}|x|<N\,\|.
$$

**命题 AR.5（每个 Cauchy 实数局部有界）.** 对任意 $x:\mathbb R_C$，有
$$
\|\,\sum_{N:\mathbb N}|x|<N\,\|.
$$

**证明（证明核）.** 对 $x$ 作 HIIT 归纳。有理数情形由有理数的阿基米德性质。若 $x=\mathsf{lim}(a)$，取近似 $a(1)$，由归纳假设或有理数近似得到 $|a(1)|<N$；Cauchy 条件给 $d(a(\varepsilon),a(1))<1+\varepsilon$，故 $x$ 被 $N+2$ 控制。因为界的具体数值可被截断，选择只需命题截断消去。$\square$

## AR.3 乘法

**定义 AR.6（乘法）.** 在有理数生成元上：
$$
\mathsf{rat}(q)\cdot\mathsf{rat}(r)\coloneqq\mathsf{rat}(qr).
$$
若 $x=\mathsf{lim}(a)$，定义
$$
x\cdot y
\coloneqq
\mathsf{lim}\bigl(\lambda\varepsilon.\,a(\delta_\varepsilon)\cdot y\bigr),
$$
其中 $\delta_\varepsilon$ 由 $y$ 的局部界和误差预算选择。右变量极限情形类似。

**证明义务 AR.7（乘法 well-definedness）.** 定义 AR.6 需要：

1.  每个实数局部有界；
2.  若 $a$ 是 Cauchy 近似，则 $\lambda\varepsilon.\,a(\delta_\varepsilon)\cdot y$ 是 Cauchy 近似；
3.  乘法尊重近似相等；
4.  双变量极限递归分支相容；
5.  乘法尊重分离路径和集合截断。

**命题 AR.8（乘法 Cauchy 估计）.** 若 $|x|<M$、$|y|<N$、$d(x,x')<\alpha$、$d(y,y')<\beta$，则
$$
d(xy,x'y')<M\beta+N\alpha+\alpha\beta.
$$

**证明.** 展开
$$
xy-x'y'=(x-x')y+x'(y-y')
$$
或
$$
xy-x'y'=(x-x')(y-y')+(x-x')y'+x'(y-y').
$$
用三角不等式、乘法对绝对值的估计和界 $M,N$。这是有序域层计算，在 HIIT 中作为乘法构造的误差预算核心。$\square$

**定理 AR.9（交换环结构，证明核 / 外部输入）.** $\mathbb R_C$ 在 $0,1,+,-,\cdot$ 下构成交换环。

**证明状态.** 对各环律作 HIIT 归纳；有理数生成元归约到 $\mathbb Q$ 的环律，极限情形由 AR.8 的乘法 Cauchy 估计和极限唯一性关闭。完整证明需要统一误差预算并避免在命题截断外提取界。本书登记为证明核。

## AR.4 序关系

**定义 AR.10（正性）.** 定义 $x>0$ 为存在正有理数 $\varepsilon$，使得 $x$ 与某个有理近似相距小于 $\varepsilon/2$ 且该近似大于 $\varepsilon$。形式上可取：
$$
x>0
\coloneqq
\left\|
\sum_{\varepsilon:\mathbb Q_{>0}}
\sum_{q:\mathbb Q}
\bigl(d(x,\mathsf{rat}(q))<\varepsilon/2\bigr)\times(q>\varepsilon)
\right\|.
$$
定义
$$
x<y\coloneqq y-x>0,\qquad
x\le y\coloneqq \neg(y<x)
$$
或采用 located preorder 的替代表述。

**命题 AR.11（$<$ 是命题值关系）.** 对 $x,y:\mathbb R_C$，$x<y$ 是命题。

**证明.** 定义中外层是命题截断，因此为命题。$\square$

**命题 AR.12（传递性）.** 若 $x<y$ 且 $y<z$，则 $x<z$。

**证明（证明核）.** 展开为 $y-x>0$ 和 $z-y>0$。取截断代表元的有理正下界 $\varepsilon,\delta$；由加法和距离三角不等式得到
$$
z-x=(z-y)+(y-x)>0
$$
的有理正下界，例如 $\min(\varepsilon,\delta)/2$。目标是命题，故可对两个存在性截断消去。$\square$

**命题 AR.13（加法单调性）.** 若 $x<y$，则
$$
x+z<y+z.
$$

**证明.** 因
$$
(y+z)-(x+z)=y-x
$$
由环律和加法逆元律给出，正性保持。$\square$

**命题 AR.14（正数乘法封闭）.** 若 $0<x$ 且 $0<y$，则 $0<x\cdot y$。

**证明状态.** 取 $x,y$ 的正有理下界和足够小误差预算。若 $x$ 近似于 $q>\varepsilon$，$y$ 近似于 $r>\delta$，则 $xy$ 近似于 $qr$，且 $qr>0$。误差由 AR.8 控制。目标是命题，可对正性截断消去。

## AR.5 倒数与域结构

**定义 AR.15（远离零）.** 定义
$$
x\#0\coloneqq (x<0)+(0<x)
$$
或其命题截断版本。构造性域结构通常要求对远离零的元素构造倒数。

**定理 AR.16（远离零元素有倒数）.** 若 $x\#0$，则存在 $y:\mathbb R_C$ 使
$$
x\cdot y=1.
$$

**证明（证明核）.** 若 $0<x$，由正性取得有理下界 $\varepsilon>0$，从而 $x$ 的足够精细近似均远离零。定义倒数近似
$$
y(\delta)\coloneqq 1/a(\gamma_\delta)
$$
其中 $\gamma_\delta$ 由下界 $\varepsilon$ 和误差预算选择。函数 $t\mapsto1/t$ 在 $[\varepsilon/2,\infty)$ 上 Lipschitz，故 $y$ 是 Cauchy 近似。令 $y=\mathsf{lim}(y)$，用极限唯一性证明 $xy=1$。$x<0$ 情形转化为 $-x>0$。$\square$

**定理 AR.17（构造性完备有序域接口）.** $\mathbb R_C$ 满足：

1.  交换环结构；
2.  远离零元素有倒数；
3.  $<$ 为命题值严格序；
4.  加法和乘法与 $<$ 相容；
5.  每个 Cauchy 近似有唯一极限。

**证明状态.** 由 AK.9、AR.9、AR.12-AR.16 汇总。该结构是构造性完备有序域接口；若要得到 classical complete ordered field 的三歧性和最小上界性质，需要额外原则或改用 Dedekind/located cut 证明。
