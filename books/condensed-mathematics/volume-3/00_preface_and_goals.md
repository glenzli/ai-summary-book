# 序章：从函数空间到复几何定理

在射影直线上，线丛 $\mathcal O(d)$ 的上同调可以用两个仿射坐标图直接计算；换到一般
紧复流形后，局部截面却是无限维 Fréchet 空间，商映射的像未必闭，逐点向量空间方法
不再保存分析信息。第三卷研究的正是这段距离：怎样从带拓扑的全纯函数与 Dolbeault
复形出发，经过 liquid/analytic 派生范畴，得到有限性、对偶、代数化比较和
Riemann--Roch 公式，同时不把其中任何深层复几何定理伪装成形式范畴论。

贯穿各章的可计算对象是 $(\mathbb P^1,\mathcal O(d))$。它既能显示 Čech 微分怎样留下
有限维上同调，也能用留数写出 Serre 配对，还能同时接受代数与解析解释；最终，
$\chi(\mathcal O(d))=d+1$ 又与 Chern character 和 Todd class 的积分吻合。一般定理
将以精确外部输入出现，而这个模型及接受输入后的凝聚数学形式后果都在正文中完成。

## 0.1 一条可计算的线索

取标准覆盖
$U_0=\{X_0\ne0\}$、$U_\infty=\{X_1\ne0\}$，并令
$z=X_1/X_0$、$w=z^{-1}$。对 $\mathcal O(d)$，Čech 微分为

$$
\delta:\mathbb C[z]\oplus\mathbb C[w]\longrightarrow
\mathbb C[z,z^{-1}],
\qquad
\delta(f_0,f_\infty)=z^df_\infty(z^{-1})-f_0.
$$

非负幂来自 $f_0$，不大于 $d$ 的幂来自 $z^df_\infty(z^{-1})$。因此

$$
h^0(\mathbb P^1,\mathcal O(d))=
\begin{cases}
d+1,&d\ge0,\\
0,&d<0,
\end{cases}
$$

而

$$
h^1(\mathbb P^1,\mathcal O(d))=
\begin{cases}
0,&d\ge-1,\\
-d-1,&d\le-2.
\end{cases}
$$

这个两项复形已经包含后续四种现象。两项上同调都有限维；因
$\omega_{\mathbb P^1}\simeq\mathcal O(-2)$，互补指数的单项式由留数完美配对；
同一线丛的代数与解析截面相符；交错维数恒为 $d+1$，正好等于
$\int_{\mathbb P^1}\operatorname{ch}(\mathcal O(d))
\operatorname{td}(T_{\mathbb P^1})$。附录 H、T、U 分别保留这些计算的完整多项式和
高维版本，正文会在需要时写出核心步骤，而不是只指向附录。

## 0.2 三种同时存在的结构

复几何中的一个截面至少有三种身份。作为集合元素，它满足 sheaf 粘合；作为
$\mathcal O_X$-模元素，它参与核、余核与有限表示；作为拓扑向量空间中的点，它还受
一致收敛、闭像与连续算子控制。前两种结构由经典环化空间和相干层语言表达，第三种
结构通过

$$
S\longmapsto\operatorname{Cont}(S,V)
$$

把拓扑向量空间 $V$ 送入凝聚对象，再由第二卷的 $p$-liquid 理论辨认适合做同调代数的
分析对象。

这并不意味着“凝聚化”自动保持所有正合列。若连续满射没有局部连续提升，凝聚
cokernel 仍可能与朴素拓扑商不一致。第三章在使用 Dolbeault 复形时会明确区分：各项
进入 liquid 范畴是对象层输入，复形确实计算导出全局截面则是更强的建模输入。

## 0.3 外部定理与书内推论

以下深层结果保持为外部输入：Oka--Cartan 相干性、Dolbeault 局部引理中的解析估计、
椭圆 Fredholm/Hodge 理论、Grauert 有限性、一般 Serre--Grothendieck 对偶、GAGA、
Chern 类构造以及 HRR/GRR。它们的精确版本和来源登记在
[SOURCES.md](SOURCES.md)、[附录 F](F_classical_complex_geometry_prerequisites.md)、
[附录 AQ](AQ_main_theorem_package_and_condensed_closure.md) 与
[附录 AR](AR_clausen_scholze_complex_geometry_core_theorem_atlas.md)。

接受这些输入以后，仍有大量必须在书内证明的内容：fine resolution 为何计算
$R\Gamma$，有限性如何沿有限 resolution 和谱序列传播，链级积分配对为何下降到
上同调，exact GAGA 等价为何诱导 bounded derived 等价，以及 Euler characteristic
为何定义 $K$-群同态。这些形式后果构成第三至七章的论证主干。

## 0.4 对象链而非定理清单

后续各章沿下面的对象链相接：

$$
\mathcal O_X(U)
\rightsquigarrow
\underline{\mathcal O_X(U)}
\rightsquigarrow
\mathcal A_X^{0,\bullet}(E)
\rightsquigarrow
R\Gamma(X,E)
\rightsquigarrow
\operatorname{Tr}_X,\ \chi(X,E).
$$

第一箭头保留函数空间拓扑，第二个复形把全纯层解析为 fine sheaves，第三箭头产生有限
上同调，最后的 trace 同时进入 Serre 对偶和 Riemann--Roch。第八章再把这些构造放入
$f^*,f_*,f_!,f^!,\otimes,R\mathcal Hom$ 的关系网，并以具体数学开放问题收束。

## 练习

**练习 0.1.** 由上面的 Čech 微分证明
$H^1(\mathbb P^1,\mathcal O(-2))$ 由 $z^{-1}$ 的类生成。

**练习 0.2.** 说明留数配对为何把
$H^0(\mathbb P^1,\mathcal O(d))$ 的基 $1,z,\ldots,z^d$
与 $H^1(\mathbb P^1,\mathcal O(-d-2))$ 的某组基逐项配对。

**练习 0.3.** 给出一个无限维 Fréchet 空间复形，其上同调有限维；指出仅看复形各项
维数为何不能推出或否定有限性。
