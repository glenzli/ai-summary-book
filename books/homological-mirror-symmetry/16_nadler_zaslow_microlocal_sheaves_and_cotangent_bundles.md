# 第十六章：Nadler-Zaslow、microlocal sheaves 与 cotangent bundles

普通 support 只记录层在哪些点非零，却不能区分一个截面能否越过某个超曲面继续延拓；这个方向性缺失正由 cotangent bundle 中的 microsupport 补上。另一方面，conormal branes 也天然位于 $T^*Q$，其 Floer 态射与标准、余标准层之间的 Ext 具有相同局部模型。Nadler--Zaslow 对应把这两种描述提升为范畴关系，wrapped/stopped 扩展又把无穷远 Reeb chords 对应到无界但紧的层对象。本章在第二章的 dg 层范畴与第六章的 wrapped 口径之间建立精确字典，并以一维分层的可计算例子解释 microsupport 的方向。

## 16.1 Constructible sheaves

**定义 16.1.** 设 $Q$ 是实解析流形，$\mathcal S$ 是有限 Whitney stratification。一个复形 $\mathcal F$ 称为 $\mathcal S$-constructible，若对每个 stratum $S\in\mathcal S$，限制 $\mathcal F|_S$ 的 cohomology sheaves 为局部常值且有限维。constructible sheaves 的 dg category 记为
$$
\operatorname{Sh}_c(Q).
$$

**定义 16.2.** 设 $\mathcal F$ 是 $Q$ 上的 sheaf complex。称
$(x_0,\xi_0)\notin SS(\mathcal F)$，若存在 $T^\ast Q$ 中含
$(x_0,\xi_0)$ 的开邻域 $U$，使得对任意 $C^1$ 函数 $\varphi$ 和任意
$x$ 满足 $(x,d\varphi_x)\in U$，都有局部上同调 stalk
$$
\bigl(R\Gamma_{\{y:\varphi(y)\ge\varphi(x)\}}\mathcal F\bigr)_x\simeq0.
\tag{16.1}
$$
其补集 $SS(\mathcal F)\subset T^\ast Q$ 称为 $\mathcal F$ 的 microsupport。
它是闭的、对正实数缩放协变量不变的 conic subset。式 (16.1) 表示：沿
$d\varphi_x$ 指定的协方向没有阻碍截面传播的局部上同调。

**警告 16.3.** Microsupport 不是普通 support。普通 support 位于 $Q$，microsupport 位于 $T^\ast Q$，记录方向性信息。

**例 16.3A（半直线的方向）.** 在 $Q=\mathbb R$ 上，令
$F_+=k_{[0,\infty)}$ 为闭半直线常值层的零延拓，令
$F_-=k_{(0,\infty)}$ 为开半直线常值层的零延拓。除各自 support 上的零截面外，
边界 $0$ 处有
$$
SS(F_+)\cap T_0^\ast\mathbb R=\{\xi\,dx:\xi\ge0\},\qquad
SS(F_-)\cap T_0^\ast\mathbb R=\{\xi\,dx:\xi\le0\}.
\tag{16.2}
$$
例如对 $F_+$ 取 $\varphi(x)=x$，支撑条件
$\{\varphi\ge\varphi(0)\}=[0,\infty)$ 检测到非零局部上同调，所以正协方向
属于 microsupport；开半直线的边界方向相反。相同的普通 support 闭包因此
可以带有不同的 microlocal 边界数据。

## 16.2 Cotangent Fukaya category

**定义 16.4.** $T^\ast Q$ 带 canonical Liouville form。其 conic Lagrangians 与 constructible sheaves 的 microsupport 条件相匹配。对 conic Lagrangian $\Lambda\subset T^\ast Q$，记
$$
\operatorname{Sh}_\Lambda(Q)
$$
为 microsupport 包含于 $\Lambda$ 的 constructible sheaves category。

**外部输入定理 16.5（Nadler--Zaslow）.** 对 compact real analytic
manifold $Q$，取有限 constructibility 条件下的 dg derived category 与
Nadler--Zaslow 所规定的 exact、tame、graded branes in $T^\ast Q$ 之导出
Fukaya category，则存在把标准/余标准 sheaves 送到相应 conormal branes 的
增强等价。不同文献的 opposite-category 和 brane sign 约定可能改变箭头方向；
本章固定为 sheaf 到 brane 的上述方向。
来源：Nadler--Zaslow, *Constructible Sheaves and the Fukaya Category* 给出
quasi-embedding；Nadler, *Microlocal branes are constructible sheaves*,
arXiv:math/0612399，证明该 quasi-embedding 本质满，从而升级为上述
quasi-equivalence。

**外部输入定理 16.6（GPS wrapped/microlocal 扩展）.** 设 $Q$ 是实解析
流形，$\Lambda\subset S^\ast Q$ 是 subanalytic isotropic subset。则
$T^\ast Q$ 在 $\Lambda$ 处 stopped 的 partially wrapped Fukaya category
等价于无界 derived sheaf category 中 microsupport at infinity 包含于
$\Lambda$ 的 compact objects 子范畴。若一个 Weinstein sector 具有 stable
polarization，GPS 的 embedding argument 还给出相应 sheaf-theoretic 描述。
来源：Ganatra--Pardon--Shende, *Microlocal Morse theory of wrapped Fukaya
categories*, arXiv:1809.08807。

## 16.3 标准对象

**定义 16.7.** 若 $S\subset Q$ 是 stratum，标准 sheaf 是 $j_{S!}k_S$，余标准 sheaf 是 $j_{S*}k_S$，其中 $j_S:S\hookrightarrow Q$。

**解释 16.8.** 在 Fukaya 侧，标准和余标准 sheaves 对应于正/负 conormal branes。Morphism spaces 的计算对应 sheaf Ext groups 与 Floer cochains 的比较。

**命题 16.8A（conormal 是 Lagrangian）.** 若 $S\subset Q$ 是光滑
submanifold，则
$$
T_S^\ast Q=\{(x,\xi):x\in S,\ \xi|_{T_xS}=0\}
\subset T^\ast Q
$$
是维数 $\dim Q$ 的 Lagrangian submanifold。

**证明.** 若 $\dim Q=n$、$\dim S=s$，则 conormal fiber 维数为 $n-s$，
故 $\dim T_S^\ast Q=s+(n-s)=n$。设 $\theta$ 是 $T^\ast Q$ 的 tautological
$1$-form。对 $v\in T_{(x,\xi)}T_S^\ast Q$，其底空间投影
$d\pi(v)$ 属于 $T_xS$，所以
$$
\theta_{(x,\xi)}(v)=\xi(d\pi(v))=0.
$$
因此 $\theta|_{T_S^\ast Q}=0$，从而
$\omega=d\theta$ 在该子流形上限制为零。它是半维 isotropic 子流形，故为
Lagrangian。证毕。

**命题 16.9.** 若 $Q$ 有有限 stratification 且 constructible category 由标准 sheaves 生成，则对应 Fukaya category 由相应 conormal branes split-generate。

**证明.** 由 Nadler-Zaslow 型等价，标准 sheaves 的生成性传递到对应 conormal branes。增强等价保持 thick closure 和 split-closure。证毕。

## 16.4 HMS 中的作用

Microlocal sheaf 模型在 HMS 中有三种作用：

1. 把 Fukaya category 的计算转化为 sheaf-theoretic Ext 计算；
2. 为 stopped/partially wrapped categories 提供 cosheaf/sheaf 模型；
3. 在 skeleton 上给出 combinatorial category，从而连接 tropical geometry 和 mirror symmetry。

**命题 16.10（经 sheaf 模型传递 HMS）.** 若 $M$ 是具有相应
microlocal theorem 所需 stable polarization 的 Weinstein sector，
$\mathfrak L$ 是其 skeleton，且已有明确的 Morita 等价
$$
\mathcal W(M)\simeq \operatorname{Sh}_{\mathfrak L}(Q),
$$
又有 B-side 增强范畴 $\mathcal B$ 及 Morita 等价
$$
\operatorname{Sh}_{\mathfrak L}(Q)\simeq \mathcal B
$$
则 $\mathcal W(M)\simeq_{\mathrm{Morita}}\mathcal B$。

**证明.** Morita equivalences 在 Morita homotopy category 中可复合；复合
上述两条等价即可。第一条等价的 stable-polarization、stop 和 compact-object
假设不能由第二条 B-side 比较补偿。证毕。

Microsupport 把“不能沿哪个协方向传播”编码成 conic Lagrangian 条件，标准与余标准层则对应两种方向相反的 conormal branes。外部输入等价因此把 Floer 计算转为 Ext 计算，并把 stop 与 microsupport 约束放进同一个局部化框架。更一般 Weinstein sector 只有在稳定极化等附加假设下才可采用这样的 sheaf 描述；在该范围内，生成性和 descent 可以从层侧传回 wrapped 侧。

## 练习

**练习 16.1.** 区分 support 与 microsupport，并给出常值 sheaf 的 microsupport。

**练习 16.2.** 对区间分层，写出标准 sheaves 和余标准 sheaves 的例子。

**练习 16.3.** 解释 conormal bundle 为什么是 Lagrangian。

**练习 16.4.** 对命题 16.10 写出两条等价在对象和 morphism complexes 上
的复合，并解释为何只比较两个三角影子不够。
