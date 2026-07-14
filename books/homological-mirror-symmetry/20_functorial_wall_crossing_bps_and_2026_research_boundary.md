# 第二十章：函子化、墙穿越与尚未统一的范畴结构

前几章出现了三类变化：移除 stop 改变 wrapped 范畴，改变稳定参数可能改变半稳定对象，而改变奇点或势函数又改变 matrix factorization 模型。把它们统称为“函子化 HMS”会掩盖重要差别；尤其“BPS category”目前不是一个跨文献固定的数学定义。本章不编排近期论文清单，而从三个可分开检验的问题出发：几何操作是否给出增强函子，参数跨墙是否给出确定的等价或自同构，以及物理术语是否已经选定具体范畴模型。第十七至十九章提供函子、不变量与奇点模型，下面只在这些类型都明确时陈述结论。

## 20.1 范畴值图与自然镜像等价

**约定 20.1.** 记 $\operatorname{Cat}^{\mathrm{perf}}_k$ 为小、幂等完备、
$k$-线性 stable categories 的 Morita 局部化。若使用 dg 或 $A_\infty$
模型，先取其 perfect-module stable category 再进入该口径。

**定义 20.2（函子化 HMS 数据）.** 设 $I$ 是小索引范畴，
$$
\mathcal A,\mathcal B:I\longrightarrow
\operatorname{Cat}^{\mathrm{perf}}_k
$$
是两个范畴值图。一个函子化 HMS 数据是自然等价
$E:\mathcal A\Rightarrow\mathcal B$；具体地，它包括每个 $i\in I$ 上的
Morita equivalence $E_i:\mathcal A(i)\simeq\mathcal B(i)$，以及每条箭头
$a:i\to j$ 上的可逆 $2$-morphism
$$
\eta_a:E_j\circ\mathcal A(a)
\Longrightarrow \mathcal B(a)\circ E_i,
\tag{20.1}
$$
并要求对恒等箭头和可复合箭头满足通常的单位与 cocycle coherence。

**命题 20.3.** 设 $F:\operatorname{Cat}^{\mathrm{perf}}_k\to\mathcal D$ 是
任意 Morita-invariant functor。定义 20.2 的 $E$ 诱导图
$F\mathcal A,F\mathcal B:I\to\mathcal D$ 之间的自然等价。特别地，
$K$-theory 和 Hochschild homology 上得到自然交换图，而不只是逐点同构。

**证明.** 对每个 $i$，Morita 不变性使 $F(E_i)$ 为等价；对每条 $a$，
functoriality 把 (20.1) 送到可逆 $2$-morphism。$E$ 的单位与复合 coherence
经 $F$ 后仍成立，故这些分量组成自然等价。证毕。

Pasquarella 2025 的论文把 topological field theory 与 BPS 计数视为发展
函子化 HMS 的动机，并讨论具体物理设置；其公开摘要本身强调该形式主义仍需
进一步发展。因此这里把它作为研究视角，而不从中抽取一个尚未给定完整
假设的普遍等价定理。

## 20.2 稳定条件与一堵可计算的墙

“参数跨墙”只有在稳定性概念固定后才有数学含义。下面采用有限秩数值格上的
Bridgeland 口径。

**定义 20.4（稳定条件）.** 设 $\mathcal D$ 是 $k$-线性三角范畴，
$v:K_0(\mathcal D)\to\Gamma_{\mathrm{ch}}$ 是到有限秩自由 Abelian group
$\Gamma_{\mathrm{ch}}$ 的类映射。一个
数值 Bridgeland stability condition 是 $(Z,\mathcal P)$，其中
$Z:\Gamma_{\mathrm{ch}}\to\mathbb C$ 为群同态，$\mathcal P(\phi)$ 为
$\mathcal D$ 的 full additive subcategories，并满足：

1. 非零 $E\in\mathcal P(\phi)$ 时，
   $Z(v(E))=m(E)e^{i\pi\phi}$，$m(E)>0$；
2. $\mathcal P(\phi+1)=\mathcal P(\phi)[1]$；
3. 若 $\phi_1>\phi_2$，则
   $\operatorname{Hom}(\mathcal P(\phi_1),\mathcal P(\phi_2))=0$；
4. 每个非零对象有有限 Harder--Narasimhan filtration，其因子相位严格下降；
5. slicing 局部有限，并满足 support property：存在
   $\Gamma_{\mathrm{ch},\mathbb R}:=\Gamma_{\mathrm{ch}}\otimes_\mathbb Z\mathbb R$
   上的范数
   和常数 $C>0$，使每个 semistable $E$ 满足
   $|Z(v(E))|\ge C\lVert v(E)\rVert$。

**定义 20.5（固定分解的墙）.** 固定
$\gamma=\gamma_1+\gamma_2\in\Gamma_{\mathrm{ch}}$。使两个中心荷同相的参数满足
$$
\operatorname{Im}\!\left(
Z(\gamma_1)\overline{Z(\gamma_2)}
\right)=0,\qquad
\operatorname{Re}\!\left(
Z(\gamma_1)\overline{Z(\gamma_2)}
\right)>0.
\tag{20.2}
$$
在 support property 保持的参数域中，这个实余维一 locus 称为该分解的墙；
其补集的连通分支称为 chambers。

**例 20.6（$A_2$ quiver 的稳定对象跨墙）.** 令
$\mathcal D=\mathrm D^b\operatorname{Rep}_k(1\to2)$，简单对象为
$S_1,S_2$。唯一非分裂扩张
$$
0\longrightarrow S_2\longrightarrow P_1\longrightarrow S_1
\longrightarrow0
\tag{20.3}
$$
对应 $\operatorname{Ext}^1(S_1,S_2)\cong k$。取心
$\operatorname{Rep}_k(1\to2)$，并令 $Z(S_1),Z(S_2)$ 位于严格上半平面。
由于 $S_2$ 是 $P_1$ 的唯一非平凡真子对象，$P_1$ 稳定当且仅当
$$
\phi(S_2)<\phi(P_1),
$$
而 $Z(P_1)=Z(S_1)+Z(S_2)$ 的相位严格位于两者之间。因此
$P_1$ 在 $\phi(S_2)<\phi(S_1)$ 的 chamber 稳定，在反向 chamber 不稳定，
并在 $\phi(S_1)=\phi(S_2)$ 的墙上严格 semistable。这里跨墙改变的是稳定
对象集合；它本身尚未产生 derived autoequivalence。

## 20.3 Spherical twist 不是“墙”的同义词

**定义 20.7.** 设 $\mathcal D$ 是 Hom-finite、具有 Serre pairing 的
$k$-线性增强三角范畴。对象 $S$ 称为 $d$-spherical，若
$$
\operatorname{Ext}^\ast(S,S)\cong k\oplus k[-d]
$$
作为分次代数成立，且复合与 Serre pairing 给出所需非退化性。定义 twist
$T_S$ 的对象值由 exact triangle
$$
\mathbf R\operatorname{Hom}(S,E)\otimes S
\xrightarrow{\mathrm{ev}}E\longrightarrow T_S(E)\longrightarrow[1]
\tag{20.4}
$$
给出。

**外部输入定理 20.8（Seidel--Thomas）.** 在定义 20.7 的增强与 finiteness
假设下，$T_S$ 是 $\mathcal D$ 的 exact autoequivalence；适当的 spherical
collections 还产生 braid-group actions。来源：Seidel--Thomas,
*Braid group actions on derived categories of coherent sheaves*,
arXiv:math/0001043。

**命题 20.9.** 对任意 $E\in\mathcal D$，spherical twist 在 Grothendieck
group 上作用为
$$
[T_S(E)]=[E]-\chi(S,E)[S].
\tag{20.5}
$$

**证明.** 在 $K_0$ 中对 (20.4) 使用 exact triangle 的加性，得到
$$
[T_S(E)]=[E]-
[\mathbf R\operatorname{Hom}(S,E)\otimes S].
$$
有限维复形 $\mathbf R\operatorname{Hom}(S,E)$ 的类等于其 Euler 特征，
所以第二项为 $\chi(S,E)[S]$。证毕。

某些镜像族的 monodromy 或稳定性墙确实由 spherical twists 描述，但这需要
额外定理把几何 continuation、vanishing cycle 或 moduli wall 与 (20.4)
识别。仅观察到 (20.5) 的反射公式，不能推出范畴级 wall-crossing functor。

## 20.4 “BPS category”必须展开成具体数据

**约定 20.10（model-specified BPS package）.** 本书只在给出五元组
$$
(\mathcal C,\Gamma_{\mathrm{ch}},v,Z,\mathfrak s)
\tag{20.6}
$$
后使用“BPS category”一词。其中 $\mathcal C$ 是明确的 dg/$A_\infty$/stable
category，$\Gamma_{\mathrm{ch}}$ 是 charge lattice，
$v:K_0(\mathcal C)\to\Gamma_{\mathrm{ch}}$ 是类映射，
$Z:\Gamma_{\mathrm{ch}}\to\mathbb C$ 是 central charge，而 $\mathfrak s$ 记录实际采用的
stability、Calabi--Yau、monoidal 或 orientation data。这是本书的类型约定，
不是宣称不同物理文献中的 BPS categories 已有统一定义。

**例 20.11.** 三种常见载体说明为何 (20.6) 中的 $\mathcal C$ 不能省略。

1. 奇点或 Landau--Ginzburg 模型可取 equivariant matrix-factorization
   category；其 $\mathbb Z/2$ 或 graded 结构和群作用属于对象定义。
2. Donaldson--Thomas 语境可取带 $3$-Calabi--Yau 结构的范畴及其 Hall/CoHA
   构造；计数依赖 stability 与 orientation data。
3. 辛几何语境可取 stopped、wrapped 或 Fukaya--Seidel category；charge
   可能来自相对同调，而 wall-crossing 还依赖 continuation 与 disk counts。

这三类对象之间有许多已知或猜想的联系，却不存在一个形式操作把名称相同的
五元组自动识别。

**命题 20.12（已给定等价时的数据传递）.** 设两个 (20.6) 型数据包之间有
Morita equivalence $E:\mathcal C_A\simeq\mathcal C_B$，并给定格同构
$e:\Gamma_{\mathrm{ch},A}\to\Gamma_{\mathrm{ch},B}$，满足
$$
e\circ v_A=v_B\circ K_0(E),\qquad Z_B\circ e=Z_A.
\tag{20.7}
$$
则对应对象的 central charges 相同，且 $K$-theory 与 Hochschild homology
经 $E$ 同构。若还声称 semistable objects、Calabi--Yau traces 或 Hall
products 对应，必须另加 $E$ 与 $\mathfrak s_A,\mathfrak s_B$ 的相容性。

**证明.** Central charge 结论由 (20.7) 逐对象代入；其余两项由 Morita
不变性。稳定性、trace 与 Hall multiplication 不是裸 Morita 类型的函数，
故不在这些假设下自动传递。证毕。

## 20.5 三个已证锚点及其不同结论

截至本章资料核查日期，下面三项可以作为统一问题的锚点，但不能互相替换。

**来源状态 20.13.**

1. Ganatra--Hanlon--Hicks--Pomerleano--Sheridan 对其论文规定的一大类
   Batyrev mirror pairs 证明 HMS；结果在特征零及除有限多个素数外的正特征
   成立。精确 polytope 类与 category model 仍按定理 12.7 的来源版本使用。
2. Abouzaid--Auroux 对 maximally degenerating hypersurfaces in
   $(\mathbb C^\ast)^n$ 构造 fiberwise wrapped 模型，并得到定理 13.7 的
   coherent-side quasi-embedding；缺少目标侧生成性时不能改写为等价。
3. Lekili--Ueda 对非 Calabi--Yau 型 Brieskorn--Pham 情形证明 Rabinowitz
   Fukaya category 与 equivariant matrix factorizations 的 HMS，并由此计算
   Rabinowitz Floer homology；其 Rabinowitz 模型与群作用是假设的一部分。

第一项是大类 mirror pairs 的等价，第二项明确停在 quasi-embedding，第三项
则属于奇点/Rabinowitz 模型。它们共同支持更函子化的图景，却没有给出连接
三类范畴的自然等价族。

## 20.6 一个类型明确的开放问题

**开放问题 20.14.** 是否存在一个几何索引 $\infty$-category $I$，其对象同时
容纳适当的 stopped Liouville sectors、Landau--Ginzburg degenerations 与
奇点数据，并存在
$$
\mathcal A,\mathcal B:I\longrightarrow
\operatorname{Cat}^{\mathrm{perf}}_k,
\qquad \mathcal A\simeq\mathcal B,
\tag{20.8}
$$
使得：

1. 在已知子范畴上恢复 sectorial descent、stop removal、Batyrev HMS、
   fiberwise wrapped quasi-embedding 与 Rabinowitz HMS；
2. 几何路径和回路诱导的 continuation/monodromy 与 B-side spherical twists
   或 mutations 通过 (20.1) 相容；
3. 选择 (20.6) 的附加数据后，稳定性墙和相应计数在两侧对应。

主要障碍不是缺少一个总称，而是箭头的方差不同、source categories 的完成
方式不同、virtual/compactness inputs 尚未统一，以及某些已知结果只有
fully faithful/quasi-embedding 而非 equivalence。任何解答都必须先解决这些
类型差异，不能从对象层镜像字典直接推出 (20.8)。

这三条前沿不能靠一个总称合并。函子化 HMS 要比较增强范畴值图；wall-crossing 要先固定稳定性数据、类与跨墙变换；BPS 范畴则必须落实为某个 DT、CoHA、matrix-factorization 或 Fukaya 型模型。Batyrev pairs、代数环面 hypersurfaces 与 Brieskorn--Pham 奇点给出已经证明但假设各异的锚点，统一框架仍是开放问题，而不是可在现有定理之间形式插值得到的结论。

## 练习

**练习 20.1.** 展开定义 20.2 对两条可复合箭头
$i\xrightarrow a j\xrightarrow b k$ 的 coherence 条件，并证明命题 20.3
在该复合上仍交换。

**练习 20.2.** 在例 20.6 中取
$Z(S_1)=e^{3\pi i/4}$、$Z(S_2)=e^{\pi i/4}$，判断 $P_1$ 是否稳定；交换两个
中心荷后再判断一次。

**练习 20.3.** 由 triangle (20.4) 推导公式 (20.5)，并计算
$[T_S(S)]$。

**练习 20.4.** 对来源状态 20.13 的三项分别写出 A-side category、B-side
category 与结论强度，并指出为何第二项不能由名称“HMS”自动升级为等价。
