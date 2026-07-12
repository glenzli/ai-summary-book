# 附录 Q：Koszul complex、bar-cobar 谱序列与计算样例

本附录补充定义 8.4--定义 8.16、定义 9.14--定理 9.20 和定义 I.11--命题 I.21 中仍偏抽象的部分：如何在低权重和低 arity 中看见 Koszul complex、bar-cobar differential 和谱序列。完整 Koszul 判别仍是外部输入；本附录只证明可直接由定义推出的计算。

## Q.1 二次 operad 的权重分解

设 $k$ 为域，链复形采用同调分次。设 $E$ 是集中在 arity $2$ 的对称序列，$\mathbb F(E)$ 是自由 operad。按顶点数定义权重：
$$
\mathbb F(E)=\bigoplus_{r\ge0}\mathbb F^{(r)}(E).
$$
其中
$$
\mathbb F^{(0)}(E)=I,\qquad
\mathbb F^{(1)}(E)=E.
$$

**命题 Q.1.** 若 $E$ 集中在 arity $2$，则 $\mathbb F^{(r)}(E)$ 由有 $r$ 个二元顶点的树给出，故只在 arity $r+1$ 非零。

**证明.** 每个顶点有两个输入。设树有 $r$ 个内部顶点、$l$ 个叶。对有根树，边数计数给出
$$
\sum_{v\in V(T)}\operatorname{in}(v)=l+r-1.
$$
左侧等于 $2r$，故 $l=r+1$。自由 operad 的权重 $r$ 部分正由 $r$ 个生成元装饰的树组成，所以只在 arity $r+1$ 出现。$\square$

**定义 Q.2.** 一个二次 operad 写作
$$
\mathcal P=\mathcal P(E,R)=\mathbb F(E)/(R),
$$
其中
$$
R\subset \mathbb F^{(2)}(E).
$$
权重 $2$ 是关系所在层，对应有两个二元顶点、三个叶的树。

## Q.2 权重 $2$ 的关系空间

在非对称情形中，若 $E=k\cdot\mu$ 由一个二元生成元生成，则
$$
\mathbb F^{(2)}(E)(3)
=
k\{\mu\circ_1\mu,\ \mu\circ_2\mu\}.
$$

**定义 Q.3.** 非对称 associative operad $\operatorname{Ass}_{ns}$ 是
$$
\operatorname{Ass}_{ns}=\mathbb F_{ns}(\mu)/(r),
$$
其中
$$
r=\mu\circ_1\mu-\mu\circ_2\mu.
$$

**命题 Q.4.** $\operatorname{Ass}_{ns}$-代数正是非含单位结合代数。

**证明.** 一个 $\mathbb F_{ns}(\mu)$-代数等价于给定一个二元运算
$$
m:A\otimes A\to A.
$$
关系 $r=0$ 在 endomorphism operad 中的像为
$$
m(m(a,b),c)-m(a,m(b,c)).
$$
因此商 operad 的代数等价于满足结合律的二元运算，即非含单位结合代数。$\square$

**说明 Q.5.** 含单位结合代数需要 arity $0$ 单位或非齐次关系；二次 reduced Koszul 理论先处理非含单位核心，再通过 augmented/unital 版本恢复单位。

## Q.3 二次对偶的低阶形状

本节采用定义 8.15 的 cooperad 口径：二次对偶 cooperad
$$
\mathcal P^¡=\mathcal C(sE,s^2R)\subseteq\mathbb T^c(sE)
$$
的 cogenerators 是 $sE$，weight-$2$ corelations 是 $s^2R$。线性对偶 $E^\vee$ 与正交补 $R^\perp$ 用于定义 8.12 的对偶 operad $\mathcal P^!$，不是本节 $\mathcal P^¡$ 的 cogenerators。

**命题 Q.6.** 在二元二次情形，$\mathcal P^¡$ 的权重 $0,1,2$ 部分满足：
$$
(\mathcal P^¡)^{(0)}=I,\qquad
(\mathcal P^¡)^{(1)}=sE,
$$
而权重 $2$ 部分是嵌入 $\mathbb T^c(sE)^{(2)}$ 的子对象 $s^2R$。

**证明.** 定义 8.15 把 $\mathcal P^¡$ 定义为 $\mathbb T^c(sE)$ 中由 $s^2R$ 决定的最大子cooperad。权重 $0$ 是 coaugmentation 单位，权重 $1$ 尚无 corelation，故为 $sE$；在 weight $2$，定义要求且只要求该部分落入 $s^2R$，所以恰为显示的子对象。$\square$

**警告 Q.7.** 文献中 $\mathcal P^!$、$\mathcal P^¡$、$\mathcal P^{\ash}$ 的 suspension 和 dual convention 不同。本书把有限型线性对偶 operad 写作 $\mathcal P^!$，把 $\mathcal C(sE,s^2R)$ 写作 $\mathcal P^¡$；后者的 $s$ 是定义 9.2 的链悬挂。把 $\mathcal P^¡$ 与 $\mathcal P^!$ 作有限型比较时才另外出现线性对偶、sign representation 和 operadic suspension，不能把定义 E.11 的 $\Lambda$ 直接代入 $sE$。

## Q.4 Koszul twisting morphism 的低权重行为

设 $\kappa:\mathcal P^¡\to\mathcal P$ 为 Koszul twisting morphism。

**定义 Q.8.** $\kappa$ 在权重 $1$ 上由 desuspension 给出：
$$
sE\longrightarrow E.
$$
在权重 $0$ 和权重 $\ge2$ 上为零。

**命题 Q.9.** 右 Koszul complex
$$
K_r(\mathcal P)=\mathcal P^¡\circ_\kappa\mathcal P
$$
的 twisting differential 只作用于 $\mathcal P^¡$ 中被 infinitesimal decomposition 分出的一个权重 $1$ cogenerator。

**证明.** Twisting differential 的定义为
$$
\mathcal P^¡
\xrightarrow{\Delta_{(1)}}
\mathcal P^¡\circ_{(1)}\mathcal P^¡
\xrightarrow{\operatorname{id}\circ_{(1)}\kappa}
\mathcal P^¡\circ_{(1)}\mathcal P
\to
\mathcal P^¡\circ\mathcal P.
$$
由于 $\kappa$ 在权重 $1$ 外为零，只有 infinitesimal decomposition 中内层因子权重为 $1$ 的项存活。$\square$

**推论 Q.10.** $K_r(\mathcal P)$ 的 differential 降低 cooperad 权重 $1$，并把该权重转移为右侧 operad 的一个复合操作。

**证明.** Q.9 的证明写出 twisting differential 为
$\Delta_{(1)}$ 后接 $\operatorname{id}\circ_{(1)}\kappa$。前者抽出一个
内层 cooperad 因子，后者只在该因子权重为 $1$ 时非零，并用 twisting
morphism 把它替换成 $\mathcal P$ 中的操作。因此左侧 cooperad 总权重
恰降 $1$，右侧增加一次 operadic composition。$\square$

## Q.5 非对称 Ass 的 Koszul complex 形状

本节内部检查定向结合律的终止性与唯一临界对，但“合流二次 rewriting 推出 Koszul”以及完整 exactness 仍使用 LV-3/LV-2。令 $\operatorname{Ass}_{ns}$ 为定义 Q.3 的非对称 associative operad。它的 Koszul dual cooperad 在非对称 reduced convention 下与 coassociative cooperad 对应，记为 $\operatorname{coAss}_{ns}$。

**外部输入定理 Q.11（非对称结合 operad 的 Koszul 性；LV-3）.**
$\operatorname{Ass}_{ns}$ 是 Koszul；等价地，
$$
\operatorname{coAss}_{ns}\circ_\kappa\operatorname{Ass}_{ns}\to I
$$
是 quasi-isomorphism。

**证明路线（内部 rewriting 检查 + 外部 Koszul 判据）.** 把唯一关系定向为
$$
\mu\circ_1\mu\longrightarrow\mu\circ_2\mu,
\qquad (xy)z\longrightarrow x(yz).
$$
对一棵平面二元树 $T$，令
$$
L(T)=\sum_{v\in V(T)}\#\{\text{$v$ 的左输入子树中的内部顶点}\}.
$$
一次改写 $\mu(\mu(A,B),C)\to\mu(A,\mu(B,C))$ 使 $L$ 减少
$1+|V(A)|$，所以改写终止。唯一的重叠临界单项式是三顶点左梳
$((ab)c)d$。先改写顶层得到
$$
((ab)c)d\longrightarrow (ab)(cd)\longrightarrow a(b(cd));
$$
先改写左下层则得到
$$
((ab)c)d\longrightarrow (a(bc))d
\longrightarrow a((bc)d)
\longrightarrow a(b(cd)).
$$
故唯一临界对合流。Loday--Vallette Theorem 8.1.1 及其紧随的
$\operatorname{As}$ 例子（LV-3）把这个 terminating/confluent quadratic
rewriting system 升级为 Koszul 性，进而由 LV-2 得到上式的
quasi-isomorphism。这里的外部输入恰是“合流二次 rewriting system
推出 Koszul”；终止性和临界对检查已在上面完成。

**低阶形状 Q.12.** 在 arity $1$，
$$
K_r(\operatorname{Ass}_{ns})(1)\cong k
$$
集中在单位层。

在 arity $2$，只有一个二元生成层，complex 的非单位部分由
$$
s\mu^\vee\otimes \mathbf 1
\quad\text{和}\quad
\mathbf 1\otimes\mu
$$
类型的项组成；twisting differential 把前者送到后者，符号由 suspension convention 决定。

在 arity $3$，树形项对应两种括号：
$$
(\mu\circ_1\mu),\qquad(\mu\circ_2\mu).
$$
Koszul differential 的边界正检测二者在 $\operatorname{Ass}_{ns}$ 中被关系
$$
\mu\circ_1\mu-\mu\circ_2\mu
$$
识别。

**说明 Q.13.** 上述描述解释 Koszul complex 如何“解析单位”：arity $>1$ 的同调应消失，arity $1$ 保留单位。但这种消失不是由低阶形状自动推出，而是 Ass Koszul 性的内容。

## Q.6 Bar construction 的低权重 differential

设 $\mathcal P$ 是 augmented dg-operad。Bar construction
$$
B\mathcal P=\mathbb T^c(s\overline{\mathcal P})
$$
的 differential 为
$$
d=d_{\mathrm{int}}+d_{\mathrm{bar}}.
$$

**命题 Q.14.** $d_{\mathrm{bar}}$ 在二顶点树上由 operad composition 给出：
$$
s p\ \circ_i\ s q
\longmapsto
\pm s(p\circ_i q).
$$

**证明.** Bar differential 的二次部分由收缩一条内部边定义。二顶点树只有一条内部边；收缩该边正是把两个顶点装饰按对应 slot 作 operad partial composition。悬挂因子移过张量因子产生符号。$\square$

**命题 Q.15.** 在三顶点树上，$d_{\mathrm{bar}}^2=0$ 等价于 operad partial composition 的结合律加 Koszul 符号抵消。

**证明.** 对三顶点树连续收缩两条内部边有两种顺序。若两条边嵌套，两个顺序对应 operad 的嵌套结合律；若两条边分离，两个顺序对应交换两个收缩操作并产生 Koszul 反号。每个最终一顶点树项出现两次且符号相反，故和为零。$\square$

## Q.7 Cobar construction 的低权重 differential

设 $\mathcal C$ 是 coaugmented conilpotent dg-cooperad。Cobar construction
$$
\Omega\mathcal C=\mathbb F(s^{-1}\overline{\mathcal C})
$$
的 differential 为
$$
d=d_{\mathrm{int}}+d_{\mathrm{cobar}}.
$$

**命题 Q.16.** $d_{\mathrm{cobar}}$ 在一个生成元 $s^{-1}c$ 上由 infinitesimal decomposition 给出：
$$
d_{\mathrm{cobar}}(s^{-1}c)
=
\sum \pm (s^{-1}c')\circ_i(s^{-1}c'')
$$
其中
$$
\Delta_{(1)}(c)=\sum c'\circ_i c''.
$$

**证明.** Cobar construction 的二次 differential 首先对 cooperad 元素作 infinitesimal decomposition，然后对每个分量 desuspend，并作为 derivation 延拓到自由 operad。公式正是该定义在 generator 上的写法。$\square$

**命题 Q.17.** $d_{\mathrm{cobar}}^2=0$ 的二次部分由 cooperad 余结合律给出。

**证明.** 对 $c$ 连续作两次 infinitesimal decomposition。两种分解顺序对应先分解外层再分解内层，或先分解内层再分解外层。Cooperad 余结合律识别这些分量；desuspension 和 derivation 符号使相同项成对抵消。$\square$

## Q.8 Bar-cobar counit 的低权重形式

设 $\mathcal P$ 是 augmented dg-operad。Bar-cobar counit
$$
\epsilon:\Omega B\mathcal P\to\mathcal P
$$
的底层 graded operad 是
$$
\mathbb F\bigl(s^{-1}\overline{B\mathcal P}\bigr),
\qquad
\overline{B\mathcal P}
=
\bigoplus_{q\ge1}\mathbb T^{c,(q)}(s\overline{\mathcal P}).
$$
因此每棵正 bar 权重的树本身都是一个 cobar 生成元。Counit 在这些生成元上的定义是
$$
\epsilon(s^{-1}b)
=
\begin{cases}
p,& b=sp\in\mathbb T^{c,(1)}(s\overline{\mathcal P}),\\
0,& b\in\mathbb T^{c,(q)}(s\overline{\mathcal P}),\ q>1.
\end{cases}
$$
然后由自由 operad 的泛性质唯一延拓为 graded operad morphism。特别地，“bar 权重 $q>1$ 的单个 cobar 生成元映到 $0$”与“若干 bar 权重 $1$ 的 cobar 生成元在外层自由 operad 中复合后映到 $\mathcal P$ 的相应复合”是两个不同陈述。

**命题 Q.18.** 上述 $\epsilon$ 的链映射检查可在自由 operad 生成元上完成。关键的 bar 权重 $2$ 情形中，bar 收缩项与 cobar 分解项在施加 $\epsilon$ 后给出同一个 partial composition，且符号相反，因而抵消。

**证明.** 权重 $1$ 的生成元为 $s^{-1}sp$。其 differential 只有由 $p$ 的内部微分诱导的线性项，desuspension-suspension 的符号约定给出
$$
\epsilon d(s^{-1}sp)=d_{\mathcal P}p
=d_{\mathcal P}\epsilon(s^{-1}sp).
$$

现取 bar 权重 $2$ 的单个 cobar 生成元
$$
z=s^{-1}(sp\circ_i sq),
$$
其中括号内表示一棵二顶点 bar 树。按定义 $\epsilon(z)=0$。忽略仍保持 bar 权重 $2$、因而被 $\epsilon$ 杀掉的内部微分项，$d(z)$ 的两个结构项具有形状
$$
(-1)^\chi s^{-1}s(p\circ_i q)
-
(-1)^\chi
\bigl(s^{-1}sp\bigr)\circ_i\bigl(s^{-1}sq\bigr).
$$
第一项来自 bar differential 收缩二顶点树的唯一内部边；第二项来自 cobar differential 沿该边作 reduced infinitesimal decomposition。两项的相对负号正是 bar 与 cobar 在总微分中的 suspension/desuspension 约定，$\chi$ 是两处共同的 Koszul 符号。施加 $\epsilon$ 后，两项分别成为
$$
(-1)^\chi(p\circ_i q),
\qquad
-(-1)^\chi(p\circ_i q),
$$
所以和为 $0=d_{\mathcal P}\epsilon(z)$。

若单个 cobar 生成元的 bar 权重 $q>2$，bar 收缩项仍有权重 $q-1>1$，故映到 $0$；reduced cobar 分解把 $q$ 写成两个正权重之和，其中至少一个大于 $1$，故对应外层自由 operad 复合也含一个映到 $0$ 的因子。于是链映射等式在所有生成元上成立。Cobar differential 是导子，而 $\epsilon$ 按自由 operad 泛性质作复合延拓，所以等式推广到整个 $\Omega B\mathcal P$。$\square$

**外部输入定理 Q.19（modern bar-cobar counit；FRE-4）.** 采用 Fresse, arXiv:0902.0177 的模型：$\mathcal P$ 是 $C$-cofibrant augmented operad，并且其增广理想满足
$$
\overline{\mathcal P}(0)=\overline{\mathcal P}(1)=0.
$$
在该假设下，来源第 3.14 节给出 bar-cobar counit
$$
\Omega B\mathcal P\to\mathcal P
$$
为 weak equivalence。若还要称其为 operad 的 cofibrant resolution，必须另外验证 $\Omega B\mathcal P$ 在所用 dg-operad 模型结构中 cofibrant；FRE-5 是 algebra-level replacement theorem，不能用来填补这个 operad-level 条件。

**证明路线（外部输入）.** 来源使用 connected tree filtration 和 twisted-composite acyclicity。命题 Q.14--命题 Q.18 只核对低权重 differential 与 counit 的链映射性质，不证明全权重 weak equivalence。

## Q.9 谱序列页面的使用边界

设 $C$ 是带递增滤过 $F_pC$ 的链复形，且 differential 满足
$$
d(F_pC)\subseteq F_pC.
$$
可形成谱序列 $E^r_{p,q}$。

**说明 Q.20.** 在 bar-cobar 证明中，必须使用两个方向不同的权重滤过：

1. $d_{\mathrm{int}}$ 保持权重；
2. bar differential 降低权重，所以 $B\mathcal P$ 使用
   $$
   F_pB\mathcal P=\bigoplus_{q\le p}B^{(q)}\mathcal P;
   $$
3. cobar differential 增加权重，所以 $\Omega\mathcal C$ 使用
   $$
   F^p\Omega\mathcal C=\bigoplus_{q\ge p}\Omega^{(q)}\mathcal C;
   $$
4. twisted differential 改变 cooperad/operad 权重分配，必须另选总权重并证明它被 differential 保持。

Connected 情形下，命题 I.21 的树计数使这些滤过逐 arity 有限。非 connected 情形不能只把上标改成下标；反例 I.22.1 表明完成化会把直和改成乘积。

**警告 Q.21.** “谱序列退化”不是一个无条件短语。必须说明：

1. 收敛到哪个 filtered homology；
2. 是否强收敛；
3. 是否有 boundedness 或 complete/exhaustive 条件；
4. 页码 $E^r$ 的 convention。

对本书的 connected quadratic Koszul 应用，逐 arity 有限性给出强收敛。对含 unary cogenerators、completed cobar 或无界总权重的应用，本书不作收敛声明，除非另附完整 filtration theorem。

## Q.10 小结

本附录给出以下可检查内容：

1. 二元二次 operad 的权重 $r$ 只在 arity $r+1$ 出现；
2. 非对称 associative operad 的关系是 $\mu\circ_1\mu-\mu\circ_2\mu$；
3. Koszul twisting morphism 只在权重 $1$ 非零；
4. Koszul differential 通过 infinitesimal decomposition 检测关系；
5. bar differential 是收缩内部边；
6. cobar differential 是展开 cooperad 分解；
7. bar-cobar counit 杀掉 bar 权重 $>1$ 的单个 cobar 生成元，并在权重 $2$ 上由 bar 收缩项与 cobar 分解项抵消保证链映射性；
8. 完整 exactness、Koszul 性和 resolution 结论仍是外部输入。

这些计算应作为定义 8.16、定理 9.20 和定义 10.5--定义 10.10 中同伦代数构造的局部校验模板。
