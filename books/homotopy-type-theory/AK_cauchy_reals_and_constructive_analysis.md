# 附录 AK：Cauchy 实数与构造性分析证明核

本附录补入 HoTT Book 第十一章方向：用高阶归纳-归纳类型构造 Cauchy 实数，并说明该构造对本书分析接口的影响。这里的目标不是发展完整分析学，而是把实数对象、相等、完备性和代数结构的最小严格接口写清楚。

## AK.1 有理数输入

**输入 AK.1（有理数域）.** 固定集合 $\mathbb Q$，带有 $0,1,+,-,\cdot,<,\le$，满足有序域公理。其相等类型是命题，$<$ 和 $\le$ 是命题值关系，并有阿基米德型估计用于选择足够小的正有理数。

**使用边界.** 本书不重建整数、有理数和有序域的全部代数。整数加法群律见附录 W；有理数域可由集合商或标准库构造作为外部代数输入。

## AK.2 预度量空间

**定义 AK.2（有理值预度量结构）.** 类型 $X$ 的预度量结构为关系
$$
d_X(x,y)<\varepsilon
$$
其中 $x,y:X$，$\varepsilon:\mathbb Q_{>0}$，并满足：

1.  反身性：$d_X(x,x)<\varepsilon$；
2.  对称性：$d_X(x,y)<\varepsilon\to d_X(y,x)<\varepsilon$；
3.  三角不等式：若 $d_X(x,y)<\varepsilon$ 且 $d_X(y,z)<\delta$，则
    $d_X(x,z)<\varepsilon+\delta$；
4.  预度量分离：若对所有 $\varepsilon>0$ 有 $d_X(x,y)<\varepsilon$，则 $x=y$。

**定义 AK.3（Cauchy 近似）.** 到预度量空间 $X$ 的 Cauchy 近似由函数
$$
a:\mathbb Q_{>0}\to X
$$
和证明
$$
\prod_{\varepsilon,\delta>0}d_X(a(\varepsilon),a(\delta))<\varepsilon+\delta
$$
组成。

**定义 AK.4（近似相等）.** 两个 Cauchy 近似 $a,b$ 相等，若
$$
\prod_{\varepsilon,\delta>0}d_X(a(\varepsilon),b(\delta))<\varepsilon+\delta.
$$

**命题 AK.5（近似相等是等价关系）.** 关系 AK.4 自反、对称、传递。

**证明（证明核）.** 自反性是 Cauchy 条件。对称性由预度量对称性。传递性：若 $a\sim b$ 且 $b\sim c$，给定 $\varepsilon,\delta>0$，选择 $\gamma>0$ 使得 $\gamma+\gamma<\varepsilon+\delta$ 的剩余预算可容纳中间项；由
$$
d(a(\varepsilon/2),b(\gamma))<\varepsilon/2+\gamma
$$
和
$$
d(b(\gamma),c(\delta/2))<\gamma+\delta/2
$$
及三角不等式得
$$
d(a(\varepsilon/2),c(\delta/2))<\varepsilon/2+2\gamma+\delta/2<\varepsilon+\delta.
$$
再用 Cauchy 条件把 $a(\varepsilon)$ 与 $a(\varepsilon/2)$、$c(\delta/2)$ 与 $c(\delta)$ 连接。严格展开需固定有理数预算的具体选择；数学内容只用稠密性和有序域代数。$\square$

## AK.3 HoTT Book 的 Cauchy 实数 HIT

**输入 AK.6（Cauchy 实数高阶归纳-归纳规范）.** Cauchy 实数 $\mathbb R_C$ 与距离关系
$$
d(x,y)<\varepsilon
$$
可由高阶归纳-归纳类型给出，包含：

1.  有理数嵌入 $\mathsf{rat}:\mathbb Q\to\mathbb R_C$；
2.  对每个 Cauchy 近似 $a:\mathbb Q_{>0}\to\mathbb R_C$，极限点
    $$
    \mathsf{lim}(a):\mathbb R_C;
    $$
3.  距离关系的构造子，确保 $\mathsf{rat}$ 保距、$\mathsf{lim}$ 确为极限；
4.  路径构造子：若 $d(x,y)<\varepsilon$ 对所有 $\varepsilon>0$ 成立，则 $x=y$；
5.  集合截断构造子：$\mathsf{isSet}(\mathbb R_C)$。

**使用边界.** 这是高级 HIT/HIIT 输入。Brough 2026 给出该构造的严格路线；本书据此把 AK.6 视为外部输入，而不是从基础 HIT 清单附录 L 自动推出。

**命题 AK.7（实数是集合）.** $\mathbb R_C$ 是集合。

**证明.** 由输入 AK.6 的集合截断构造子。$\square$

**命题 AK.8（极限唯一性）.** 若 $x,y:\mathbb R_C$ 都是同一 Cauchy 近似 $a$ 的极限，则 $x=y$。

**证明（证明核）.** 要用 AK.6 的分离路径构造子，只需证明对任意 $\varepsilon>0$ 有 $d(x,y)<\varepsilon$。由 $x$ 是 $a$ 的极限，取预算 $\varepsilon/2$ 得 $d(x,a(\varepsilon/2))<\varepsilon/2$；由 $y$ 是 $a$ 的极限得 $d(a(\varepsilon/2),y)<\varepsilon/2$。三角不等式给 $d(x,y)<\varepsilon$。$\square$

**定理 AK.9（Cauchy 完备性）.** 每个 $\mathbb R_C$ 中的 Cauchy 近似都有极限。

**证明.** 令 $a$ 为 Cauchy 近似，定义极限为 $\mathsf{lim}(a)$。极限性质由输入 AK.6 的距离构造子给出；唯一性由 AK.8。$\square$

## AK.4 代数结构

**定义 AK.10（实数加法）.** 对 $x,y:\mathbb R_C$，用 $\mathbb R_C$ 的双重递归定义 $x+y$。在有理数生成元上：
$$
\mathsf{rat}(q)+\mathsf{rat}(r)\coloneqq\mathsf{rat}(q+r).
$$
在极限生成元上：
$$
\mathsf{lim}(a)+y\coloneqq
\mathsf{lim}(\lambda \varepsilon.\,a(\varepsilon/2)+y),
$$
右变量同理。

**证明义务 AK.11（well-definedness）.** 定义 AK.10 必须证明：

1.  若 $a$ 是 Cauchy 近似，则 $\lambda\varepsilon.\,a(\varepsilon/2)+y$ 是 Cauchy 近似；
2.  加法尊重分离路径构造子；
3.  加法尊重集合截断；
4.  双变量递归的两个极限分支相容。

**命题 AK.12（加法群律，证明核 / 外部输入）.** $\mathbb R_C$ 在 $+$ 下构成交换群。

**证明状态.** 证明对两个变量作 HIIT 递归/归纳。生成元为有理数时归约到 $\mathbb Q$ 的交换群律；极限情形逐点使用归纳假设，并用极限唯一性 AK.8 关闭等式。关键技术是统一有理数误差预算，保证所有 Cauchy 近似和极限证明相容。本书不把全部预算算术逐行重写。

**定义 AK.13（乘法与序）.** 乘法、负元、绝对值、序关系可类似定义，但乘法需要局部有界性估计。序关系通常定义为 Cauchy 近似的正距离证据或 Dedekind 风格切割谓词的 HIIT 翻译。

**证明义务 AK.14（有序域结构）.** 要把 $\mathbb R_C$ 作为有序域，需证明：

1.  $+$ 和 $\cdot$ 的环律；
2.  非零元倒数存在或局部倒数构造；
3.  $<$ 与 $\le$ 的传递性、三歧性或相应构造性替代；
4.  加法和乘法对序的单调性；
5.  Cauchy 完备性与代数运算相容。

这些义务是构造性分析教材的主体。附录 AR 进一步展开乘法、序、远离零倒数和构造性完备有序域接口；本附录只给出 HIIT、完备性和加法入口。

## AK.5 与选择公理和 setoid 的关系

**事实 AK.15（避免 countable choice 的口径）.** HoTT Book 的 HIIT Cauchy 实数构造旨在避免传统 Cauchy 完备化中常见的 countable choice 依赖，并避免 Bishop setoid 构造在每个定理中携带等价关系 bookkeeping。

**边界.** 这不是说所有分析定理都无选择原则。具体定理必须逐条记录是否使用 countable choice、dependent choice、excluded middle、propositional resizing 或 quotient exactness。

## AK.6 本附录的接口

1.  第八章的商与截断只能给出集合层 quotient；Cauchy 实数的 HIIT 构造需要更强输入 AK.6。
2.  附录 AR 继续展开 $\mathbb R_C$ 的环、序、倒数和完备有序域接口。
3.  第十五章来源定位应把 2026 年 Cauchy 实数工作列为“构造性分析”入口。
4.  第十七章不能只说“HoTT 可做分析”；必须区分实数对象、完备性、代数结构、序结构和具体分析定理。
