# 第四章：有限谱的 type、nilpotence 与 periodicity

Morava $K$-理论只有在它能把有限谱真正分层时才具有分类力量。对非零有限谱，首次非消失的 $K(n)$ 给出 type；令人意外的是，这个数值不仅检测同调，还完全控制有限谱的 thick 子范畴，并通过 $v_n$ 自映射产生周期族。本章先从定义与低高度例子说明 type 怎样运作，再精确陈述 nilpotence、periodicity 和 thick subcategory 三个深定理，最后展示它们怎样被调用而不把外部证明压缩成口号。所需范畴论来自第一章，$K(n)$ 来自第二、三章；大型定理采用 Hopkins--Smith 与 Devinatz--Hopkins--Smith 的外部输入口径。

## 4.1 Type

**定义 4.1.** 对非零有限 $p$-局部谱 $X$，若集合
$$
\{r\in\mathbb Z_{\ge0}\mid K(r)_*X\ne0\}
$$
非空，则定义 $X$ 的 type 为该集合的最小元：
$$
\operatorname{type}(X)=
\min\{r\ge0\mid K(r)_*X\ne0\}.
$$
这里 $K(0)=H\mathbb Q$。零谱的 type 约定为 $\infty$。

这个定义把“首次非消失高度”直接写入量词，不把较低高度的消失偷偷
交给后续定理。

**外部输入定理 4.2（finite detection，CHT-P0-07）.** 若有限
$p$-局部谱 $X$ 满足 $K(r)_*X=0$ 对所有 $r\ge0$，则 $X\simeq0$。
这是 Hopkins--Smith II, Theorem 14 的直接后果：此时 $X$ 的 Morava
检测集合与零谱同为空集，故 $X$ 与零谱 Bousfield 等价；取 smash 因子
$\mathbb S_{(p)}$ 即得 $X\simeq0$。因此定义 4.1 中的集合对非零有限谱
非空，而良序性保证其最小元存在。本书不重证 Theorem 14。

**命题 4.3.** 若有限谱 $X$ 有 type $n$，则
$$
K(m)_*X=0\quad(m<n),
\qquad K(n)_*X\ne0.
$$

**证明.** 两项都是定义 4.1 中最小元条件的展开。证毕。

**外部输入 4.3A（有限谱的高度单调性）.** 对有限 $p$-局部谱 $X$，
$$
K(r)_*X=0\Longrightarrow K(m)_*X=0\quad(0\le m<r).
$$
等价地，若 $X$ 的 type 为 $n$，则 $K(m)_*X\ne0$ 对每个 $m\ge n$
成立。这是 invariant-prime-ideal/thick-subcategory 定理包的一部分；
Hopkins--Smith II 在 Theorem 7 前明确记录了子范畴
$\mathcal C_{r+1}\subseteq\mathcal C_r$ 的非形式性。

**警告 4.4.** 不能把定义 4.1 改写成“只要 $K(n)_*X\ne0$ 就 type
$n$”。由外部输入 4.3A，type $n$ 的非零有限谱会被每个更高的
$K(m)$（$m\ge n$）检测到；type 只记录第一个非消失高度。

## 4.2 厚子范畴

**定义 4.5.** 有限谱范畴的 thick 子范畴是全稳定子范畴 $\mathcal T\subseteq\mathbf{Sp}^{\omega}_{(p)}$，对 cofiber、悬挂、脱悬挂和 retract 封闭。

**定义 4.6.** 对 $n\ge0$，定义
$$
\mathcal C_n=
\{X\in\mathbf{Sp}^{\omega}_{(p)}\mid K(i)_*X=0\text{ 对所有 }0\le i<n\},
$$
其中空条件给出 $\mathcal C_0=\mathbf{Sp}^{\omega}_{(p)}$，并记
$\mathcal C_\infty=\{0\}$。按定义，$\mathcal C_n$ 正是 type 至少为
$n$ 的有限谱；接受外部输入 4.3A 后，也可写成文献常用形式
$$
\mathcal C_n=\{X\mid K(n-1)_*X=0\}\qquad(n\ge1).
$$

**命题 4.7.** $\mathcal C_n$ 是 thick 子范畴。

**证明.** 对每个 $0\le i<n$，$K(i)_*(-)$ 是同调理论，因此把
cofiber 序列送到长正合列。若两项的 $K(i)_*$ 为零，第三项也为零；
对所有 $i<n$ 同时应用此论证即可。悬挂只平移次数。若 $X$ 是 $Y$
的 retract，则每个 $K(i)_*X$ 都是 $K(i)_*Y$ 的 retract。因此
$\mathcal C_n$ 对等价、cofiber、悬挂、脱悬挂和 retract 封闭。证毕。

**外部输入定理 4.8（Hopkins--Smith thick subcategory theorem，
CHT-P0-06）.** 有限 $p$-局部谱范畴中的每个 thick 子范畴都等于
唯一的 $\mathcal C_n$，其中 $n\in\mathbb Z_{\ge0}\cup\{\infty\}$。
非零 proper 情形对应 $1\le n<\infty$。而且
$$
\mathcal C_0\supsetneq\mathcal C_1\supsetneq\mathcal C_2\supsetneq\cdots
\supsetneq\mathcal C_\infty.
$$
该定理见 Hopkins--Smith II, Theorem 7；严格包含还使用各高度有限谱
的存在性。本文不重证该定理。

**使用说明.** 本书把定理 4.8 作为外部输入。任何用 type 分类 thick 子范畴的证明必须引用该定理，不能只引用 $K(n)_*$ 的系数环形式。

## 4.3 Nilpotence

**定义 4.9.** 设 $R$ 是 ring spectrum。元素 $\alpha\in\pi_dR$ 称为 nilpotent，若存在 $N>0$ 使得乘法意义下
$$
\alpha^N=0\in\pi_{Nd}R.
$$

**外部输入定理 4.10（Devinatz--Hopkins--Smith nilpotence，
CHT-P0-04）.** 设 $R$ 是有单位、同伦结合的 ring spectrum，令
$$
h_{MU}:\pi_*R\longrightarrow MU_*R
$$
为 $MU$-Hurewicz 映射。则 $\ker(h_{MU})$ 中每个元素在 graded ring
$\pi_*R$ 中幂零。这里采用 Devinatz--Hopkins--Smith I, Theorem
1(i)；本书只使用这个 ring-spectrum 版本，不重证其 nilpotence 技术。

**外部输入定理 4.10A（Serre finiteness）.** 对每个 $d>0$，球谱稳定同伦群 $\pi_d\mathbb S$ 是有限 Abel 群。

**推论 4.11（Nishida nilpotence）.** 每个正次数元素 $\alpha\in\pi_d\mathbb S$（$d>0$）都是幂零的。

**证明.** 由定理 4.10A，$\alpha$ 是 torsion 元素。另一方面
$$
MU_*\cong\mathbb Z[x_1,x_2,\ldots],\qquad |x_i|=2i,
$$
作为 Abel 群无挠。因此 $\alpha$ 在 unit-induced Hurewicz map
$$
\pi_d\mathbb S\longrightarrow MU_d
$$
下的像必为零。对 ring spectrum $R=\mathbb S$ 应用定理 4.10，得到某个 $N>0$ 使 $\alpha^N=0$。证毕。

## 4.4 Periodicity 和 $v_n$ self-map

**定义 4.12.** 设 $n\ge1$，且 $X$ 是 type $n$ 有限谱。一个 $v_n$
self-map 是正次数映射
$$
v:\Sigma^dX\to X,\qquad d>0,
$$
使 $K(n)_*v$（计入次数平移后）为同构，并使 $K(m)_*v$ 对每个
$m\ne n$ 为幂零自映射。这里“幂零”指某次迭代
$$
v^N:\Sigma^{Nd}X\longrightarrow X
$$
在 $K(m)_*$ 上为零。

**约定 4.12A（高度零）.** $v_0$-self-map 允许次数 $d=0$。本书固定
$$
p:\mathbb S_{(p)}\longrightarrow\mathbb S_{(p)}
$$
作为基本 $v_0$-map：它在 $K(0)=H\mathbb Q$ 上可逆，而在
$K(m)$（$m\ge1$）上为零。不能把定义 4.12 的 $d>0$ 条件套到高度零。

**外部输入定理 4.13（Hopkins--Smith periodicity，CHT-P0-05）.**
设 $n\ge1$。

1. 每个 type $n$ 有限 $p$-局部谱存在 $v_n$-self-map；
2. 同一有限谱上的两个 $v_n$-self-maps $v,w$ 存在正整数 $a,b$，使
   $v^a$ 与 $w^b$ 在悬挂次数对齐后同伦；
3. type $n$ 有限谱之间的任意映射与所选 $v_n$-self-maps 在分别取幂
   后交换。

存在性定位为 Hopkins--Smith II, Theorem 9；第二、三项分别定位为
Corollaries 3.7 和 3.8。本书不重证这些结论。

**定义 4.14.** 给定 $v_n$ self-map $v:\Sigma^dX\to X$，令
$\widetilde v:X\to\Sigma^{-d}X$ 为脱悬挂伴随。其 telescope 定义为
$$
v^{-1}X=\operatorname*{colim}\left(
X\xrightarrow{\widetilde v}\Sigma^{-d}X
\xrightarrow{\Sigma^{-d}\widetilde v}\Sigma^{-2d}X\to\cdots\right).
$$
对 $n\ge1$，其 Bousfield 类记作 $T(n)$。对 $n=0$，本书定义
$$
T(0)=\operatorname*{colim}
(\mathbb S_{(p)}\xrightarrow p\mathbb S_{(p)}\xrightarrow p\cdots)
\simeq H\mathbb Q.
$$

**外部输入 4.14A（telescope 选择无关，CHT-P1-19）.** 对固定
$n\ge1$，不同 type $n$ 有限谱及不同 $v_n$-self-maps 所得 telescopes
Bousfield 等价。self-map 选择无关使用 Corollary 3.7；跨有限谱的比较
还使用 Hopkins--Smith II, Theorem 14 的 finite-spectrum class
invariance，不能只归因于“取 colimit”。

**警告 4.15.** $T(n)$ 与 $K(n)$ 在 Bousfield 局部化上不能默认相同。telescope conjecture 断言的正是这类比较；2023 年之后高度至少 $2$ 的一般等同性已被反例否定。

## 4.5 Type 的低阶例子

**例 4.16.** 球谱 $\mathbb S_{(p)}$ 是 type $0$。

**证明.** $K(0)_*\mathbb S_{(p)}=H\mathbb Q_*\mathbb S_{(p)}\cong\mathbb Q$ 集中在 degree $0$，非零。按 type $0$ 定义，球谱为 type $0$。证毕。

**例 4.17.** Moore spectrum $M(p)$ 不是 type $0$。

**证明.** $M(p)$ 是 cofiber
$$
\mathbb S_{(p)}\xrightarrow{p}\mathbb S_{(p)}\to M(p).
$$
张量 $H\mathbb Q$ 后，乘以 $p$ 在 $H\mathbb Q$ 上为等价，故 cofiber 为零。因此 $K(0)_*M(p)=0$，所以不是 type $0$。证毕。

**命题 4.18.** 对每个素数 $p$，Moore spectrum $M(p)$ 是 type $1$。

**证明.** 例 4.17 已给出 $K(0)_*M(p)=0$。对 defining cofiber
sequence 施加 $K(1)_*(-)$，得到长正合列，其中
$$
K(1)_*\mathbb S_{(p)}\xrightarrow{p}K(1)_*\mathbb S_{(p)}
$$
是零映射，因为 $K(1)_*=\mathbb F_p[v_1^{\pm1}]$ 的特征为 $p$。
因此从其 cokernel 到 $K(1)_*M(p)$ 的映射给出单射
$$
K(1)_*\hookrightarrow K(1)_*M(p),
$$
特别地 $K(1)_*M(p)\ne0$。按定义 4.1，$M(p)$ 的 type 恰为 $1$。
这个计算对所有素数成立；需要 periodicity theorem 的是 $v_1$-map 的
存在性，而不是 type 的判定。证毕。

## 4.6 Thick theorem 的使用格式

**命题 4.19.** 若 $\mathcal T$ 是有限 $p$-局部谱的非零 thick 子范畴，则存在唯一 $n$ 使
$$
\mathcal T=\mathcal C_n.
$$

**证明.** 这是外部输入定理 4.8 对非零情形的直接应用。唯一性来自
同一定理包中的严格包含链
$$
\mathcal C_0\supsetneq\mathcal C_1\supsetneq\cdots.
$$
严格包含本身也是外部输入的一部分，因为需要存在每个 type 的有限谱。
因此本段是“由外部定理推出的推论”，不是书内重证 thick theorem。
证毕。

**使用规则 4.20.** 调用 thick theorem 时必须说明：

1. 对象在有限 $p$-局部谱范畴；
2. 子范畴对 cofiber、悬挂、脱悬挂和 retract 封闭；
3. 结论分类的是 thick 子范畴，不是 arbitrary full subcategory；
4. 若涉及 tensor ideal，应额外说明张量封闭是否已知。

## 4.7 有限谱的色层分类

有限谱是 chromatic theory 最刚性的对象。type 给出高度分层，thick subcategory theorem 说明这个分层穷尽所有 thick 子范畴，periodicity theorem 给出 $v_n$ self-map 和 telescopes。三大定理均是外部输入，本书后续会频繁使用，但不会伪装成内部证明。

## 练习

**练习 4.1.** 证明命题 4.7 中 retract 封闭的细节。

**练习 4.2.** 若 $X$ 是 type $n$ 有限谱，解释为什么 $K(n)_*X\ne0$ 不推出 $L_{K(n)}X\simeq X$。

**练习 4.3.** 写出 telescope $v^{-1}X$ 的 colimit 定义中每个箭头的
次数，检查它们都变成从同一悬挂规范下的谱到下一项的映射。
