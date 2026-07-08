# 第七章：紧化、T-duality 和 D-branes

## 本章目标

本章建立 string theory 中第一个真正偏离点粒子直觉的结构：圆紧化有 winding sectors，闭弦谱在小圆和大圆之间等价；对开弦，T-duality 把 Neumann 边界条件变为 Dirichlet 边界条件，从而迫使 D-branes 出现。

## 依赖前置知识

需要第二章的边界变分、第四章的模展开与质量公式、第六章中顶点算子和开闭弦谱的对应。

## 7.1 圆紧化和 winding sectors

**定义 7.1（圆紧化）.** 令一个 target coordinate 取值于半径为 $R$ 的圆：
$$
X\sim X+2\pi R.
$$
闭弦 embedding 允许 winding number $w\in\mathbb Z$：
$$
X(\tau,\sigma+2\pi)=X(\tau,\sigma)+2\pi wR.
$$
沿圆方向的 center-of-mass momentum 由单值波函数条件量子化为
$$
p=\frac nR,\qquad n\in\mathbb Z.
$$

**定义 7.2（左右动量）.** 在本书闭弦归一化下，紧化方向的零模可写成左右动量
$$
p_L=\frac nR+\frac{wR}{\alpha'},
\qquad
p_R=\frac nR-\frac{wR}{\alpha'}.
$$
对应的零模项为
$$
X_{\mathrm{zero}}(\tau,\sigma)
=x+\frac{\alpha'}2(p_L+p_R)\tau
+\frac{\alpha'}2(p_L-p_R)\sigma.
$$

**命题 7.3（左右动量公式）.** 定义 7.2 满足 momentum quantization 与 winding boundary condition，并且由二者唯一确定。

**证明.** 由零模表达式得
$$
X(\tau,\sigma+2\pi)-X(\tau,\sigma)
=\pi\alpha'(p_L-p_R).
$$
令其等于 $2\pi wR$，得到
$$
p_L-p_R=\frac{2wR}{\alpha'}.
$$
另一方面，圆方向中心动量为
$$
p=\frac12(p_L+p_R)=\frac nR.
$$
解这两个线性方程即得公式。$\square$

## 7.2 紧化闭弦质量公式

**命题 7.4（圆紧化质量公式）.** 若只把一个空间方向紧化为 $S^1_R$，其余方向非紧，并记非紧动量平方给出的质量为 $M^2$，则玻色闭弦满足
$$
M^2
=
\left(\frac nR\right)^2
+\left(\frac{wR}{\alpha'}\right)^2
+\frac2{\alpha'}(N+\tilde N-2a),
$$
以及 level matching 条件
$$
N-\tilde N+nw=0.
$$
临界玻色弦取 $a=1$。

**证明.** 紧化方向左右动量进入 Virasoro 零模：
$$
L_0=\frac{\alpha'}4(-M^2+p_L^2)+N,\qquad
\tilde L_0=\frac{\alpha'}4(-M^2+p_R^2)+\tilde N.
$$
物理条件 $L_0=\tilde L_0=a$ 给出
$$
M^2=p_L^2+\frac4{\alpha'}(N-a)
=p_R^2+\frac4{\alpha'}(\tilde N-a).
$$
两式相加并代入 $p_L,p_R$ 得质量公式；两式相减并用
$$
p_L^2-p_R^2=\frac{4nw}{\alpha'}
$$
得 level matching。$\square$

**例 7.5（self-dual radius）.** 当 $R=\sqrt{\alpha'}$ 时，$n$ 和 $w$ 的贡献对称。某些带非零 winding 与 momentum 的态可变为 massless，从而增强 worldsheet current algebra。玻色弦圆紧化在该半径有典型的 enhanced symmetry；完整 current algebra 描述依赖附录 B、C。

## 7.3 T-duality

**定理 7.6（闭弦谱的 T-duality）.** 圆紧化闭弦谱在变换
$$
R\longleftrightarrow \frac{\alpha'}R,
\qquad
n\longleftrightarrow w
$$
下不变。该变换等价于
$$
p_L\mapsto p_L,\qquad p_R\mapsto -p_R.
$$

**证明.** 直接代入定义 7.2：
$$
p'_L=\frac w{\alpha'/R}+\frac{n(\alpha'/R)}{\alpha'}=
\frac{wR}{\alpha'}+\frac nR=p_L,
$$
而
$$
p'_R=\frac w{\alpha'/R}-\frac{n(\alpha'/R)}{\alpha'}=
\frac{wR}{\alpha'}-\frac nR=-p_R.
$$
命题 7.4 的质量公式只依赖 $p_L^2,p_R^2$ 或等价的 $n^2/R^2+w^2R^2/\alpha'^2$，level matching 中 $nw$ 也不变，故谱不变。$\square$

**定义 7.7（dual coordinate）.** T-dual coordinate $\widetilde X$ 在平坦圆紧化中由
$$
\partial_\tau \widetilde X=\partial_\sigma X,\qquad
\partial_\sigma \widetilde X=\partial_\tau X
$$
定义，等价地在左右移动分解中保持 $X_L$ 而改变 $X_R$ 的符号。

**注 7.8（CFT 等价）.** 定理 7.6 只是谱层面的陈述。完整 T-duality 是 compact boson CFT 的等价，包括 OPE、operator algebra 和 partition function。该完整陈述在本书中作为二维 CFT 标准结果使用。

## 7.4 开弦边界条件和 T-duality

**命题 7.9（边界条件互换）.** 对开弦紧化方向，T-duality 沿该方向把 Neumann 边界条件变为 Dirichlet 边界条件，反之亦然。

**证明.** Polyakov 作用量的边界变分含有
$$
\delta S_{\partial\Sigma}
=-\frac1{2\pi\alpha'}\int_{\partial\Sigma}d\tau\,
\delta X\,\partial_\sigma X.
$$
Neumann 条件为 $\partial_\sigma X=0$。由定义 7.7 得
$$
\partial_\tau\widetilde X=\partial_\sigma X,
$$
所以 Neumann 条件变为 $\partial_\tau\widetilde X=0$，即端点的 $\widetilde X$ 不随 $\tau$ 变化，这是 Dirichlet 条件。反向同理。$\square$

**定义 7.10（D-brane）.** D$p$-brane 是 target spacetime 中一个 $(p+1)$ 维世界体子流形 $W$，使开弦端点在 $TW$ 方向满足 Neumann 条件，在法向满足 Dirichlet 条件。

**推论 7.11（T-duality 改变 brane 维数）.** 若沿 D$p$-brane 的切向圆做 T-duality，则 Neumann 变为 Dirichlet，得到 D$(p-1)$-brane。若沿法向圆做 T-duality，则 Dirichlet 变为 Neumann，得到 D$(p+1)$-brane。

**证明.** 逐方向应用命题 7.9。$\square$

## 7.5 开弦端点、Chan-Paton factors 和 gauge fields

**定义 7.12（Chan-Paton labels）.** 对一组 $N$ 个重合 D-branes，开弦端点可携带标签 $i,j\in\{1,\ldots,N\}$。开弦态写为
$$
|\psi;i,j\rangle.
$$
这些标签在线性组合中形成 $N\times N$ 矩阵自由度。

**命题 7.13（重合 branes 上的 $U(N)$ gauge symmetry）.** $N$ 个重合 D-branes 上的开弦 massless vector states 组织为 $U(N)$ gauge field 的伴随表示。

**证明草图.** 单个 brane 的开弦 massless vector 已由命题 6.6 给出。加入 Chan-Paton labels 后，极化还带矩阵 $\lambda^a_{ij}$。Disk 振幅的边界 ordering 给出 Chan-Paton trace
$$
\operatorname{Tr}(\lambda^{a_1}\cdots \lambda^{a_n}),
$$
其代数闭合为 $\mathfrak u(N)$。低能三点与四点振幅匹配 Yang-Mills 相互作用。$\square$

**命题 7.14（分离 branes 间开弦的质量）.** 若开弦两端分别位于相距 $L$ 的平行 D-branes 上，则其质量公式变为
$$
M^2=\left(\frac{L}{2\pi\alpha'}\right)^2+\frac1{\alpha'}(N-a).
$$

**证明草图.** Dirichlet 方向的经典解含有线性项，使弦具有最小长度 $L$。其经典能量为 $L/(2\pi\alpha')$，对 $M^2$ 的贡献为 $L^2/(2\pi\alpha')^2$。振子部分仍给出通常的开弦项。$\square$

## 7.6 D-brane 的动力学接口

**外部输入定理 7.15（D-brane tension 与 RR charge）.** 在 type II superstring 中，BPS D$p$-brane 的物理张力为
$$
\tau_p=\frac1{(2\pi)^p g_s(\alpha')^{(p+1)/2}},
$$
并携带相应 RR charge。若 WZ coupling 使用 string-frame R-R potentials 的常用规范，则其裸系数为
$$
\mu_p=g_s\tau_p=\frac1{(2\pi)^p(\alpha')^{(p+1)/2}}.
$$
BPS tension-charge equality 应理解为转到 canonical RR fields 后的等式。该结论由 disk 振幅、open-closed duality 和低能 supergravity 解共同固定。

**注 7.16.** 本章只使用 D-brane 的边界条件定义。D-brane 的低能有效作用
$$
S_{\mathrm{DBI}}+S_{\mathrm{WZ}}
$$
在第十二章展开；其 RR charge、K-theory 分类和 anomaly inflow 不在本章证明。

## 7.7 多圆紧化和 Narain lattice

**定义 7.17（Narain lattice）.** 对 $T^d$ 紧化，momentum/winding quantum numbers 组合成 signature $(d,d)$ 的偶自对偶 lattice
$$
\Gamma^{d,d}.
$$
左、右动量 $(p_L,p_R)$ 是该 lattice 中向量，并依赖 metric $G$ 与 $B$-field moduli。

**定理 7.18（torus T-duality group）.** $T^d$ 紧化的 perturbative T-duality group 为
$$
O(d,d;\mathbb Z),
$$
它保持 Narain lattice pairing，并作用在 $G+B$ moduli 上。

**证明草图.** 闭弦谱由 lattice norm $p_L^2,p_R^2$、oscillator levels 和 level matching 决定。保持整数 momentum/winding lattice 及其 bilinear form 的变换构成 $O(d,d;\mathbb Z)$；这些变换给出等价 CFT。完整证明属于 Narain compactification 的标准 CFT 构造。$\square$

## 7.8 Orbifold 与 orientifold 接口

**定义 7.19（orbifold）.** 若 target CFT 有离散对称群 $G$，orbifold CFT 形式上由投影到 $G$-invariant states 并加入 twisted sectors 构成。

**命题 7.20（twisted sectors 的必要性）.** 闭弦 orbifold 中，若只投影 untwisted Hilbert space 而不加入 twisted sectors，则一般不能得到 modular invariant partition function。

**证明草图.** Torus partition function 需对沿两个基本 cycles 的 twists 求和。Modular $S$ transformation 会交换 temporal 与 spatial twists；因此只保留 untwisted sector 不在 modular group 下闭合。$\square$

**定义 7.21（orientifold）.** Orientifold 是把 worldsheet orientation reversal 与 target-space involution 组合后取商的构造。它引入 unoriented strings 和 orientifold planes，常用于 tadpole cancellation 与构造 type I/string compactifications。

## 本章小结

圆紧化闭弦具有 momentum 与 winding 两类量子数，质量谱在 $R\leftrightarrow\alpha'/R$ 下不变。对开弦，T-duality 把边界条件互换，使 D-branes 从世界面边界变分中必然出现。Chan-Paton factors 解释了 brane stack 上的 gauge fields，并为后续 D-brane effective theory 提供入口。

## 练习

**练习 7.1.** 证明 T-duality 下 $p_L$ 不变而 $p_R$ 变号。

**练习 7.2.** 从 $L_0=\tilde L_0=a$ 推导圆紧化闭弦的 level matching 条件。

**练习 7.3.** 用 dual coordinate 的定义证明 Neumann 条件变为 Dirichlet 条件。

**练习 7.4.** 说明为什么 orbifold 闭弦理论需要 twisted sectors。
