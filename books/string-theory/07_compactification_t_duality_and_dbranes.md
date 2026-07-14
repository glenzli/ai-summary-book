# 第七章：紧化、T-duality 和 D-branes

把一个 target 方向卷成半径 $R$ 的圆后，点粒子只记得离散 momentum，闭弦却还能
绕圆 $w$ 次。两种整数共同进入 $p_L,p_R$，使半径 $R$ 与 $\alpha'/R$ 的质量谱
完全相同；这不是低能近似，而是紧化 free-boson CFT 的等价。更意外的是，同一变换
对开弦会交换 Neumann 与 Dirichlet 边界条件，使端点被限制在 target 子流形上。
以下从第二章的边界变分、第四章的模展开和质量公式出发，先计算圆紧化谱，再沿
T-duality 追踪边界条件、Chan--Paton labels 与 D-branes；第六章的顶点算子语言
用于辨认对应的时空态。

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

**定理 7.6（闭弦谱的 T-duality）.** Free compact boson 的闭弦质量谱与
level matching 条件在变换
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

**定义 7.7（on-shell dual coordinate）.** T-dual coordinate $\widetilde X$ 在平坦
圆紧化的 classical solution space 上局部由
$$
\partial_\tau \widetilde X=\partial_\sigma X,\qquad
\partial_\sigma \widetilde X=\partial_\tau X
$$
定义，等价地在左右移动分解中保持 $X_L$ 而改变 $X_R$ 的符号。$\widetilde X$
的局部可积条件是 $X$ 的波动方程；其全局周期由 momentum/winding 量子数决定。
因此这不是任意 off-shell map $X\mapsto\widetilde X$ 的逐点场重定义。

**外部输入定理 7.8（compact-boson CFT 等价）.** 在选定相容 vertex-operator
cocycles 后，半径 $R$ 与 $\alpha'/R$ 的 unitary compact free-boson CFT 同构。
同构把 $(n,w)$ 送到 $(w,n)$，保持左移场并反转右移场，因而保持 stress tensor、
OPE、operator products、全部 genus-zero correlators 与 torus partition function。

**证明路线（外部输入）.** Hilbert-space trace 的 oscillator/lattice 部分为
$$
Z_R(\tau,\bar\tau)
=\frac1{|\eta(\tau)|^2}
\sum_{n,w\in\mathbb Z}
q^{\alpha'p_L^2/4}\bar q^{\alpha'p_R^2/4}.
$$
定理 7.6 的变量替换逐项证明 $Z_R=Z_{\alpha'/R}$，而
$X_R\mapsto-X_R$ 保持 free stress tensor。把这个谱映射提升为完整 operator
algebra 同构还必须验证 vertex cocycle 与 mutual locality；本书引用该标准 CFT
结果，不用 partition-function 相等冒充完整证明。

**推导说明 7.8A（Buscher 路径积分输入）.** 对 period $2\pi$、
$G_{yy}=R^2$ 的 isometric coordinate，在 Euclidean sigma-model path integral 中
gauging shift isometry、加入 Lagrange multiplier，再 Gaussian 积掉 gauge field，给出
$$
\widetilde R=\frac{\alpha'}R,
\qquad
\widetilde\Phi=\Phi-\frac12\log\frac{R^2}{\alpha'}.
$$
第一式的 classical part 与定理 7.6 一致；第二式来自 Gaussian determinant 的
Weyl anomaly，依赖 heat-kernel/zeta regulator 的标准 Buscher 推导。该步骤是
gauge-fixed 路径积分输入，不是定义 7.7 的 classical 等式。

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

**定义 7.10（几何 D-brane 的领先阶模型）.** 在弱曲率、弱耦合且存在几何
sigma-model 描述时，D$p$-brane 的领先阶数据包含 target spacetime 中一个
$(p+1)$ 维世界体子流形 $W$ 及其 Chan--Paton bundle，使开弦端点在 $TW$ 方向满足
Neumann 条件，在法向满足 Dirichlet 条件。一般 exact D-brane 是相容 boundary CFT
的边界条件；奇异背景、非几何相位或导出/K-theoretic 描述不由一个光滑子流形穷尽。

**推论 7.11（T-duality 改变 brane 维数）.** 若沿 D$p$-brane 的切向圆做 T-duality，则 Neumann 变为 Dirichlet，得到 D$(p-1)$-brane。若沿法向圆做 T-duality，则 Dirichlet 变为 Neumann，得到 D$(p+1)$-brane。

**证明.** 逐方向应用命题 7.9。$\square$

## 7.5 开弦端点、Chan-Paton factors 和 gauge fields

**定义 7.12（Chan-Paton labels）.** 对一组 $N$ 个重合 D-branes，开弦端点可携带标签 $i,j\in\{1,\ldots,N\}$。开弦态写为
$$
|\psi;i,j\rangle.
$$
这些标签在线性组合中形成 $N\times N$ 矩阵自由度。

**命题 7.13（重合 branes 上的 $U(N)$ gauge symmetry）.** 对 oriented type-II
open strings，$N$ 个重合 D-branes 上的开弦 massless vector states 组织为 $U(N)$
gauge field 的伴随表示。加入 orientifold projection 时可改为 orthogonal 或
symplectic groups，本命题不覆盖该情形。

**推导说明（标准物理口径）.** 单个 brane 的开弦 massless vector 已由命题 6.6 给出。加入 Chan-Paton labels 后，极化还带矩阵 $\lambda^a_{ij}$。Disk 振幅的边界 ordering 给出 Chan-Paton trace
$$
\operatorname{Tr}(\lambda^{a_1}\cdots \lambda^{a_n}),
$$
其代数闭合为 $\mathfrak u(N)$。低能三点与四点振幅匹配 Yang-Mills 相互作用。$\square$

**命题 7.14（分离 branes 间开弦的质量）.** 若开弦两端分别位于相距 $L$ 的平行 D-branes 上，则其质量公式变为
$$
M^2=\left(\frac{L}{2\pi\alpha'}\right)^2+\frac1{\alpha'}(N-a).
$$

**证明.** 在一个平坦 Dirichlet 方向取端点
$X(0)=y_0$、$X(\pi)=y_\pi$，并在 chosen lift 上令 $L=y_\pi-y_0$。满足边界条件的
模展开以
$$
X_{\mathrm{cl}}(\sigma)=y_0+\frac L\pi\sigma
$$
为线性部分，其余振子是 $\sin n\sigma$ modes。Gauge-fixed worldsheet Hamiltonian
中线性部分贡献
$$
\frac1{4\pi\alpha'}\int_0^\pi d\sigma\,(X'_{\mathrm{cl}})^2
=\frac{L^2}{4\pi^2\alpha'}.
$$
因此
$$
L_0=\alpha'k_\parallel^2
+\frac{L^2}{4\pi^2\alpha'}+N.
$$
代入 $(L_0-a)|\psi\rangle=0$ 与 $M^2=-k_\parallel^2$，得到所列质量公式。
该计算在 flat parallel branes 的 free boundary CFT 中精确；弯曲、背景场与 string
loops 会修正其适用模型。$\square$

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

**外部输入定理 7.18（Narain CFT 的 T-duality group）.** 对常数 $G,B$、无额外
orbifold 的 toroidal free-boson CFT，作用在 Narain moduli 上的 perturbative
T-duality identification group 为
$$
O(d,d;\mathbb Z),
$$
它保持 Narain lattice pairing，并作用在 $G+B$ moduli 上。

**证明路线（外部输入）.** 闭弦谱由 lattice norm $p_L^2,p_R^2$、oscillator levels
和 level matching 决定。保持整数 momentum/winding lattice 及其 bilinear form 的
变换构成 $O(d,d;\mathbb Z)$。从谱保持提升到含 cocycle 的 operator algebra 等价，
属于 Narain compactification 的标准 CFT 构造；本书不以谱论证替代该证明。

## 7.8 Orbifold 与 orientifold 接口

**定义 7.19（orbifold）.** 若 target CFT 有离散对称群 $G$，orbifold CFT 形式上由投影到 $G$-invariant states 并加入 twisted sectors 构成。

**命题 7.20（twisted sectors 的必要性）.** 闭弦 orbifold 中，若只投影 untwisted Hilbert space 而不加入 twisted sectors，则一般不能得到 modular invariant partition function。

**推导说明（标准物理口径）.** Torus partition function 需对沿两个基本 cycles 的 twists 求和。Modular $S$ transformation 会交换 temporal 与 spatial twists；因此只保留 untwisted sector 不在 modular group 下闭合。$\square$

**定义 7.21（orientifold）.** Orientifold 是把 worldsheet orientation reversal 与 target-space involution 组合后取商的构造。它引入 unoriented strings 和 orientifold planes，常用于 tadpole cancellation 与构造 type I/string compactifications。

圆紧化因而同时改变了谱与几何语言：momentum/winding lattice 在
$R\leftrightarrow\alpha'/R$ 下保持 CFT，开弦的 dual coordinate 则把自由端点
变成固定端点。D-brane 不是额外塞入的时空物体，而是允许的边界条件在 T-duality
下的必然像；一叠 branes 上的 Chan--Paton 矩阵又预示非阿贝尔 gauge fields。
Orbifold 必须补入 twisted sectors 的同一 modular 逻辑，将在更一般紧化中反复出现。

## 练习

**练习 7.1.** 证明 T-duality 下 $p_L$ 不变而 $p_R$ 变号。

**练习 7.2.** 从 $L_0=\tilde L_0=a$ 推导圆紧化闭弦的 level matching 条件。

**练习 7.3.** 用 dual coordinate 的定义证明 Neumann 条件变为 Dirichlet 条件。

**练习 7.4.** 说明为什么 orbifold 闭弦理论需要 twisted sectors。
