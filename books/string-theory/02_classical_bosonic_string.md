# 第二章：经典玻色弦、Nambu-Goto 与 Polyakov 作用量

点粒子的历史是一条曲线，弦的历史却是一张曲面；把固有长度直接换成世界面面积，
会得到几何上自然但量子化并不方便的 Nambu--Goto 根式。引入独立世界面 metric 后，
Polyakov 作用量把同一经典动力学改写成带 diffeomorphism 与 Weyl 冗余的二维场论，
其 metric 方程正是 Virasoro constraints。以下从面积泛函、变分和边界项逐步验证这层
等价，再在 conformal gauge 中求出开弦与闭弦的经典模展开。计算沿用第一章的作用量
与 stress tensor 语言；除特别说明外，target 取平坦 mostly-plus metric
$\eta_{\mu\nu}$，其余归一化遵循 [全书归一化表](NORMALIZATION_TABLE.md)。

## 2.1 世界面和诱导度量

**定义 2.1（string worldsheet）.** 一条 classical string 在 target spacetime $M$ 中的历史是映射
$$
X:\Sigma\to M.
$$
若 $\sigma^a=(\tau,\sigma)$ 是世界面坐标，则诱导度量为
$$
\gamma_{ab}=\partial_aX^\mu\partial_bX^\nu g_{\mu\nu}(X).
$$

**定义 2.2（Nambu-Goto action）.** Nambu-Goto 作用量为
$$
S_{NG}=-T\int_\Sigma d^2\sigma\sqrt{-\det\gamma_{ab}},
\qquad
T=\frac1{2\pi\alpha'}.
$$

**命题 2.3（面积泛函与重参数化不变性）.** $S_{NG}$ 等于弦世界面 Lorentzian 面积乘以 $-T$，并在世界面重参数化下不变。

**证明.** $\sqrt{-\det\gamma}\,d^2\sigma$ 是诱导 Lorentzian 面积元。坐标变换 $\sigma\mapsto\sigma'$ 下，$\det\gamma$ 与 Jacobian 的平方反向变换，故面积元不变。$\square$

**注 2.4.** $S_{NG}$ 几何直观清楚，但 square root 使量子化不便。Polyakov 作用量引入独立世界面 metric，使 gauge symmetry 和 CFT 结构显式化。

## 2.2 Polyakov 作用量和对称性

**定义 2.5A（Polyakov action）.** 令 $\Sigma$ 为二维 Lorentzian
世界面，$h$ 为其非退化 Lorentzian metric，$X:\Sigma\to\mathbb R^{1,D-1}$
至少二次可微。Lorentzian Polyakov 作用量为
$$
S_P=-\frac{1}{4\pi\alpha'}\int d^2\sigma\sqrt{-h}\,
h^{ab}\partial_aX^\mu\partial_bX_\mu.
$$
这里 $h_{ab}$ 是独立 worldsheet metric。

**命题 2.5（Polyakov 与 Nambu-Goto 的经典等价）.** 在 classical level，消去 $h_{ab}$ 后 Polyakov 作用量等价于 Nambu-Goto 作用量。

**证明.** 先假设诱导度量 $\gamma_{ab}$ 非退化。利用
$$
\delta\sqrt{-h}=-\frac12\sqrt{-h}\,h_{ab}\delta h^{ab},
$$
对 $h^{ab}$ 变分得到
$$
0=\gamma_{ab}-\frac12h_{ab}h^{cd}\gamma_{cd}.
$$
令 $A^a{}_b=h^{ac}\gamma_{cb}$。上式等价于
$$
A=\frac12\operatorname{tr}(A)\operatorname{id}_{T\Sigma}.
$$
因此 $A$ 是处处非零的标量矩阵，亦即 $\gamma_{ab}=\lambda h_{ab}$。在保持 Lorentzian signature 的连通分支上可写成
$$
h_{ab}=e^{2\omega}\gamma_{ab}.
$$
代回作用量并使用 $\gamma^{ab}\gamma_{ab}=2$，得到
$$
S_P=-\frac1{4\pi\alpha'}\int d^2\sigma\,
\sqrt{-\gamma}\,\gamma^{ab}\gamma_{ab}
=-\frac1{2\pi\alpha'}\int d^2\sigma\sqrt{-\gamma}=S_{NG}.
$$
Weyl 因子在第二个等号前已经约去。退化世界面不在这一步消元所描述的经典正则构形空间内。$\square$

**命题 2.6（方程和约束）.** 在平坦 target 中，固定边界值或施加使边界
变分消失的边界条件后，Polyakov 作用量的 $X^\mu$ 变分给出
$$
\frac1{\sqrt{-h}}\partial_a(\sqrt{-h}h^{ab}\partial_bX^\mu)=0.
$$
定义未作 gauge fixing 的经典 stress tensor 为
$$
T_{ab}:=-\frac{4\pi}{\sqrt{-h}}\frac{\delta S_P}{\delta h^{ab}}
=\frac1{\alpha'}\left(
\partial_aX\cdot\partial_bX
-\frac12h_{ab}h^{cd}\partial_cX\cdot\partial_dX
\right).
$$
对 $h^{ab}$ 的变分给出精确的经典 Euler--Lagrange 约束
$$
T_{ab}=0.
$$

**证明.** $X^\mu$ 变分为
$$
\delta_XS_P
=\frac1{2\pi\alpha'}\int d^2\sigma\,
\partial_a(\sqrt{-h}h^{ab}\partial_bX_\mu)\delta X^\mu
+\delta S_{\partial\Sigma}.
$$
bulk 项给出 Laplace-Beltrami 方程。另一方面，直接使用
$\delta\sqrt{-h}=-(1/2)\sqrt{-h}h_{ab}\delta h^{ab}$，得到
$$
\delta_hS_P=-\frac1{4\pi}\int d^2\sigma\sqrt{-h}\,
T_{ab}\delta h^{ab},
$$
故任意紧支撑的 $\delta h^{ab}$ 给出 $T_{ab}=0$。这里尚未固定 gauge；若先把
$h$ 代成平坦 metric 再变分，就会错误地丢失这一约束。$\square$

**命题 2.7（Polyakov action 的局部对称性）.** $S_P$ 具有：

1. worldsheet diffeomorphism invariance；
2. Weyl invariance $h_{ab}\mapsto e^{2\omega}h_{ab}$；
3. 平坦 target 中的 Poincare invariance。

**证明.** 在世界面坐标变换下，$X^\mu$ 是标量，$h_{ab}$ 是二阶协变张量，而 $d^2\sigma\sqrt{-h}$ 是不变体积元，故 integrand 的全收缩给出微分同胚不变量。Weyl 变换
$$
h_{ab}\mapsto e^{2\omega}h_{ab}
$$
使二维中的 $\sqrt{-h}$ 乘以 $e^{2\omega}$、$h^{ab}$ 乘以 $e^{-2\omega}$，两因子抵消。最后，target-space 平移不改变 $\partial_aX^\mu$，Lorentz 变换保持 $\eta_{\mu\nu}$，故 Poincare 变换也保持作用量。$\square$

## 2.3 Conformal gauge 和 Virasoro constraints

**定义 2.8（conformal gauge）.** Conformal gauge 是在二维局部坐标片上，利用
diffeomorphism 与 Weyl symmetry 取
$$
h_{ab}=e^{2\omega}\eta_{ab}.
$$
全局上仍可能留下 complex-structure moduli；因此这是局部 gauge-fixed 公式，
不是任意拓扑世界面上的全局 metric 恒等式。在该 gauge 下运动方程变为
$$
(\partial_\tau^2-\partial_\sigma^2)X^\mu=0.
$$

**命题 2.9（Virasoro constraints）.** Conformal gauge 下的约束为
$$
(\partial_\tau X\pm\partial_\sigma X)^2=0.
$$

**证明.** 取 light-cone coordinates
$$
\sigma^\pm=\tau\pm\sigma,\qquad
\partial_\pm=\frac12(\partial_\tau\pm\partial_\sigma).
$$
在 conformal gauge 中，命题 2.6 的未固定约束化为
$$
T_{++}=\frac1{\alpha'}\partial_+X\cdot\partial_+X,\qquad
T_{--}=\frac1{\alpha'}\partial_-X\cdot\partial_-X.
$$
约束 $T_{ab}=0$ 等价于 $T_{++}=T_{--}=0$，即
$$
(\partial_\tau X+\partial_\sigma X)^2=0,\qquad
(\partial_\tau X-\partial_\sigma X)^2=0.
$$
$\square$

**命题 2.9A（conformal gauge 中的正则约束）.** 令
$$
P_\mu=\frac1{2\pi\alpha'}\partial_\tau X_\mu
$$
为 gauge-fixed Lagrangian 的正则动量，并记 $X'=\partial_\sigma X$。则原 metric
方程在正则变量中等价于
$$
\mathcal H_\perp
=\pi\alpha'P^2+\frac1{4\pi\alpha'}(X')^2=0,
\qquad
\mathcal H_\parallel=P\cdot X'=0.
$$
它们是经典约束，不是未约束相空间上的恒等式。

**证明.** Conformal gauge 下
$$
\mathcal L=\frac1{4\pi\alpha'}(\dot X^2-(X')^2),
\qquad
P=\frac{\dot X}{2\pi\alpha'}.
$$
因此
$$
\mathcal H_\perp
=\frac1{4\pi\alpha'}(\dot X^2+(X')^2),
\qquad
\mathcal H_\parallel
=\frac1{2\pi\alpha'}\dot X\cdot X'.
$$
逐项相加、相减得到
$$
\mathcal H_\perp\pm\mathcal H_\parallel
=\frac1{4\pi\alpha'}(\dot X\pm X')^2
=\frac1\pi T_{\pm\pm}.
$$
故两条正则约束与 $T_{++}=T_{--}=0$ 互相推出。$\square$

**注 2.9B（经典、gauge-fixed 与量子陈述）.** $T_{ab}=0$ 是对独立 metric
变分得到的经典方程；$T_{\pm\pm}=0$ 是其 conformal-gauge 坐标表达。量子理论中
不能把所有正规序后的 $L_n$ 都当成零算符；第四章在公共定义域上施加正频物理态
条件，第五章再把约束改写成 BRST cohomology。Central term 正是这一步的量子输入。

**注 2.10（残余共形变换）.** Conformal gauge 固定后仍有残余变换
$$
\sigma^+\mapsto f(\sigma^+),\qquad
\sigma^-\mapsto g(\sigma^-).
$$
量子化后这些残余对称性由 Virasoro algebra 表示。

## 2.4 闭弦经典解

闭弦取
$$
\sigma\sim\sigma+2\pi.
$$
波动方程的一般解为左右移动之和：
$$
X^\mu(\tau,\sigma)=X_L^\mu(\tau+\sigma)+X_R^\mu(\tau-\sigma).
$$

**命题 2.11（闭弦模展开）.** 非紧平坦 target 中，闭弦解可写为
$$
X^\mu(\tau,\sigma)=x^\mu+\alpha'p^\mu\tau
+i\sqrt{\frac{\alpha'}2}\sum_{n\ne0}\frac1n
\left(
\alpha_n^\mu e^{-in(\tau-\sigma)}
+\tilde\alpha_n^\mu e^{-in(\tau+\sigma)}
\right).
$$

**证明.** 令 $u=\tau+\sigma$、$v=\tau-\sigma$。波动方程给出
$$
X^\mu=X_L^\mu(u)+X_R^\mu(v).
$$
非紧 target 中没有 winding；周期条件可取为 $\partial_uX_L$ 与 $\partial_vX_R$ 各自以 $2\pi$ 为周期，且两者零模相等。于是存在 Fourier 展开
$$
\partial_vX_R^\mu=\frac{\alpha'}2p^\mu
+\sqrt{\frac{\alpha'}2}\sum_{n\ne0}\alpha_n^\mu e^{-inv},
$$
$$
\partial_uX_L^\mu=\frac{\alpha'}2p^\mu
+\sqrt{\frac{\alpha'}2}\sum_{n\ne0}\tilde\alpha_n^\mu e^{-inu}.
$$
逐项积分，并把两个积分常数之和记为 $x^\mu$，便得到命题中的展开。其零模确为总动量，因为
$$
\int_0^{2\pi}d\sigma\,P^\mu
=\frac1{2\pi\alpha'}\int_0^{2\pi}d\sigma\,\partial_\tau X^\mu
=p^\mu;
$$
所有非零 Fourier modes 的积分均为零。$\square$

## 2.5 开弦边界条件

开弦取
$$
\sigma\in[0,\pi].
$$
Polyakov 作用量在 conformal gauge 的边界变分为
$$
\delta S_{\partial\Sigma}
=-\frac1{2\pi\alpha'}\int d\tau\,
\delta X_\mu\,\partial_\sigma X^\mu\bigg|_{\sigma=0}^{\sigma=\pi}.
$$

**定义 2.12（Neumann 与 Dirichlet 条件）.** 开弦端点可满足 Neumann boundary condition
$$
\partial_\sigma X^\mu|_{\partial\Sigma}=0,
$$
或 Dirichlet boundary condition
$$
\delta X^\mu|_{\partial\Sigma}=0.
$$

**命题 2.13（开弦 Neumann 模展开）.** 若所有方向均取 Neumann 条件，则开弦解为
$$
X^\mu(\tau,\sigma)=x^\mu+2\alpha'p^\mu\tau
+i\sqrt{2\alpha'}\sum_{n\ne0}\frac{\alpha_n^\mu}{n}
e^{-in\tau}\cos n\sigma.
$$

**证明.** 对固定 $\tau$，Neumann Laplacian $-\partial_\sigma^2$ 在 $[0,\pi]$ 上的完备本征函数为
$$
1,\quad \cos(n\sigma)\quad(n\ge1).
$$
将 $X^\mu$ 按此基展开并代入波动方程，常数空间模满足 $\ddot X_0^\mu=0$，第 $n$ 个非零模满足 $\ddot X_n^\mu+n^2X_n^\mu=0$。现实条件把正负频率系数联系为 $\alpha_{-n}^\mu=(\alpha_n^\mu)^*$，因而解具有命题所示形式。零模系数由总动量
$$
p^\mu=\int_0^\pi d\sigma\,P^\mu
$$
固定：由于 $P^\mu=(2\pi\alpha')^{-1}\partial_\tau X^\mu$，非零 cosine modes 的积分为零，而 $2\alpha'p^\mu\tau$ 恰贡献 $p^\mu$。振子前的 $\sqrt{2\alpha'}$ 是第四章正则对易关系所采用的标准归一化。$\square$

**注 2.14（D-brane 预告）.** Dirichlet 条件固定弦端点落在 target 中某个子流形上。量子理论中该子流形成为 D-brane 的几何模型。

面积描述与辅助度量描述因而给出同一套非退化经典世界面，但后者把冗余与约束分开
呈现。Conformal gauge 虽把 $X^\mu$ 的方程降为自由波动方程，却没有删除
$T_{++}=T_{--}=0$；这些约束决定哪些模是真正可传播的。开弦边界项又把 Neumann
与 Dirichlet 条件放在同一个变分问题中，为后面的 D-branes 留下了清楚入口。下一步
不是再次求解经典方程，而是理解这些二维场在量子理论中的短距离乘积。

## 练习

**练习 2.1.** 证明 Nambu-Goto 作用量在世界面重参数化下不变。

**练习 2.2.** 在 conformal gauge 下写出闭弦的一般经典解。

**练习 2.3.** 从开弦边界变分推出 Neumann 与 Dirichlet 条件。
