# 附录 A：微分几何、纤维丛和曲率约定

## 目标

本附录固定 differential geometry 记号：forms、connections、curvature、pullback 和 characteristic classes。

## A.1 微分形式

**定义 A.1.** $p$-form 是 $\Omega^p(M)=\Gamma(\wedge^pT^*M)$ 的元素。Exterior derivative 满足
$$
d^2=0,\qquad d(\alpha\wedge\beta)=d\alpha\wedge\beta+(-1)^p\alpha\wedge d\beta.
$$

若 $f:N\to M$ 是 smooth map，则 pullback 与 exterior derivative 交换：
$$
f^*(d\omega)=d(f^*\omega).
$$
这一本性用于把 target-space form fields 拉回到 string worldsheet 或 D-brane worldvolume。

**定义 A.1A（Hodge star）.** 在带 metric $g$ 与 orientation 的 $n$ 维流形上，Hodge star
$$
*: \Omega^p(M)\longrightarrow \Omega^{n-p}(M)
$$
由
$$
\alpha\wedge *\beta=(\alpha,\beta)_g\,\mathrm{vol}_g
$$
定义。Lorentzian signature 下 $*^2$ 的符号依赖 $p$ 与 time directions；本书在具体使用时以正文 convention 为准。

**命题 A.1B（Stokes theorem）.** 若 $M$ 是 compact oriented manifold with boundary，则
$$
\int_M d\omega=\int_{\partial M}\omega.
$$
Worldsheet 边界条件、Wess-Zumino coupling 的 gauge invariance 和 anomaly descent 都反复使用该公式。

## A.2 曲率

**定义 A.2.** Connection one-form $A$ 的 curvature 为
$$
F=dA+A\wedge A.
$$

**命题 A.3.** Bianchi identity 为
$$
d_AF=0.
$$

**证明.** 直接计算 $dF+[A,F]$ 并使用 graded Jacobi identity。$\square$

## A.3 Characteristic classes

**定义 A.4（Chern character）.** 对 complex vector bundle $E$ with curvature $F$，Chern character 为
$$
\operatorname{ch}(E)=\operatorname{Tr}\exp\left(\frac{iF}{2\pi}\right).
$$
低阶项为
$$
\operatorname{ch}(E)=\operatorname{rank}(E)+c_1(E)+\frac12(c_1(E)^2-2c_2(E))+\cdots.
$$

**定义 A.5（Pontryagin class）.** 对 real vector bundle $V$，Pontryagin classes 可由复化 bundle 的 Chern classes 定义：
$$
p_k(V)=(-1)^k c_{2k}(V\otimes_{\mathbb R}\mathbb C).
$$
在 anomaly polynomial 和 D-brane charge formula 中，Pontryagin classes 通常以 formal roots 的方式进入 $\widehat A$ genus。

**定义 A.6（$\widehat A$ genus）.** 若 $x_i$ 是 curvature formal roots，则
$$
\widehat A(R)=\prod_i\frac{x_i/2}{\sinh(x_i/2)}
=1-\frac1{24}p_1+\cdots.
$$
D-brane WZ coupling 中常出现
$$
\sqrt{\frac{\widehat A(TW)}{\widehat A(NW)}}
$$
其中 $TW$ 与 $NW$ 分别是 brane worldvolume 的 tangent 与 normal bundles。

## A.4 Forms in string theory

String theory 中最常用的 differential-form 规则如下：

1. NS-NS field strength：$H_3=dB_2$，在 heterotic theory 中会被 Chern-Simons terms 修正。
2. R-R field strengths：形式上写为 polyform $F=\sum_p F_p$，受 Bianchi identity、duality condition 和 source term 约束。
3. D-brane WZ coupling：若 $W$ 是 worldvolume，则
   $$
   S_{WZ}=\mu_p\int_W C\wedge e^{2\pi\alpha' F+B}\wedge
   \sqrt{\frac{\widehat A(TW)}{\widehat A(NW)}}.
   $$
4. Flux quantization：在适当归一化下，flux periods 落在 integral cohomology 或其 shifted variant 中。

这些公式在正文只按主线需要使用；完整证明属于 differential geometry、index theory 和 generalized cohomology。
