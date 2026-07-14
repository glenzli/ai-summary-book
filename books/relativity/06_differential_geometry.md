# 第六章 流形、张量、联络与曲率

在弯曲时空中，不同点的向量属于不同切空间，因而“直接相减两个向量”不再有坐标无关的意义。联络解决跨点比较，曲率则测量这种比较对路径的依赖；Christoffel 符号只是联络在坐标中的系数，不能与曲率本身混同。以下建立广义相对论所需的最小微分几何语言，并反复检验哪些量能在一点通过选坐标消去、哪些量仍携带几何信息。第零章的线性张量基础继续使用；光滑流形、切空间、向量场与 Lie 括号作为标准前置知识。时空固定为四维、时间定向的光滑 Lorentz 流形，号差为 $(-,+,+,+)$。

## 6.1 时空作为 Lorentz 流形

广义相对论把 Minkowski 时空推广为四维光滑流形 $M$，并在每一点的切空间 $T_pM$ 上给出 Lorentz 型内积。

**定义 6.1 (时空).** 一个相对论时空由四维光滑流形 $M$、号差
$(-,+,+,+)$ 的光滑 Lorentz 度规 $g$ 以及一个选定的时间定向组成；在
不致混淆时仍简记为 $(M,g)$。时间定向不是仅由度规自动选出的附加数据。

局部坐标 $x^\mu$ 下，

$$
g=g_{\mu\nu}(x)\,dx^\mu\otimes dx^\nu.
$$

弯曲时空的间隔为

$$
ds^2=g_{\mu\nu}dx^\mu dx^\nu.
$$

这里 $g_{\mu\nu}$ 可以随位置变化，因果分类仍由 $ds^2$ 的符号决定。

## 6.2 张量和坐标变换

坐标变换 $x^\mu\mapsto x'^\alpha$ 下，反变矢量和协变矢量分量满足

$$
V'^\alpha=\frac{\partial x'^\alpha}{\partial x^\mu}V^\mu,
\qquad
\omega'_\alpha=\frac{\partial x^\mu}{\partial x'^\alpha}\omega_\mu.
$$

$(r,s)$ 型张量按相应 Jacobian 变换。张量方程的意义是：如果它在一个坐标系成立，就在所有坐标系成立。

度规给出升降指标：

$$
V_\mu=g_{\mu\nu}V^\nu,\qquad
V^\mu=g^{\mu\nu}V_\nu.
$$

其中 $g^{\mu\nu}$ 是 $g_{\mu\nu}$ 的逆矩阵。

## 6.3 协变导数

普通偏导数 $\partial_\mu V^\nu$ 不是张量，因为坐标变换会产生二阶导数项。需要引入联络。

**定义 6.2（仿射联络）.** 流形 $M$ 上的仿射联络是映射
$$
\nabla:\mathfrak X(M)\times\mathfrak X(M)\to\mathfrak X(M),
\qquad (X,Y)\mapsto\nabla_XY,
$$
它对第一变量在 $C^\infty(M)$ 上线性、对第二变量在 $\mathbb R$ 上线性，并满足
$$
\nabla_X(fY)=X(f)Y+f\nabla_XY.
$$
在局部坐标基 $\partial_\mu$ 中定义联络系数
$$
\nabla_{\partial_\mu}\partial_\nu
=\Gamma^\rho{}_{\mu\nu}\partial_\rho.
$$
于是

$$
\nabla_\mu V^\rho=\partial_\mu V^\rho+\Gamma^\rho{}_{\mu\nu}V^\nu,
$$

对协变矢量

$$
\nabla_\mu \omega_\nu=\partial_\mu\omega_\nu-\Gamma^\rho{}_{\mu\nu}\omega_\rho.
$$

对一般张量，$\nabla$ 由 Leibniz 法则、与缩并相容及对函数满足 $\nabla_Xf=X(f)$ 唯一延拓。$\Gamma^\rho{}_{\mu\nu}$ 是联络在坐标基中的系数，并非张量分量。

**定义 6.2A（挠率与度规相容）.** 联络的挠率为
$$
T(X,Y)=\nabla_XY-\nabla_YX-[X,Y].
$$
称 $\nabla$ 与 $g$ 相容，若对任意向量场 $X,Y,Z$，
$$
X(g(Y,Z))=g(\nabla_XY,Z)+g(Y,\nabla_XZ).
$$
在坐标基中，无挠条件等价于 $\Gamma^\rho{}_{\mu\nu}=\Gamma^\rho{}_{\nu\mu}$，度规相容等价于 $\nabla_\rho g_{\mu\nu}=0$。

## 6.4 Levi-Civita 联络

广义相对论使用由度规唯一确定的无挠、度规相容联络。

**定理 6.1 (Levi-Civita 联络).** 给定度规 $g$，存在唯一联络满足

$$
\nabla_\rho g_{\mu\nu}=0,\qquad
\Gamma^\rho{}_{\mu\nu}=\Gamma^\rho{}_{\nu\mu}.
$$

其坐标表达为

$$
\Gamma^\rho{}_{\mu\nu}
=\frac12 g^{\rho\sigma}
(\partial_\mu g_{\nu\sigma}
+\partial_\nu g_{\mu\sigma}
-\partial_\sigma g_{\mu\nu}).
$$

**证明.** 对光滑向量场 $X,Y,Z$，考虑 Koszul 公式

$$
\begin{aligned}
2g(\nabla_XY,Z)
={}&Xg(Y,Z)+Yg(Z,X)-Zg(X,Y)\\
&-g(X,[Y,Z])+g(Y,[Z,X])+g(Z,[X,Y]).
\end{aligned}
$$

右侧对 $Z$ 是 $C^\infty(M)$-线性的。由 $g$ 的逐点非退化性，它唯一
确定一个向量场 $\nabla_XY$。直接把 $fX$、$fY$ 代入右侧并展开 Lie
括号，得到联络对第一变量的 $C^\infty(M)$-线性、对第二变量的
Leibniz 法则和实线性，所以该公式确实在全流形上定义一个联络。交换
$X,Y$ 后相减得到

$$
\nabla_XY-\nabla_YX=[X,Y],
$$

故挠率为零；分别写出 $g(\nabla_XY,Z)$ 与
$g(Y,\nabla_XZ)$ 的 Koszul 公式并相加，Lie 括号项抵消，得到

$$
Xg(Y,Z)=g(\nabla_XY,Z)+g(Y,\nabla_XZ),
$$

故联络与 $g$ 相容。这证明存在性。

反之，任何无挠且度规相容的联络，把后三个等式按循环排列后组合，都
必须满足同一 Koszul 公式；非退化性遂给出唯一性。在坐标基中
$[\partial_\mu,\partial_\nu]=0$，Koszul 公式化为

$$
2\Gamma^\sigma{}_{\mu\nu}g_{\sigma\rho}
=\partial_\mu g_{\nu\rho}
+\partial_\nu g_{\mu\rho}
-\partial_\rho g_{\mu\nu}.
$$

乘以 $g^{\rho\lambda}$ 即得正文的 Christoffel 公式。证毕。

## 6.5 平行移动和测地线

沿曲线 $x^\mu(\lambda)$ 的向量 $V^\mu(\lambda)$ 若满足

$$
\frac{dV^\mu}{d\lambda}
+\Gamma^\mu{}_{\rho\sigma}
\frac{dx^\rho}{d\lambda}V^\sigma=0,
$$

则称为沿曲线平行移动。

曲线切向量若沿自身平行移动，则得到测地线方程：

$$
\frac{d^2x^\mu}{d\lambda^2}
+\Gamma^\mu{}_{\rho\sigma}
\frac{dx^\rho}{d\lambda}
\frac{dx^\sigma}{d\lambda}=0.
$$

这将在下一章从作用量重新推导。

## 6.6 曲率

曲率测量平行移动绕小闭合回路后的失败。内禀地定义
$$
R(X,Y)Z
=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z.
$$
在坐标向量场上 $[\partial_\mu,\partial_\nu]=0$，故

$$
(\nabla_\mu\nabla_\nu-\nabla_\nu\nabla_\mu)V^\rho
=R^\rho{}_{\sigma\mu\nu}V^\sigma.
$$

坐标表达为

$$
R^\rho{}_{\sigma\mu\nu}
=
\partial_\mu\Gamma^\rho{}_{\nu\sigma}
-\partial_\nu\Gamma^\rho{}_{\mu\sigma}
+\Gamma^\rho{}_{\mu\lambda}\Gamma^\lambda{}_{\nu\sigma}
-\Gamma^\rho{}_{\nu\lambda}\Gamma^\lambda{}_{\mu\sigma}.
$$

Ricci 张量和标量曲率为

$$
R_{\mu\nu}=R^\rho{}_{\mu\rho\nu},
\qquad
R=g^{\mu\nu}R_{\mu\nu}.
$$

**命题 6.2B（曲率的张量性）.** $R(X,Y)Z$ 对 $X,Y,Z$ 的每个变量均为 $C^\infty(M)$-线性，因此定义一个 $(1,3)$ 型张量。

**证明.** 对第一变量，利用 $[fX,Y]=f[X,Y]-Y(f)X$，有
$$
\begin{aligned}
R(fX,Y)Z
&=f\nabla_X\nabla_YZ-\nabla_Y(f\nabla_XZ)
-\nabla_{f[X,Y]-Y(f)X}Z\\
&=fR(X,Y)Z;
\end{aligned}
$$
两个含 $Y(f)\nabla_XZ$ 的项抵消。第二变量由反对称性 $R(X,Y)=-R(Y,X)$ 得到。第三变量中，对 $R(X,Y)(fZ)$ 展开 Leibniz 法则；含一阶和二阶导数的项与 $[X,Y](f)Z$ 精确抵消，只剩 $fR(X,Y)Z$。证毕。

Riemann 张量满足基本对称性：

$$
R_{\rho\sigma\mu\nu}=-R_{\sigma\rho\mu\nu}
=-R_{\rho\sigma\nu\mu},
\qquad
R_{\rho\sigma\mu\nu}=R_{\mu\nu\rho\sigma}.
$$

还有第一 Bianchi 恒等式

$$
R_{\rho[\sigma\mu\nu]}=0.
$$

**命题 6.3（Levi-Civita 曲率的代数对称性）.** 对 Levi-Civita 联络，降下第一指标后的曲率满足上述两两反对称、pair exchange symmetry 与第一 Bianchi 恒等式。

**证明.** 记
$$
R(X,Y,Z,W)=g(R(X,Y)Z,W).
$$
无挠条件把 $R(X,Y)Z+R(Y,Z)X+R(Z,X)Y$ 展开为 Jacobi 恒等式，得到第一 Bianchi 恒等式。度规相容给出
$$
g(R(X,Y)Z,W)=-g(Z,R(X,Y)W),
$$
从而得到末两个指标的反对称性；$R(X,Y)=-R(Y,X)$ 给出首对指标的反对称性。将第一 Bianchi 恒等式分别与这些反对称性组合，可得
$$
R(X,Y,Z,W)=R(Z,W,X,Y).
$$
这些是坐标无关恒等式，写成分量即为正文公式。证毕。

## 6.7 正规坐标

**外部输入定理 6.4（Lorentz 正规坐标）.** 对任意 $p\in M$，存在以 $p$ 为中心的局部坐标使

$$
g_{\mu\nu}(p)=\eta_{\mu\nu},
\qquad
\partial_\rho g_{\mu\nu}(p)=0.
$$

于是

$$
\Gamma^\rho{}_{\mu\nu}(p)=0.
$$

但一般不能让二阶导数同时消失；曲率正是无法在一点邻域内完全消去的量。

**使用边界.** 正规坐标的存在由指数映射和局部逆函数定理证明，本书把该构造作为微分几何基础输入。该定理只在单点消去一阶度规导数；若曲率在 $p$ 非零，不存在坐标使度规在 $p$ 的整个二阶 jet 与 Minkowski 度规相同。

## 6.8 局部平直与不可消去的曲率

联络是作用于向量场的微分算子，Christoffel 符号只是其坐标系数。度规唯一确定无挠、度规相容的 Levi-Civita 联络；曲率由协变导数交换子定义并且是真正张量。正规坐标只能在一点消去联络系数，不能消去曲率。

## 习题

1. 对二维球面度规 $ds^2=d\theta^2+\sin^2\theta\,d\phi^2$ 计算非零 Christoffel 符号。
2. 验证 Levi-Civita 公式满足 $\nabla_\rho g_{\mu\nu}=0$。
3. 证明标量场 $\phi$ 的二阶协变导数满足 $\nabla_\mu\nabla_\nu\phi=\nabla_\nu\nabla_\mu\phi$。
4. 解释为什么 $\Gamma^\rho{}_{\mu\nu}$ 不是张量。
5. 在平直 Minkowski 坐标中计算 Riemann 张量。
