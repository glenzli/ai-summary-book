# 第八章 Einstein 方程

## 8.1 方程应满足什么

广义相对论的场方程要把几何和物质联系起来。几何侧应由度规及其至多二阶导数组成，并满足协变守恒；物质侧应是能动张量 $T_{\mu\nu}$。最简单候选是 Einstein 张量

$$
G_{\mu\nu}=R_{\mu\nu}-\frac12Rg_{\mu\nu}.
$$

第二 Bianchi 恒等式给出

$$
\nabla^\mu G_{\mu\nu}=0.
$$

因此若场方程取

$$
G_{\mu\nu}=8\pi G T_{\mu\nu},
$$

就自动要求

$$
\nabla^\mu T_{\mu\nu}=0.
$$

加入宇宙学常数后为

$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}.
$$

## 8.2 Einstein-Hilbert 作用量

几何作用量为

$$
S_{\mathrm{EH}}
=\frac{1}{16\pi G}\int (R-2\Lambda)\sqrt{-g}\,d^4x.
$$

总作用量

$$
S=S_{\mathrm{EH}}+S_{\mathrm{matter}}.
$$

物质能动张量定义为

$$
T_{\mu\nu}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_{\mathrm{matter}}}{\delta g^{\mu\nu}}.
$$

对 $g^{\mu\nu}$ 变分，并忽略边界项，几何部分给出

$$
\delta S_{\mathrm{EH}}
=\frac{1}{16\pi G}
\int
(G_{\mu\nu}+\Lambda g_{\mu\nu})
\delta g^{\mu\nu}\sqrt{-g}\,d^4x.
$$

物质部分为

$$
\delta S_{\mathrm{matter}}
=-\frac12
\int T_{\mu\nu}\delta g^{\mu\nu}\sqrt{-g}\,d^4x.
$$

令总变分为零，得到 Einstein 方程。

严格处理边界项需要加入 Gibbons-Hawking-York 边界项。本书主线只需局部场方程，边界变分细节作为高级补充。

## 8.3 Newton 极限

在弱场慢速极限中取

$$
g_{tt}=-(1+2\Phi),
$$

且物质主要由静止质量密度给出：

$$
T_{tt}\approx \rho.
$$

线性化计算给出

$$
G_{tt}\approx 2\nabla^2\Phi.
$$

Einstein 方程 $G_{tt}=8\pi G\rho$ 化为

$$
\nabla^2\Phi=4\pi G\rho.
$$

这正是 Newton 引力的 Poisson 方程。系数 $8\pi G$ 由要求 Newton 极限正确固定。

## 8.4 真空方程

无物质且 $\Lambda=0$ 时，

$$
G_{\mu\nu}=0.
$$

取迹：

$$
g^{\mu\nu}G_{\mu\nu}=R-2R=-R=0.
$$

故真空方程等价于

$$
R_{\mu\nu}=0.
$$

注意 $R_{\mu\nu}=0$ 不意味着 Riemann 张量为零。Schwarzschild 外部是真空，但仍有非零曲率和潮汐力。

## 8.5 守恒律的含义

协变守恒

$$
\nabla_\mu T^{\mu\nu}=0
$$

不是普通意义上的全局能量守恒。一般弯曲时空没有全局时间平移对称性，因此没有自然定义的全局能量。

在存在 Killing 矢量 $\xi^\mu$ 时，可构造守恒流

$$
J^\mu=T^{\mu\nu}\xi_\nu.
$$

因为

$$
\nabla_\mu J^\mu
=
(\nabla_\mu T^{\mu\nu})\xi_\nu
+T^{\mu\nu}\nabla_\mu\xi_\nu=0,
$$

其中第二项因 $T^{\mu\nu}$ 对称和 Killing 方程 $\nabla_{(\mu}\xi_{\nu)}=0$ 消失。

## 8.6 宇宙学常数

含 $\Lambda$ 的方程可写为

$$
G_{\mu\nu}=8\pi G(T_{\mu\nu}+T^{(\Lambda)}_{\mu\nu}),
$$

其中

$$
T^{(\Lambda)}_{\mu\nu}
=-\frac{\Lambda}{8\pi G}g_{\mu\nu}.
$$

这等价于真空能量密度

$$
\rho_\Lambda=\frac{\Lambda}{8\pi G},
\qquad
p_\Lambda=-\rho_\Lambda.
$$

## 习题

1. 证明 $g^{\mu\nu}G_{\mu\nu}=-R$。
2. 解释为什么 $R_{\mu\nu}=0$ 不等于平直。
3. 从 Killing 方程证明 $J^\mu=T^{\mu\nu}\xi_\nu$ 守恒。
4. 推导 $\Lambda$ 对应的状态方程 $p=-\rho$。
5. 说明 Newton 极限如何固定 Einstein 方程右侧系数。
