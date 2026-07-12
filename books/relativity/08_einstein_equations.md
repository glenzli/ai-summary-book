# 第八章 Einstein 方程

## 本章目标

本章从 Einstein-Hilbert 作用量推导场方程，说明协变守恒的几何来源，并在完整声明弱场、静态和规范假设后恢复 Newton 的 Poisson 方程。

## 依赖前置知识

需要第五章的变分法和能动张量、第六章的 Levi-Civita 联络与曲率，以及第七章的弱场测地线极限。

## 8.1 方程应满足什么

广义相对论的场方程要把几何和物质联系起来。几何侧应由度规及其至多二阶导数组成，并满足协变守恒；物质侧应是能动张量 $T_{\mu\nu}$。最简单候选是 Einstein 张量

$$
G_{\mu\nu}=R_{\mu\nu}-\frac12Rg_{\mu\nu}.
$$

缩并的第二 Bianchi 恒等式给出

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

**命题 8.1（Einstein-Hilbert 的一阶变分）.** 设 $U\subset M$ 是相对紧坐标区域，度规变分 $\delta g^{\mu\nu}$ 光滑且紧支撑于 $U$。则

$$
\delta S_{\mathrm{EH}}
=\frac{1}{16\pi G}
\int
(G_{\mu\nu}+\Lambda g_{\mu\nu})
\delta g^{\mu\nu}\sqrt{-g}\,d^4x.
$$

若允许变分在非空边界上不消失，则必须加入 Gibbons-Hawking-York 项并固定诱导边界度规，或另行指定能消去法向导数项的边界条件。

**证明.** 逆矩阵和行列式变分给出
$$
\delta g_{\mu\nu}=-g_{\mu\rho}g_{\nu\sigma}\delta g^{\rho\sigma},
\qquad
\delta\sqrt{-g}=-\frac12\sqrt{-g}\,g_{\mu\nu}\delta g^{\mu\nu}.
$$
又由 Palatini 恒等式
$$
\delta R_{\mu\nu}
=\nabla_\rho\delta\Gamma^\rho{}_{\mu\nu}
-\nabla_\nu\delta\Gamma^\rho{}_{\mu\rho}
$$
得到
$$
\delta R=R_{\mu\nu}\delta g^{\mu\nu}+\nabla_\rho V^\rho
$$
对某个由 $g$ 与 $\nabla\delta g$ 线性构成的向量场 $V^\rho$ 成立。因此
$$
\delta(\sqrt{-g}R)
=\sqrt{-g}\,G_{\mu\nu}\delta g^{\mu\nu}
+\sqrt{-g}\,\nabla_\rho V^\rho.
$$
最后一项是边界散度；紧支撑假设使其积分为零。宇宙学常数项由 $\delta\sqrt{-g}$ 贡献 $\Lambda g_{\mu\nu}\delta g^{\mu\nu}$，合并即得命题公式。证毕。

物质部分为

$$
\delta S_{\mathrm{matter}}
=-\frac12
\int T_{\mu\nu}\delta g^{\mu\nu}\sqrt{-g}\,d^4x.
$$

**推论 8.2（Einstein 方程）.** 若物质作用量的度规变分按上式定义 $T_{\mu\nu}$，并且总作用量对任意紧支撑对称变分 $\delta g^{\mu\nu}$ 驻定，则
$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}.
$$

**证明.** 将命题 8.1 与物质变分相加，得到
$$
\delta S
=\frac1{16\pi G}\int_U\sqrt{-g}\,
\bigl(G_{\mu\nu}+\Lambda g_{\mu\nu}-8\pi G T_{\mu\nu}\bigr)
\delta g^{\mu\nu}\,d^4x.
$$
变分基本引理应用于任意紧支撑对称张量 $\delta g^{\mu\nu}$，给出括号内张量逐点为零。证毕。

**注 8.3.** 上述证明只承担局部场方程。含边界时 GHY 项的精确符号还依赖边界是类时、类空还是带角点；本书不把这些不同变分问题压缩成同一个无条件公式。

## 8.3 Newton 极限

在弱场、静态、慢速极限中，取 Newtonian gauge 的一阶度规

$$
ds^2=-(1+2\Phi)dt^2+(1-2\Psi)\delta_{ij}dx^idx^j,
\qquad |\Phi|,|\Psi|\ll1,
$$

且物质主要由静止质量密度给出：

$$
T_{tt}\approx \rho.
$$

并忽略 $\partial_t\Phi,\partial_t\Psi$。本小节令 $\Lambda=0$；若 $\Lambda\ne0$，则应围绕相应的非平直背景作弱场展开，不能把宇宙学常数无条件附加到下面的 Minkowski 背景线性化公式。线性化曲率计算给出

$$
G_{00}=2\nabla^2\Psi
+O\!\left(h\,\partial^2h\right)
+O\!\left((\partial h)^2\right),
$$

其中 $h$ 统称 $\Phi,\Psi$ 的一阶度规扰动；余项记号同时记录扰动次数与导数阶，而不能只按 $\Phi^2$ 计数。

若物质的各向异性应力在该阶可忽略，则 Einstein 方程的无迹空间分量约束 $\Phi-\Psi$；在孤立源的衰减边界条件下得到 $\Phi=\Psi$。于是

Einstein 方程 $G_{tt}=8\pi G\rho$ 化为

$$
\nabla^2\Phi=4\pi G\rho.
$$

这正是 Newton 引力的 Poisson 方程。系数 $8\pi G$ 由要求 Newton 极限正确固定。只声明 $g_{tt}=-(1+2\Phi)$ 而不声明空间扰动、规范和应力假设，不足以单独推出 $G_{00}=2\nabla^2\Phi$。

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

## 本章小结

Einstein 方程是带明确边界口径的度规变分方程。Bianchi 恒等式保证几何侧协变守恒；Newton 极限还需要静态弱场展开、空间度规扰动和可忽略各向异性应力，不能只凭 $g_{tt}$ 的单个分量判断。
