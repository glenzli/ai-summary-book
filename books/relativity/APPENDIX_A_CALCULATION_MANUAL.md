# 附录 A 计算手册

本附录收集本书反复使用的计算流程。它不是新理论，而是帮助读者把定义落实成可检查的步骤。

## A.1 从度规计算 Christoffel 符号

给定坐标系下的度规 $g_{\mu\nu}$：

1. 写出矩阵 $g_{\mu\nu}$。
2. 求逆矩阵 $g^{\mu\nu}$。
3. 对各坐标求偏导 $\partial_\rho g_{\mu\nu}$。
4. 代入公式

$$
\Gamma^\rho_{\mu\nu}
=\frac12 g^{\rho\sigma}
\left(
\partial_\mu g_{\nu\sigma}
+\partial_\nu g_{\mu\sigma}
-\partial_\sigma g_{\mu\nu}
\right).
$$

5. 利用 $\Gamma^\rho_{\mu\nu}=\Gamma^\rho_{\nu\mu}$ 减少计算。

**检查点.** 若度规分量不依赖某坐标，对该坐标的偏导全部为零。若度规是对角的，很多交叉 Christoffel 符号自动消失。

## A.2 从 Christoffel 符号计算曲率

曲率张量按本书约定为

$$
R^\rho{}_{\sigma\mu\nu}
=\partial_\mu\Gamma^\rho_{\nu\sigma}
-\partial_\nu\Gamma^\rho_{\mu\sigma}
+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}
-\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma}.
$$

Ricci 张量为

$$
R_{\sigma\nu}=R^\rho{}_{\sigma\rho\nu}.
$$

标量曲率为

$$
R=g^{\sigma\nu}R_{\sigma\nu}.
$$

Einstein 张量为

$$
G_{\mu\nu}=R_{\mu\nu}-\frac12Rg_{\mu\nu}.
$$

**常见错误.** 不同教材的曲率号差可能相反。比较公式时必须同时比较 $R^\rho{}_{\sigma\mu\nu}$ 的定义和度规号差。

## A.3 Killing 守恒量

若 $K^\mu$ 是 Killing 向量场，满足

$$
\nabla_{(\mu}K_{\nu)}=0,
$$

则沿仿射测地线 $p^\mu=dx^\mu/d\lambda$，

$$
C=K_\mu p^\mu
$$

守恒。

计算步骤：

1. 找出度规不依赖的坐标 $x^a$。
2. 对应 Killing 向量为 $\partial_a$。
3. 写出 $C=g_{\mu a}p^\mu$。
4. 对时间平移通常取 $E=-p_t$，对轴对称取 $L_z=p_\phi$。

## A.4 测地线有效势

对静态球对称度规

$$
ds^2=-A(r)dt^2+B(r)dr^2+r^2d\Omega^2,
$$

限制在赤道面 $\theta=\pi/2$。守恒量为

$$
E=A(r)\dot t,
\qquad
L=r^2\dot\phi.
$$

归一化条件为

$$
g_{\mu\nu}\dot x^\mu\dot x^\nu=-\epsilon,
$$

其中类时 $\epsilon=1$，类光 $\epsilon=0$。代入得

$$
B(r)\dot r^2
=\frac{E^2}{A(r)}-\frac{L^2}{r^2}-\epsilon.
$$

这就是有效势分析的起点。

## A.5 从作用量得到能动张量

物质作用量

$$
S_m=\int \sqrt{-g}\,\mathcal L_m\,d^4x.
$$

能动张量定义为

$$
T_{\mu\nu}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_m}{\delta g^{\mu\nu}}.
$$

若 $\mathcal L_m$ 不含度规导数，则

$$
T_{\mu\nu}
=-2\frac{\partial\mathcal L_m}{\partial g^{\mu\nu}}
+g_{\mu\nu}\mathcal L_m.
$$

对实标量场

$$
\mathcal L
=-\frac12g^{\rho\sigma}\partial_\rho\phi\partial_\sigma\phi
-V(\phi),
$$

有

$$
T_{\mu\nu}
=\partial_\mu\phi\partial_\nu\phi
-\frac12g_{\mu\nu}
g^{\rho\sigma}\partial_\rho\phi\partial_\sigma\phi
-g_{\mu\nu}V(\phi).
$$

## A.6 线性化引力快速检查

写

$$
g_{\mu\nu}=\eta_{\mu\nu}+h_{\mu\nu}.
$$

到一阶有

$$
g^{\mu\nu}=\eta^{\mu\nu}-h^{\mu\nu}+O(h^2).
$$

Christoffel 符号为

$$
\Gamma^{(1)\rho}_{\mu\nu}
=\frac12\eta^{\rho\sigma}
\left(
\partial_\mu h_{\nu\sigma}
+\partial_\nu h_{\mu\sigma}
-\partial_\sigma h_{\mu\nu}
\right).
$$

Ricci 张量为

$$
R^{(1)}_{\mu\nu}
=\frac12
\left(
\partial_\rho\partial_\mu h^\rho{}_\nu
+\partial_\rho\partial_\nu h^\rho{}_\mu
-\Box h_{\mu\nu}
-\partial_\mu\partial_\nu h
\right).
$$

若引入

$$
\bar h_{\mu\nu}=h_{\mu\nu}-\frac12\eta_{\mu\nu}h
$$

并取 Lorenz 规范，则

$$
\Box\bar h_{\mu\nu}=-16\pi GT_{\mu\nu}.
$$

## A.7 恢复 $c$ 的常用公式

| 自然单位公式 | 恢复 $c$ |
| --- | --- |
| $E^2=p^2+m^2$ | $E^2=p^2c^2+m^2c^4$ |
| $r_s=2GM$ | $r_s=2GM/c^2$ |
| $\alpha=4GM/b$ | $\alpha=4GM/(bc^2)$ |
| $\Delta\phi=6\pi GM/[a(1-e^2)]$ | $\Delta\phi=6\pi GM/[a(1-e^2)c^2]$ |
| $T_H=1/(8\pi GM)$ | $T_H=\hbar c^3/(8\pi GMk_B)$ |

## A.8 最小审题清单

做相对论题目时先问：

1. 采用什么号差和单位？
2. 题目中的对象是张量、张量分量还是坐标函数？
3. 是否有 Killing 向量给出守恒量？
4. 曲线是类时、类光还是类空？
5. 参数是否为固有时或仿射参数？
6. 可观测量是否坐标无关，或至少是否与指定观察者有关？
7. 是否把坐标奇异当成曲率奇异？

这份清单能避免大多数初学错误。
