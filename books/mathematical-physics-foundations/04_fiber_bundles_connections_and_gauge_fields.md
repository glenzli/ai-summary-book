# 第四章：纤维丛、联络与规范场

规范场不是“多余变量”本身，而是局部平凡化之间的粘合规则和联络选择。电磁势 $A$ 在一个坐标片中是一形式，换规范时会变化；曲率 $F$ 才是可全局比较的场强。非交换规范理论中同一结构由主丛、联络和曲率表达。本章建立这一语言，并把 Chern-Weil 形式作为异常和拓扑荷的前置接口。

## 4.1 主丛和局部规范

**定义 4.1.** 右主 $G$-丛是光滑映射 $\pi:P\to M$，带自由且传递于纤维的右 $G$ 作用，并局部同构于 $U\times G\to U$。

**定义 4.2.** 主丛联络是 $P$ 上的 $\mathfrak g$ 值一形式 $\Theta$，满足：
1. $\Theta(\xi_P)=\xi$；
2. $R_g^*\Theta=\operatorname{Ad}_{g^{-1}}\Theta$。

局部截面 $s:U\to P$ 拉回给出局部规范势 $A=s^*\Theta\in\Omega^1(U,\mathfrak g)$。

**命题 4.1 (`P`).** 若 $s'=sg$ 是同一开集上的另一局部截面，则
$$
A'=g^{-1}Ag+g^{-1}dg.
$$

**证明.** 固定 $x\in U$ 和 $v\in T_xU$，取曲线 $x(t)$ 满足 $x(0)=x$、$\dot x(0)=v$。写 $p(t)=s(x(t))$、$h(t)=g(x(t))$。乘积曲线 $p(t)h(t)$ 的导数分成右平移后的 $\dot p(0)$ 与由 $\dot h(0)$ 产生的竖直向量。联络的一形式等变性与对基本竖直向量的再现性分别给出
$$
\Theta_{s'(x)}(ds'_xv)
=\operatorname{Ad}_{g(x)^{-1}}\Theta_{s(x)}(ds_xv)
+(g^{-1}dg)_x(v).
$$
这对所有 $x,v$ 成立，拉回后即为 $A'=g^{-1}Ag+g^{-1}dg$。$\square$

## 4.2 曲率和 Bianchi 恒等式

**定义 4.3.** 局部曲率二形式为
$$
F_A=dA+\frac12[A\wedge A],
$$
其中 $[A\wedge A](X,Y)=[A(X),A(Y)]-[A(Y),A(X)]$。

**命题 4.2 (`P`).** 规范变换下曲率满足
$$
F_{A'}=g^{-1}F_Ag.
$$

**证明.** 记 $\theta=g^{-1}dg$。由 $d(g^{-1})=-g^{-1}(dg)g^{-1}$ 可得 Maurer-Cartan 恒等式 $d\theta+\frac12[\theta\wedge\theta]=0$。把 $A'=g^{-1}Ag+\theta$ 代入曲率公式，使用分次 Leibniz 规则，得到
$$
\begin{aligned}
F_{A'}
&=g^{-1}(dA)g-[\theta\wedge g^{-1}Ag]+d\theta\\
&\quad+\frac12[g^{-1}Ag\wedge g^{-1}Ag]
+[g^{-1}Ag\wedge\theta]+\frac12[\theta\wedge\theta].
\end{aligned}
$$
两个交叉项相消，纯 $\theta$ 项由 Maurer-Cartan 恒等式相消，而括号与共轭相容，所以剩余项为
$$
g^{-1}\left(dA+\frac12[A\wedge A]\right)g=g^{-1}F_Ag.
$$
$\square$

**定义 4.4.** 协变外微分 $D_A$ 作用在伴随丛值形式上：
$$
D_A\alpha=d\alpha+[A\wedge\alpha].
$$

**命题 4.3 (`P`, Bianchi 恒等式).**
$$
D_AF_A=0.
$$

**证明.** 展开
$$
D_AF_A=d(dA+\tfrac12[A\wedge A])+[A\wedge dA]+\tfrac12[A\wedge[A\wedge A]].
$$
由 $d^2A=0$，并按分次括号的 Leibniz 规则
$$
d[A\wedge A]=[dA\wedge A]-[A\wedge dA],
$$
注意二形式 $dA$ 与一形式 $A$ 交换时分次符号为正，所有含 $dA$ 的项两两相消。剩余的 $[A\wedge[A\wedge A]]/2$ 在三个向量上求值后是 Lie 代数 Jacobi 和，因而为零。$\square$

## 4.3 Chern-Weil 形式

**定义 4.5.** $\operatorname{Ad}$-不变多项式 $P$ 是 $\mathfrak g$ 上满足 $P(\operatorname{Ad}_gX_1,\ldots,\operatorname{Ad}_gX_k)=P(X_1,\ldots,X_k)$ 的对称 $k$-线性形式。

**命题 4.4 (`P`).** 若 $P$ 为 $\operatorname{Ad}$-不变多项式，则 $P(F_A,\ldots,F_A)$ 是闭形式。

**证明.** $\operatorname{Ad}$-不变性的无穷小形式是：对任意 $Y,X_1,\ldots,X_k\in\mathfrak g$，
$$
\sum_{r=1}^kP(X_1,\ldots,[Y,X_r],\ldots,X_k)=0.
$$
把该恒等式逐点应用于 $A$ 与 $\mathfrak g$ 值形式，所有联络项相消，故
$$
dP(F_A^k)=kP(D_AF_A,F_A,\ldots,F_A)=0
$$
其中最后一步使用命题 4.3。故该 $2k$-形式闭合。$\square$

**定理 4.5 (`E`).** Chern-Weil 形式的 de Rham 上同调类与联络选择无关，并等于相应特征类的实系数像。

**外部输入边界.** 本书使用该定理解释拓扑荷、instantons 和异常多项式的几何来源；完整证明依赖 de Rham 理论和特征类，定位见 [SOURCES.md](SOURCES.md) 的 `E-4.5`。

**例 4.6（平面上的常磁场）.** 在平凡 $U(1)$-丛上，以实一形式约定取
$$
A=\frac B2(-y\,dx+x\,dy).
$$
因 $U(1)$ Abelian，$F=dA=B\,dx\wedge dy$。规范变换 $A'=A+d\lambda$ 满足 $F'=F$。对半径 $R$ 的逆时针圆周 $C_R$，参数化
$(x,y)=(R\cos\theta,R\sin\theta)$ 给出
$$
\oint_{C_R}A
=\int_0^{2\pi}\frac{BR^2}{2}\,d\theta
=B\pi R^2
=\int_{D_R}F.
$$
这直接验证 Stokes 定理，并显示闭路积分记录规范不变的磁通。

## 练习

**练习 4.1.** 对 $G=U(1)$，证明规范变换律退化为 $A'=A+g^{-1}dg$，曲率 $F=dA$ 规范不变。

**练习 4.2.** 从 Yang-Mills 作用量 $\int \operatorname{tr}(F_A\wedge *F_A)$ 的一阶变分推出 $D_A*F_A=0$。
