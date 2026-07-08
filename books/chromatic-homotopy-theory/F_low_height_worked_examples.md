# 附录 F：低高度计算样例

## F.1 高度 0：有理稳定同伦

**命题 F.1.** $L_{K(0)}X\simeq H\mathbb Q\otimes X$。

**证明草图.** $K(0)=H\mathbb Q$。有理化是关于 $H\mathbb Q$ 的 Bousfield localization，因而 $L_{K(0)}$ 是 rational localization。完整证明依赖谱的有理化模型。证毕。

**例 F.2.** 对球谱，
$$
\pi_*L_{K(0)}\mathbb S\cong
\begin{cases}
\mathbb Q,& *=0,\\
0,& *\ne0.
\end{cases}
$$

**证明.** $H\mathbb Q\otimes\mathbb S\simeq H\mathbb Q$，其同伦群如上。证毕。

## F.2 高度 1：K-theory 型信息

**例 F.3.** 在奇素数 $p$，高度 $1$ 的 Morava E-theory 与 $p$-complete complex K-theory 的 Adams summand 相关。其 stabilizer group 与 $\mathbb Z_p^\times$ 相关。

**警告 F.4.** 高度 $1$ 的 explicit formula 对 $p=2$ 与奇素数有不同形式。使用 $KO$、$KU$、Adams summand 或 $E_1$ 时必须说明模型。

**命题 F.5.** 乘法形式群在特征 $p$ 下高度为 $1$。

**证明.** 见附录 A 的推论 A.7。证毕。

## F.3 Moore spectrum 的 type

**定义 F.6.** $p$-local Moore spectrum $M(p)$ 定义为 cofiber
$$
\mathbb S_{(p)}\xrightarrow{p}\mathbb S_{(p)}\to M(p).
$$

**命题 F.7.** $M(p)$ 的有理同调为零，因此若非零，它的 type 至少为 $1$。

**证明.** 对 cofiber 序列张量 $H\mathbb Q$，得到
$$
H\mathbb Q\xrightarrow{p}H\mathbb Q\to H\mathbb Q\otimes M(p).
$$
在 $\mathbb Q$ 上乘以 $p$ 是同构，所以 cofiber 为零。故 $K(0)_*M(p)=0$。证毕。

**外部输入 F.8.** $M(p)$ 是否携带特定 $v_1$ self-map 及其周期性依赖素数和有限谱构造。完整 type $1$ 周期性属于 Hopkins-Smith periodicity theorem 的低高度实例。

## F.4 高度 2：supersingular elliptic curves

**例 F.9.** 若 $C/\overline{\mathbb F}_p$ 是 supersingular elliptic curve，则其形式群高度为 $2$。在该点的 Lubin-Tate 变形给出高度 $2$ Morava E-theory 的几何来源。

**外部输入 F.10.** Supersingular 点附近的 tmf/TMF 局部模型与 $K(2)$-local Morava E-theory 和有限 stabilizer subgroup descent 相关。完整陈述需 tmf 构造和 level structure 定位。

## F.5 低高度 fracture

**例 F.11.** $n=1$ 时 fracture square 形如
$$
\begin{array}{ccc}
L_1X & \longrightarrow & L_{K(1)}X\\
\downarrow & & \downarrow\\
L_0X & \longrightarrow & L_0L_{K(1)}X
\end{array}
$$
它把 rational part 和 $K(1)$-local part 粘合。

**警告 F.12.** 即便在高度 $1$，右下角也不能在无假设下删除。粘合项控制 rational information inside the $K(1)$-local piece。

## 本附录小结

低高度样例提供直观入口，但不能替代一般定理。高度 $0$ 是有理化，高度 $1$ 与 K-theory 型周期性相关，高度 $2$ 进入椭圆曲线和 tmf。每个低高度公式都需要记录 prime、completion 和模型。

## 练习

**练习 F.1.** 用 cofiber 序列证明 $H\mathbb Q_*M(p)=0$。

**练习 F.2.** 查阅 $p=2$ 时 $K(1)$-local sphere 与 $KO$ 的关系，记录与奇素数情形的差异。

**练习 F.3.** 给出一个 supersingular elliptic curve 的例子，并记录其所在素数条件。
