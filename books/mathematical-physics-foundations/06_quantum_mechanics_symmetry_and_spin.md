# 第六章：量子力学、对称性与自旋

量子力学的形式语言把态表示为 Hilbert 空间中的射线，把可观测量表示为自伴算符，把对称性表示为保持转移概率的变换。这个框架并不只是“把经典量加帽子”：正则对易关系、自伴性、群的酉表示和谱分解共同决定了可测量结构。本章把第二、三、五章的几何、表示和分析语言合并为有限自由度量子论。

## 6.1 态、可观测量和演化

**定义 6.1.** 纯态是 Hilbert 空间 $\mathcal H$ 中非零向量的相位等价类 $[\psi]$。可观测量由自伴算符 $A$ 表示；其谱投影 $E_A(\Delta)$ 给出测量值落入 Borel 集 $\Delta$ 的概率
$$
\|E_A(\Delta)\psi\|^2
$$
其中 $\|\psi\|=1$。

**定义 6.2.** 若 $H$ 为自伴 Hamiltonian，则时间演化为
$$
U(t)=e^{-itH}.
$$
Stone 定理保证每个强连续酉时间演化都有这种形式。

**定义 6.3.** 正则对易关系为
$$
[\hat q^i,\hat p_j]=i\delta^i{}_j,\qquad [\hat q^i,\hat q^j]=[\hat p_i,\hat p_j]=0.
$$

**命题 6.1 (`P`).** 设 $H$ 是 Hilbert 空间 $\mathcal H$ 上的有界自伴算符，$A_{\rm S}:I\to\mathcal B(\mathcal H)$ 在算符范数下可微，并令 $U(t)=e^{-itH}$、$A(t)=U(t)^*A_{\rm S}(t)U(t)$。则
$$
\frac{dA(t)}{dt}=i[H,A(t)]+\left(\frac{\partial A}{\partial t}\right)(t).
$$

**证明.** 有界性保证指数级数在算符范数下收敛，并可逐项求导，因此 $\dot U=-iHU$、$\dot U^*=iU^*H$。算符范数中的乘积法则给出
$$
\begin{aligned}
\dot A(t)
&=iU(t)^*HA_{\rm S}(t)U(t)
+U(t)^*\dot A_{\rm S}(t)U(t)
-iU(t)^*A_{\rm S}(t)HU(t)\\
&=i[H,A(t)]+U(t)^*\dot A_{\rm S}(t)U(t),
\end{aligned}
$$
其中 $H$ 与 $U(t)$ 交换。最后一项按定义就是 $(\partial A/\partial t)(t)$。$\square$

对无界 $H$ 或 $A$，上述算符范数证明不适用。物理上使用 Heisenberg 方程时，必须另给共同不变稠密核，并在该核上的强导数或矩阵元意义解释等式；本书不把缺少这些条件的形式对易子升级为 `P`。

**定理 6.2 (`E`, Stone-von Neumann).** 有限自由度 Weyl 形式 CCR 的不可约强连续酉表示在相位因子外唯一，等价于 Schrödinger 表示。

**外部输入边界.** 该定理不适用于无限自由度场论；量子场论中会出现不等价表示。所用 Weyl 形式与不可约、强连续假设见 [SOURCES.md](SOURCES.md) 的 `E-6.2`。

## 6.2 对称性和 Wigner 定理

**定理 6.3 (`E`, Wigner 定理).** 保持纯态转移概率的双射由酉或反酉算符在射线空间上诱导。

**外部输入边界.** 本书将连续连通对称群取为酉表示；含时间反演的离散对称可为反酉。精确的射线双射版本见 [SOURCES.md](SOURCES.md) 的 `E-6.3`。

**定义 6.4.** 若 Lie 群 $G$ 在量子系统上作为强连续酉表示 $U:G\to\mathcal U(\mathcal H)$ 作用，则对每个 $\xi\in\mathfrak g$，映射
$$
t\longmapsto U(\exp t\xi)
$$
是强连续一参数酉群。由 Stone 定理，存在唯一自伴算符 $\hat J_\xi$，使
$$
U(\exp t\xi)=e^{-it\hat J_\xi}.
$$
生成元定义域恰可表为
$$
\mathcal D(\hat J_\xi)=
\left\{\psi\in\mathcal H:
\lim_{t\to0}\frac{U(\exp t\xi)\psi-\psi}{t}
\text{ 在 }\mathcal H\text{ 中存在}\right\},
$$
且该极限等于 $-i\hat J_\xi\psi$。因此这里的无穷小生成元一般不是处处定义的有界算符。若还要把所有 $\xi$ 同时组织成共同域上的 Lie 代数表示，则需另行引入光滑向量理论；本书后文不调用该更强结论。此处所用的精确版本见 [SOURCES.md](SOURCES.md) 的 `E-5.8`。

## 6.3 自旋

**命题 6.4 (`P`).** 设 $V$ 是有限维复 Hilbert 空间，$J_1,J_2,J_3$ 是自伴算符，生成不可约表示并满足
$$
[J_i,J_j]=i\epsilon_{ijk}J_k,
$$
则有限维不可约自旋表示由 $j\in\frac12\mathbb Z_{\ge0}$ 标记，且
$$
J^2=j(j+1)I,\qquad m=-j,-j+1,\ldots,j.
$$

**证明.** 定义 $J_\pm=J_1\pm iJ_2$。直接使用对易关系可得
$$
[J_3,J_\pm]=\pm J_\pm,
\quad
[J^2,J_i]=0,
\quad
J^2=J_-J_++J_3(J_3+1)=J_+J_-+J_3(J_3-1).
$$
由 Schur 引理，$J^2=cI$，其中 $c\in\mathbb R$。自伴算符 $J_3$ 有最大本征值 $j$ 及单位本征向量 $v$。第一式说明 $J_+v$ 若非零便有本征值 $j+1$，故 $J_+v=0$。于是第二组恒等式给出 $c=j(j+1)$。

令 $v_r=J_-^rv$。只要 $v_r\ne0$，它就是本征值 $j-r$ 的向量。有限维性保证存在最小 $N\ge0$，使 $v_N\ne0$ 而 $J_-v_N=0$。记 $m=j-N$。作用恒等式 $J^2=J_+J_-+J_3(J_3-1)$ 于 $v_N$，得到
$$
j(j+1)=m(m-1).
$$
因 $m=j-N$，上式化为 $(N+1)(2j-N)=0$，故 $N=2j\in\mathbb Z_{\ge0}$，并且最低权为 $-j$。

子空间 $W=\operatorname{span}\{v_0,\ldots,v_N\}$ 在 $J_3,J_-$ 下不变。由 $[J_+,J_-]=2J_3$ 与 $[J_3,J_-]=-J_-$ 对 $r$ 归纳，得到
$$
J_+J_-^rv=r(2j-r+1)J_-^{r-1}v.
$$
因此 $W$ 也在 $J_+$ 下不变。作为系数非负性的检查，
$$
\|J_-v_r\|^2
=\langle v_r,J_+J_-v_r\rangle
=\bigl(j(j+1)-(j-r)(j-r-1)\bigr)\|v_r\|^2
$$
成立。于是 $W$ 在全部 $J_i$ 下不变；不可约性迫使 $W=V$。各 $v_r$ 具有互异 $J_3$ 本征值，故线性无关，因此 $\dim V=N+1=2j+1$，且 $m=j,j-1,\ldots,-j$。$\square$

**例 6.5（自旋 $1/2$）.** 令
$$
\sigma_1=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_2=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_3=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad J_i=\frac12\sigma_i.
$$
矩阵乘法给出
$\sigma_i\sigma_j=\delta_{ij}I+i\epsilon_{ijk}\sigma_k$，所以
$[J_i,J_j]=i\epsilon_{ijk}J_k$。又
$J^2=(\sigma_1^2+\sigma_2^2+\sigma_3^2)/4=3I/4$，而 $J_3$ 的本征值为 $\pm1/2$，与 $j=1/2$ 的公式 $j(j+1)=3/4$ 一致。

## 6.4 半经典极限

**命题 6.5 (`S`).** WKB 形式 $\psi=a e^{iS/\hbar}$ 代入 Schrödinger 方程，$\hbar^0$ 阶给出 Hamilton-Jacobi 方程。

**推导说明（标准物理口径）.** 对
$$
i\hbar\partial_t\psi=\left(-\frac{\hbar^2}{2m}\Delta+V\right)\psi
$$
代入 WKB 形式并按 $\hbar$ 幂次整理。最高阶相位项为
$$
\partial_tS+\frac{|\nabla S|^2}{2m}+V=0.
$$
该推导假设相位光滑且忽略焦散与全局 Maslov 指数。$\square$

## 练习

**练习 6.1.** 假设 $q,p,H$ 有一个共同不变稠密核，且该核上 $[q,p]=iI$、$[V(q),p]=iV'(q)$。用核上的 Heisenberg 方程从 $H=p^2/(2m)+V(q)$ 推出 Ehrenfest 方程。

**练习 6.2.** 对自旋 $1/2$ 表示，写出 $J_i=\sigma_i/2$ 并验证角动量对易关系。
