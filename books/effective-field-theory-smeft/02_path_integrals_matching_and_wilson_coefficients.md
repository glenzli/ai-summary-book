# 第二章：路径积分、匹配与 Wilson 系数

第一章说明重传播子可以在低能域展开，却还没有回答每个局域算符前的数值从何而来。答案不能靠量纲猜测：同一算符可能由树级交换产生，也可能只在一圈出现，符号、群论因子和对数更必须由计算决定。匹配（matching）把 UV 理论与 EFT 放在同一外态、同一红外调节和同一重整化约定下，要求它们在指定 $Q/M_{\rm gap}$ 与圈阶内给出相同的振幅或 Green 函数。重标量诱导的四轻场作用会贯穿这一计算：先在路径积分中积掉重场，再由经典方程得到局域作用量，最后用 $s,t,u$ 三个交换道核对顶点归一化，由此看清 Wilson 系数是阈值信息在所选算符基中的坐标。

## 2.1 积掉重场

设 UV 理论含轻场 $\phi$ 和重场 $\Phi$：
$$
Z[J]=\int D\phi\,D\Phi\,
\exp\left(iS_{\mathrm{UV}}[\phi,\Phi]+i\int J\phi\right).
$$

**定义 2.1（积掉重场后的轻场作用量）.** 在微扰路径积分口径下，形式定义
$$
\exp(iS_{\mathrm{heavy\ out}}[\phi])
\coloneqq
\mathcal N^{-1}\int D\Phi\,\exp(iS_{\mathrm{UV}}[\phi,\Phi]),
$$
其中 $\mathcal N$ 去掉与 $\phi$ 无关的归一化。这里积掉了重场的全部动量模式，但没有积掉轻场的高动量模式；所得对象一般非局域，也不是 1PI 有效作用量 $\Gamma[\phi]$。

**定义 2.1A（Wilsonian 作用量）.** 另选一个 coarse-graining 尺度 $\nu$ 与具体 regulator，把所有场的 $|k|>\nu$ 模式以及质量远高于 $\nu$ 的自由度积掉，所得 $S_\nu[\phi_{|k|<\nu}]$ 才称为该 regulator 下的 Wilsonian 作用量。它依赖 $\nu$ 和 coarse-graining 方案；Wilson 系数随 $\nu$ 的流动与第三章的重整化群语言相关，但两种“尺度依赖”必须在选定方案中对齐后才能比较。

**形式主义边界 2.1B.** 本章的 Minkowski 路径积分等式是标准微扰 QFT 输入：它按 Feynman 图、正规化和反项逐阶解释，不声称已经构造无限维振荡测度。树级匹配可独立由经典方程和振幅核验；圈级结论只在指定微扰阶、正规化和重整化方案内使用。

**警告 2.2.** $S_{\mathrm{heavy\ out}}$ 一般含非局域项。只有其重场 1PI 核在相对 $M_{\rm gap}$ 的低能域内展开后，解析部分才转化为局域算符级数；轻场 pole 和非解析项必须由 EFT 图保留。

## 2.2 树级匹配例子

**例 2.3（重标量诱导四轻场算符）.** 令
$$
\mathcal L_{\mathrm{UV}}
=
\frac12(\partial\phi)^2-\frac12m^2\phi^2
+\frac12(\partial\Phi)^2-\frac12M^2\Phi^2
-\frac{g}{2}\Phi\phi^2.
$$
在 $E\ll M$ 时，重场方程为
$$
(\Box+M^2)\Phi=-\frac{g}{2}\phi^2.
$$
形式求解得
$$
\Phi
=
-\frac{g}{2}\frac{1}{M^2+\Box}\phi^2
=
-\frac{g}{2M^2}\left(1-\frac{\Box}{M^2}+\cdots\right)\phi^2.
$$
代回拉氏量得到
$$
\Delta\mathcal L_{\mathrm{EFT}}
=
\frac{g^2}{8M^2}\phi^4
-\frac{g^2}{8M^4}\phi^2\Box\phi^2+\cdots,
$$
其中总导数按约定丢弃。

**推导状态.** 这是树级经典匹配。圈级匹配还需指定正规化和重整化方案。

## 2.3 振幅层面的同一例子

对例 2.3 的 $\phi\phi\to\phi\phi$ 散射，重场交换给出 $s,t,u$ 三个道。若顶点取为 $-ig$，则
$$
i{\cal A}_{\rm UV}
=(-ig)^2\left[
{i\over s-M^2}
+{i\over t-M^2}
+{i\over u-M^2}
\right].
$$
因此
$$
{\cal A}_{\rm UV}
=-g^2\left[
{1\over s-M^2}
+{1\over t-M^2}
+{1\over u-M^2}
\right].
$$
在 $|s|,|t|,|u|\ll M^2$ 下，
$$
{\cal A}_{\rm UV}
={g^2\over M^2}
\left(3+{s+t+u\over M^2}
 +{s^2+t^2+u^2\over M^4}
 +\cdots\right).
$$
对相同质量外线，$s+t+u=4m^2$。若在无质量近似下取 $m=0$，则第一阶导数修正从 $s^2+t^2+u^2$ 开始。

另一方面，EFT 中
$$
\Delta{\cal L}_{\rm EFT}\supset {g^2\over8M^2}\phi^4
$$
给出四点接触顶点
$$
i{\cal A}_{\rm EFT}^{(0)}
=i\,4!\,{g^2\over8M^2}
=i{3g^2\over M^2},
$$
正好对应三个交换道的低能常数项。导数算符
$$
-{g^2\over8M^4}\phi^2\Box\phi^2
$$
给出下一阶动量依赖项。

**警告 2.4（归一化）.** 若把 $\phi^4$ 写成 $-\lambda\phi^4/4!$ 或把源写成 $J=\phi^2/2$，顶点因子会重新分配。匹配时必须固定拉氏量归一化，不能只比较符号和量纲。

## 2.4 匹配条件

**定义 2.5（重整化振幅匹配）.** 固定外部轻态、低能运动学域、外线 residue 约定、轻参数重整化条件、规范、UV/IR regulator 和 subtraction scheme。对红外有限的 on-shell 振幅，或在 UV 与 EFT 两边采用同一红外 regulator 的振幅，要求
$$
\mathcal A_{\mathrm{UV}}^{\rm ren}
=
\mathcal A_{\mathrm{EFT}}^{\rm ren}
+R_{N,L}
$$
在所有保留的外态与独立运动学结构上成立，其中 $R_{N,L}$ 明确属于遗漏的 $Q/M_{\rm gap}$ 阶或圈阶。由此确定选定算符商空间中的 Wilson 坐标。

**定义 2.6（Green 函数匹配）.** 固定规范参数、正规化与 subtraction scheme、轻场和轻参数的重整化条件及外部场归一化，使一组足以分离目标局域张量结构的重整化 Green 函数在低能展开后相等。该等式是 off-shell、规范和场坐标依赖的；匹配完成前必须保留计算闭合所需的 EOM、BRST-exact、gauge-fixing 和 evanescent 结构，最后才投影到物理算符基。

**命题 2.7（算符基变换下的协变性）.** 固定某一质量维数、量子数和 flavor sector，并在分部积分与 EOM 商空间中取有限基。把算符和 Wilson 系数写成列向量 $\mathcal O$ 与 $C$，使
$$
\Delta\mathcal L=C^T\mathcal O.
$$
若另一组基满足 $\mathcal O'=B\mathcal O$，其中 $B$ 可逆，则同一拉氏量在新基中的系数为
$$
C'=B^{-T}C.
$$
任意外态之间的算符矩阵元列向量满足 $\mathcal M'=B\mathcal M$，所以组合 $C^T\mathcal M=C'^T\mathcal M'$ 不变。

**证明.** 由 $\mathcal O=B^{-1}\mathcal O'$，
$$
C^T\mathcal O=C^TB^{-1}\mathcal O'
=(B^{-T}C)^T\mathcal O'.
$$
这证明系数变换式。矩阵元对算符线性，故 $\mathcal M'=B\mathcal M$；代入即有
$$
C'^T\mathcal M'=C^TB^{-1}B\mathcal M=C^T\mathcal M.
$$
$\square$

**外部输入定理 2.7A（局域场重定义的等价定理，EFT-EQ）.** 在局域、扰动可逆、保持渐近单粒子 pole 且无 anomalous Jacobian 的场重定义下，适当归一化的 on-shell S-matrix 不变；Jacobian 和接触项须按所选 regulator 与重整化方案处理。Off-shell Green 函数、单个 Wilson 系数和中间 counterterm 可以改变。

**使用边界.** 该定理承担“场重定义不改物理”的量子层面结论，精确文献与适用假设见附录 B 的 EFT-EQ 项。命题 2.7 只证明有限维算符商空间中的线性基变换，不替代等价定理，也不覆盖 anomalous Jacobian、非局域或不可逆变换；高阶 EFT 截断还须保留场重定义诱导的高阶项，见第四章。

### 2.4.1 圈级匹配中的 hard/soft 分离

**命题 2.7B（共同软部分的消去）.** 固定一个圈阶和一个到 $\lambda^N$ 的低能截断，其中 $\lambda=Q/M_{\rm gap}$。假设所有更低阶 Wilson 系数已经匹配，并且在同一 IR regulator $\eta_{\rm IR}$ 下有
$$
\begin{aligned}
\mathcal A_{\rm UV}^{\rm ren}
&=\mathcal A_{\rm soft}^{[N]}(p,m,\mu,\eta_{\rm IR})
+H^{[N]}(p;M,\mu)+R_{\rm UV}^{[N]},\\
\mathcal A_{\rm EFT}^{\rm ren}
&=\mathcal A_{\rm soft}^{[N]}(p,m,\mu,\eta_{\rm IR})
+L^{[N]}(p;C(\mu),\mu)+R_{\rm EFT}^{[N]},
\end{aligned}
$$
其中 $H^{[N]}$ 与 $L^{[N]}$ 是外动量和轻质量的局域多项式，两个余项在所选紧致运动学域上一致为下一阶。则该阶匹配等价于
$$
H^{[N]}(p;M,\mu)=L^{[N]}(p;C(\mu),\mu),
$$
且共同软部分中的 IR poles、$\log(-p^2)$ 与 $\eta_{\rm IR}$ 依赖不进入 Wilson 系数。

**证明（书内推导）.** 将两条分解相减，共同的 $\mathcal A_{\rm soft}^{[N]}$ 逐项消去，得到
$$
\mathcal A_{\rm UV}^{\rm ren}-\mathcal A_{\rm EFT}^{\rm ren}
=H^{[N]}-L^{[N]}+R_{\rm UV}^{[N]}-R_{\rm EFT}^{[N]}.
$$
要求左边只到余项阶，恰好等价于局域多项式在所有独立张量结构上的系数相等。$\square$

**外部输入方法 2.7C（区域展开，EFT-REGIONS）.** 对满足给定质量/动量层级的微扰 Feynman 积分，expansion by regions 用各动量区域的齐次 integrand expansion 构造上述 hard/soft 分解。阈值处可能出现 hard、soft、potential、ultrasoft 等多个区域，不能只做零外动量 Taylor 展开。本书把该方法作为圈积分渐近展开的外部输入，不声称它为任意非微扰 QFT 提供一般收敛定理；来源见附录 B。

**反例 2.7D（把软对数误认作 Wilson 系数）.** 若 UV 与 EFT 圈图都含同一个 $\log(-p^2/\eta_{\rm IR}^2)$，直接把 UV 图的该项读入 $C_i$ 会得到依赖外动量且在 $p^2=0$ 非解析的“系数”，它不可能乘在局域算符上。命题 2.7B 要求先减去 EFT 软图；只剩对 $p$ 解析、但可含 $\log(M^2/\mu^2)$ 的 hard 部分进入阈值系数。

**警告 2.7E（投影顺序）.** On-shell 匹配只能确定 EOM/IBP 商空间中的坐标，看不见冗余方向；off-shell 匹配则会产生这些方向以及规范依赖结构。两种方法都必须先在所选阶完成重整化和张量分解，再作 EOM/IBP/BRST 投影。过早把 EOM 算符置零可能漏掉它投影到物理算符的 counterterm，第三、四章给出代数条件。

## 2.5 匹配尺度

单重阈值问题通常取 $\mu_{\rm match}\simeq M_{\rm gap}$。若一圈匹配给出
$$
C_i(\mu_{\rm match})=a_i+b_i\log{M^2\over\mu_{\rm match}^2},
$$
则选择 $\mu_{\rm match}\simeq M$ 可避免大对数。若实验尺度 $\mu_{\rm exp}\ll M$，大对数应由 RGE 从 $\mu_{\rm match}$ 运行到 $\mu_{\rm exp}$ 得到，而不是留在固定阶匹配系数中。$\mu_{\rm match}$ 的改变由阈值系数与 RG 演化在所算阶内抵消，并不移动物理阈值 $M$。

若存在 $M_1\gg M_2\gg Q$，在一个共同尺度同时积掉两者会产生 $\log(M_1/M_2)$。顺序匹配先在 $M_1$ 附近积掉第一层，再运行到 $M_2$ 并作第二次匹配，可系统重求和该对数；若选择一次匹配，必须把残留大对数计入固定阶误差。

**原则 2.8（阈值与运行分工）.** 匹配负责短程阈值常数，运行负责跨尺度对数。把二者混在一起会导致重复计数或遗漏对数。

## 2.6 阈值信息如何留下

重标量例子从经典解和三道散射振幅得到同一个 $\phi^4$ 系数，说明匹配既可在作用量层面完成，也可在 on-shell 商空间中完成。到了圈级，两边必须共享 scheme、轻参数与 IR 处方；共同的 soft 非解析部分由 EFT 自身重现，二者相减后剩余的 hard 局域多项式才是 Wilson 系数。若有多个分离阈值，就在各物理质量附近依次匹配并在其间运行，避免把大质量对数留给单次固定阶计算。

## 练习

**练习 2.1.** 对例 2.3 计算 $\phi\phi\to\phi\phi$ 的树级振幅，并与 EFT 的 $\phi^4$ 顶点比较。

**练习 2.2.** 说明 on-shell 振幅匹配为什么看不到某些 EOM 冗余算符。

**练习 2.3.** 用本章的顶点归一化，验证 $\phi^4$ 接触项给出的常数振幅为 $3g^2/M^2$。

**练习 2.4.** 设 UV 与 EFT 振幅各含相同的 $1/\epsilon_{\rm IR}+\log(\mu^2/(-p^2))$。按命题 2.7B 写出二者之差，并说明为何该组合不应出现在局域 Wilson 系数中。
