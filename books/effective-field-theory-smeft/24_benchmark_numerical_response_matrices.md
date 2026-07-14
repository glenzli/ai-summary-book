# 第二十四章：基准数值响应矩阵

抽象响应式只有代入一组可手算数字后，才能暴露符号、量纲和退化方向中的错误。这里依次把电弱输入传播、Higgs 信号强度、高质量 dilepton 和 flavor 匹配链写成小型数值矩阵，并用给定协方差计算 Fisher 信息；这些数字是计算基准，不冒充实验拟合。dilepton 例子尤其需要两种比值：$\rho_{\rm loc,b}=\sqrt{s_b}/M_{\rm gap}$ 衡量 bin 到最近遗漏奇点的距离，$\rho_{\rm ins,b}=|C^{(6)}s_b/\Lambda_{\rm ref}^2|$ 粗略衡量一次维数六插入的大小。参考尺度 $\Lambda_{\rm ref}$ 只定义 Wilson 坐标；即使单尺度 UV 匹配先确定 $M_{\rm gap}=M$，也还要另作 $\Lambda_{\rm ref}=M$ 的坐标选择，才能让两个尺度数值相同。

## 24.1 电弱精密基准：$m_W$

取基准值
$$
s^2=0.231,\qquad c^2=0.769,\qquad c^2-s^2=0.538.
$$
第二十三章给出
$$
{\delta m_W^2\over m_W^2}
=
{s^2\over c^2-s^2}\epsilon_\alpha
-{s^2\over c^2-s^2}\epsilon_G
-{c^2\over c^2-s^2}\epsilon_Z
+\epsilon_W.
$$
因此数值响应行为
$$
M_{m_W^2}
\simeq
\begin{pmatrix}
0.429&-0.429&-1.430&1.000
\end{pmatrix}.
$$
对 $m_W$ 本身，
$$
M_{m_W}
\simeq
\begin{pmatrix}
0.215&-0.215&-0.715&0.500
\end{pmatrix}.
$$

**解释 24.1.** $\epsilon_Z$ 的系数较大，是因为 $m_W$ 预测对中性规范质量关系的位移敏感。该数值不代表实验误差，只代表输入方案的线性传播。

## 24.2 Higgs 基准：生产、衰变、总宽度

定义三个归一化振幅参数
$$
\theta=(a_g,\delta y_t,\delta_\Gamma)^T,
$$
其中 $a_g$ 是 $hgg$ contact 振幅相对 SM loop 振幅的归一化贡献，$\delta y_t$ 是 top-loop 振幅的相对位移，$\delta_\Gamma$ 是总宽度相对位移。对两个简化信号强度
$$
\mu_1=gg\to h\to ZZ,\qquad
\mu_2=gg\to h\to \gamma\gamma,
$$
若只让生产端和总宽度变化，则
$$
\Delta\mu_1=\Delta\mu_2=2a_g+2\delta y_t-\delta_\Gamma.
$$
响应矩阵为
$$
M_H=
\begin{pmatrix}
2&2&-1\\
2&2&-1
\end{pmatrix}.
$$
该矩阵秩为 $1$。它有两个 flat directions，例如
$$
(1,-1,0),\qquad (1,0,2).
$$

**解释 24.2.** 若不加入 $t\bar th$、VBF、总宽度约束或其他衰变道，这两个信号强度不能分离 $hgg$ contact、top Yukawa shift 和 total width shift。

## 24.3 高质量 dilepton 基准

这里必须先区分坐标归一化与理论有效性。固定 Wilson 系数的**参考尺度**
$\Lambda_{\rm ref}$，取两个 bin 的无量纲坐标
$$
x_b={s_b\over\Lambda_{\rm ref}^2},\qquad x_1=0.25,\quad x_2=1.00.
$$
这一定义不把 $\Lambda_{\rm ref}$ 认作 UV 质量隙。局域 EFT 展开是否适用由
$$
\rho_{\rm loc,b}={\sqrt{s_b}\over M_{\rm gap}}
$$
控制；只有在具体 UV 匹配先给出 $M_{\rm gap}=M$、并另选
$\Lambda_{\rm ref}=M$ 时，才可把 $x_b$ 同时解释为局域展开参数的平方。

固定公共定义尺度 $\mu_{\rm obs}$，令无量纲 contact 坐标
$c\coloneqq C_{\ell q}^{(6)}(\mu_{\rm obs})$。线性响应设为
$$
\Delta_b=x_b c.
$$
则
$$
M_{\ell\ell}=
\begin{pmatrix}
0.25\\
1.00
\end{pmatrix}.
$$
若保留平方项，取
$$
\Delta_b=x_bc+x_b^2c^2,
$$
则这个特定基准模型的二次响应为
$$
\mathsf Q_{\ell\ell}=
\begin{pmatrix}
0.0625\\
1.0000
\end{pmatrix}.
$$
第二个 bin 的灵敏度高，其平方项也更大。平方项与线性项之比在 $c\ne0$ 时为
$$
\rho_{\rm ins,b}=|x_bc|,
$$
这里的等式使用了线性项与平方项都取单位权重的基准假设；真实过程的比值还含
干涉、PDF、helicity 和接受度权重。第二个 bin 对**插入截断**更敏感；它是否接近 EFT 的局域性边界仍取决于
$\rho_{\rm loc,b}$，不能从 $x_2=1$ 单独判断。后续 Fisher 计算只把
$\Lambda_{\rm ref}$ 当作 Wilson 坐标的归一化，并不声称第二个 bin 已满足
或违反 $\sqrt{s_2}\ll M_{\rm gap}$。

## 24.4 Fisher 矩阵例子

在线性化点 $c=0$，设两个归一化 dilepton bin 误差独立且
$$
\Sigma=\mathrm{diag}(0.1^2,0.2^2).
$$
单参数 Fisher 信息为
$$
F=M^T\Sigma^{-1}M
={0.25^2\over0.1^2}+{1.00^2\over0.2^2}
=6.25+25=31.25.
$$
因此一标准差误差估计为
$$
\sigma_c={1\over\sqrt F}\simeq0.179.
$$
若移除第二个高能 bin，则
$$
F=6.25,\qquad \sigma_c=0.400.
$$
这展示了高能 bin 对线性 Fisher 信息的作用。该误差估计尚未使用
$\mathsf Q_{\ell\ell}$，也未对 $M_{\rm gap}$ 作假设；把第二个 bin 纳入物理限制
前，仍须另给 $\rho_{\rm loc,2}$、$\rho_{\rm ins,2}$ 与截断误差。

## 24.5 Flavor 链的二维基准

取两个 SMEFT Wilson 系数、两个 LEFT 系数和两个低能观测量，并假设所有矩阵都已
在声明的高尺度、电弱匹配尺度和低尺度上、以相容的 Wilson 归一化计算。设
$$
R_{\rm SMEFT}=
\begin{pmatrix}
1&0.1\\
0&0.9
\end{pmatrix},\quad
T=
\begin{pmatrix}
1&1\\
0&1
\end{pmatrix},\quad
R_{\rm LEFT}=
\begin{pmatrix}
0.8&0\\
0.2&1
\end{pmatrix},
$$
并设低能矩阵元响应
$$
N=
\begin{pmatrix}
1&0\\
0&2
\end{pmatrix}.
$$
总响应矩阵为
$$
M=NR_{\rm LEFT}TR_{\rm SMEFT}
=
\begin{pmatrix}
0.8&0.8\\
0.4&2.2
\end{pmatrix}.
$$

**解释 24.3.** 即使高尺度只开第二个 SMEFT 系数，运行和匹配也会使两个低能观测量同时响应。Flavor 结果若省略 $R_{\rm SMEFT}$、$T$ 或 $R_{\rm LEFT}$，就不是可复核的 SMEFT 结果。

## 24.6 接入实验数据

接入一个具体数据集时，需要由相应过程计算确定：

1.  $M$：由事件模拟、解析公式或工具链给出的线性响应；
2.  $\mathsf Q$：若保留平方项，由二次响应给出；
3.  $\Sigma$：实验和理论协方差；
4.  bin 定义、硬尺度 $Q$ 和对 $M_{\rm gap}$ 条件化的局域切割；
5.  Wilson 参数坐标、$\Lambda_{\rm ref}$、定义尺度与 $\rho_{\rm ins}$；
6.  输入方案、匹配/RGE 精度和理论误差。

## 24.7 从基准矩阵到真实数据

这些小矩阵把全书的计算链压缩成可手算对象：输入方案产生 $m_W$ 响应行，Higgs
矩阵的秩暴露两条 flat directions，高能 bin 增强 Fisher 信息，flavor 矩阵乘积则
保留匹配与运行造成的方向旋转。它们只在所声明的坐标和线性化点上成立。尤其是
$x_2=1$ 只说明该 bin 相对 $\Lambda_{\rm ref}$ 的坐标数值；局域性仍由未知或
模型给定的 $M_{\rm gap}$ 决定，插入层级则由 $|x_2c|$ 决定。

## 练习

**练习 24.1.** 用 $s^2=0.23$ 重算 $M_{m_W}$。

**练习 24.2.** 求 $M_H$ 的秩和两个线性无关的零方向。

**练习 24.3.** 若 dilepton 第二个 bin 的误差改为 $0.5$，重算 $\sigma_c$。

**练习 24.4.** 保持有量纲系数
$c_{\ell q}^{(6)}=c/\Lambda_{\rm ref}^2$ 不变，把
$\Lambda_{\rm ref}'=2\Lambda_{\rm ref}$。求新的无量纲 $c'$ 与
$x_b'=s_b/(\Lambda_{\rm ref}')^2$，并证明
$\rho_{\rm ins,b}=|x_bc|$ 不变。说明 $\rho_{\rm loc,b}$ 为何也不受这次坐标
重标度影响。
