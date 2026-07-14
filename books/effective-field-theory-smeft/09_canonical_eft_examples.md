# 第九章：典型 EFT 例子

“EFT 就是把重粒子积掉后按维数写算符”只覆盖了部分情形。低能弱作用的四费米子顶点确实来自树级 $W$ 交换，光子自散射却由电子一圈阈值产生；pion 的低能展开由 Goldstone 导数计数组织，引力则说明微扰不可重整化并不妨碍有限能区内的逐阶预测。把这四个例子并排计算，可以分清 EFT 中哪些结构是共同的，哪些取决于自由度和对称性。每个例子都要回答同样的问题：最近遗漏尺度是什么，领先局域项在哪一阶出现，Wilson 系数含哪些耦合或环因子，误差按哪一个小参数估计。Fermi 传播子展开还会提供一条可手算的主线，连接树级匹配与第一个导数修正。

## 9.1 Fermi 理论

**定义 9.1（Fermi 四费米子算符）.** 在能量远低于 $W$ 质量时，带电弱流相互作用可写为
$$
\mathcal L_F
=
-\frac{4G_F}{\sqrt2}
(\bar \nu_\mu\gamma^\mu P_L\mu)
(\bar e\gamma_\mu P_L\nu_e)
+\cdots.
$$

**命题 9.2（树级匹配关系）.** 在标准模型树级近似下，
$$
\frac{G_F}{\sqrt2}=\frac{g^2}{8m_W^2}.
$$

**推导说明.** 低能区 $q^2\ll m_W^2$ 中，$W$ 传播子给出
$$
\frac{-i}{q^2-m_W^2}
=
\frac{i}{m_W^2}
\left(1+\frac{q^2}{m_W^2}+\cdots\right).
$$
将两个带电流顶点与领先传播子相乘，并与 Fermi 拉氏量的四费米子顶点比较，得到上式。符号依赖拉氏量约定，但 $G_F/\sqrt2=g^2/(8m_W^2)$ 是标准树级关系。$\square$

**第一个导数修正。** 保留传播子下一项给出
$$
{g^2\over8m_W^4}\,J_\mu\Box J^\mu
$$
型算符。它相对 Fermi 领先项抑制
$$
{q^2\over m_W^2}.
$$
这解释了为什么 muon decay 的最低阶描述可以只用四费米子接触项，而高精度或高能过程必须恢复动量依赖。

## 9.2 Euler-Heisenberg EFT

**外部输入 9.3（低能光子自相互作用）.** 在 $E\ll m_e$ 时，电子圈图诱导纯光子有效拉氏量
$$
\Delta\mathcal L_{\gamma}
=
{\alpha^2\over90m_e^4}
\left[
(F_{\mu\nu}F^{\mu\nu})^2
+{7\over4}(F_{\mu\nu}\widetilde F^{\mu\nu})^2
\right],
$$
其中符号随 metric 和 dual tensor 约定而变，但 $1/m_e^4$ 和 $\alpha^2$ 的阶数不变。

**解释 9.4.** 四个场强的维数为 $8$，所以最低纯光子自相互作用从维数八开始。它是典型的 loop-generated EFT：没有树级电子交换图产生四光子接触项，一圈电子盒图在低能展开后给出上述局域算符。

## 9.3 手征微扰论

**外部输入 9.5（Goldstone EFT）.** QCD 的低能 pion 动力学可由手征对称性破缺
$$
SU(N_f)_L\times SU(N_f)_R\to SU(N_f)_V
$$
控制，并按动量和轻夸克质量展开。

**解释 9.6.** 手征微扰论说明 EFT 不必按 canonical dimension 单独排序；当 Goldstone 结构存在时，导数展开和手征计数更自然。最低阶 pion 拉氏量形如
$$
{\cal L}_2={f_\pi^2\over4}{\rm Tr}(\partial_\mu U^\dagger\partial^\mu U)+\cdots,
$$
其中 $U(x)\in SU(N_f)$。展开 $U=\exp(i\pi^aT^a/f_\pi)$ 后会产生无限多个 pion 相互作用，但它们由同一个对称性结构控制。

## 9.4 引力 EFT

**外部输入 9.7（广义相对论作为 EFT）.** 在低于 Planck 尺度时，引力作用量可写为
$$
S
=
\int d^4x\sqrt{-g}
\left[
\frac{M_{\mathrm{Pl}}^2}{2}R
+c_1R^2+c_2R_{\mu\nu}R^{\mu\nu}+\cdots
\right].
$$

**使用边界.** 这是 EFT 思想的广泛应用，不作为本书 SMEFT 主线的输入。

**共同结构 9.8.** 以上四个例子分别展示：

1.  Fermi 理论：树级重粒子交换；
2.  Euler-Heisenberg：圈级阈值匹配；
3.  手征微扰论：Goldstone 与导数幂计数；
4.  引力 EFT：非重整化理论也可低能预测。

这四种机制都出现在现代 SMEFT 使用中，只是自由度和对称性不同。

## 9.5 四个例子的一条主线

Fermi 理论和 Euler--Heisenberg 理论分别把树级交换与一圈阈值变成局域项；手征微扰论提醒我们 canonical dimension 未必是最合适的唯一计数；引力 EFT 则把“逐阶可预测”与“基本拉氏量可重整化”区分开。四者共同要求先选低能自由度与对称性，再按适合该系统的幂计数截断，并由匹配确定 Wilson 系数。回到 SMEFT 时，这些机制会同时出现，区别只在标准模型场表示、线性电弱对称性和具体阈值。

## 练习

**练习 9.1.** 从 $W$ 传播子展开推导 Fermi 理论中的第一个导数修正阶数。

**练习 9.2.** 说明 Euler-Heisenberg 项为什么从维数八开始。

**练习 9.3.** 比较 Fermi 理论和 Euler-Heisenberg EFT：哪一个是树级匹配，哪一个是一圈匹配？
