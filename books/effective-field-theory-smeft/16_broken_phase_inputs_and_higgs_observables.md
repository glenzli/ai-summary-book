# 第十六章：电弱破缺相、输入参数与 Higgs 可观测量

## 本章目标

本章说明从规范不变 SMEFT 到实际 Higgs 可观测量需要经过电弱破缺、场重定义和输入参数选择。本章给出一个可复核的 $hgg$ 修正推导。

## 依赖前置知识

需要第十一章的 Higgs 入口和第十三章的 $\mathcal O_{HG}$。

## 16.1 电弱破缺

采用 unitary gauge 记号
$$
H=\frac{1}{\sqrt2}
\begin{pmatrix}
0\\ v+h
\end{pmatrix}.
$$

**命题 16.1（$\mathcal O_{HG}$ 的破缺相展开）.** 对
$$
\mathcal O_{HG}=H^\dagger H\,G_{\mu\nu}^A G^{A\mu\nu},
$$
有
$$
\frac{C_{HG}}{\Lambda^2}\mathcal O_{HG}
=
\frac{C_{HG}}{\Lambda^2}
\left(
\frac{v^2}{2}+vh+\frac{h^2}{2}
\right)
G_{\mu\nu}^A G^{A\mu\nu}.
$$

**证明（书内推导）.** 由 $H^\dagger H=(v+h)^2/2$ 直接展开。$\square$

**警告 16.2（规范动能归一化）.** $v^2G^2$ 项修正规范场动能项。严格计算物理振幅前需做场重定义和耦合重定义。在线性 $1/\Lambda^2$ 阶，这些修正可系统吸收到输入参数和 Wilson 系数组合中。

**场重定义 16.3.** 规范动能项变为
$$
-{1\over4}G_{\mu\nu}^AG^{A\mu\nu}
+{C_{HG}v^2\over2\Lambda^2}G_{\mu\nu}^AG^{A\mu\nu}
=-{1\over4}\left(1-{2C_{HG}v^2\over\Lambda^2}\right)G^2.
$$
令
$$
G_\mu^A\mapsto
\left(1+{C_{HG}v^2\over\Lambda^2}\right)G_\mu^A
$$
可把动能项恢复到规范形式到线性阶。这个重定义同时移动 $g_s$ 和所有 gluon 顶点，因此实际 Higgs-gluon contact 应与输入方案一起处理。

## 16.2 输入参数方案

**定义 16.4（输入参数方案）.** 输入参数方案是用一组选定实验量确定拉氏量参数的规则。常见选择包括
$$
\{\alpha_{\mathrm{em}},m_Z,G_F\},
\qquad
\{m_W,m_Z,G_F\}.
$$

**原则 16.5.** SMEFT 预测必须说明输入方案，因为高维算符会修正输入量与拉氏量参数之间的关系。

## 16.3 $h\to gg$ 的线性 SMEFT 结构

**定义 16.6.** 写振幅为
$$
\mathcal A(h\to gg)
=
\mathcal A_{\mathrm{SM}}^{\mathrm{loop}}
+
\frac{C_{HG}}{\Lambda^2}\mathcal A_{HG}^{\mathrm{tree}}
+\cdots.
$$

**命题 16.7（宽度的线性修正）.** 在线性 $1/\Lambda^2$ 阶，
$$
\frac{\Gamma(h\to gg)}{\Gamma(h\to gg)_{\mathrm{SM}}}
=
1+
2\,\mathrm{Re}
\left(
\frac{C_{HG}}{\Lambda^2}
\frac{\mathcal A_{HG}^{\mathrm{tree}}}{\mathcal A_{\mathrm{SM}}^{\mathrm{loop}}}
\right)
+O(\Lambda^{-4}).
$$

**证明（书内推导）.** 令 $\mathcal A=\mathcal A_0+\delta\mathcal A$。则
$$
|\mathcal A|^2
=|\mathcal A_0|^2
+2\mathrm{Re}(\mathcal A_0^\ast\delta\mathcal A)
+O(\delta\mathcal A^2).
$$
相空间因子在线性近似下同乘，除以 SM 宽度即得。$\square$

**使用边界 16.8.** 精确数值还需 SM top-loop 函数、QCD K-factor、输入方案和其他算符贡献。本章只给出结构推导。

## 16.4 从 Wilson 系数到宽度的步骤

对 $h\to gg$ 的线性 SMEFT 预测，完整步骤为：

1.  在 Warsaw basis 写出 $C_{HG}(\mu)$；
2.  运行到 Higgs 尺度附近；
3.  展开 $H^\dagger H$；
4.  规范化 gluon 动能项；
5.  写出 $hG_{\mu\nu}^AG^{A\mu\nu}$ 顶点；
6.  与 SM top-loop 振幅干涉；
7.  乘以相空间和 QCD 修正；
8.  报告输入方案和理论误差。

若同时开启 $C_{uH}$ 或 top dipole，步骤 6 的 SM loop 振幅也被修改，因此 $C_{HG}$ 不能从 $h\to gg$ 单独提取。

## 本章小结

规范不变 SMEFT 算符进入可观测量前，必须处理破缺相展开、场归一化、输入参数和振幅干涉。$\mathcal O_{HG}$ 是 Higgs-gluon 物理中最直接的例子。

## 练习

**练习 16.1.** 展开 $\mathcal O_{HB}$ 并写出含一个 Higgs 场的项。

**练习 16.2.** 从 $|\mathcal A_0+\delta\mathcal A|^2$ 推导线性干涉公式。

**练习 16.3.** 验证场重定义 16.3 在线性阶恢复 gluon 动能项的规范归一化。
