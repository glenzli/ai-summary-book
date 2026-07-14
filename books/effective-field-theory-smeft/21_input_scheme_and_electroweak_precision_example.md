# 第二十一章：输入参数方案与电弱精密例子

实验用 $\alpha$、$G_F$ 和 $m_Z$ 表示电弱输入，拉氏量却以 $g,g',v$ 为参数。SMEFT 算符会直接改写三个输入关系，所以在保持测得输入不变时，基本参数必须产生补偿位移；这些位移又会传播到并未选作输入的 $m_W$。这不是符号重命名，而是一个可解的线性系统。以 $\{\alpha,G_F,m_Z\}$ 方案为例，把算符的直接贡献压缩为 $\epsilon_\alpha,\epsilon_G,\epsilon_Z$，固定输入后反解 $\delta g/g$、$\delta g'/g'$ 与 $\delta v/v$，再加入 $m_W$ 关系的直接项 $\epsilon_W$。所得响应行清楚地区分“通过输入反解进入”的间接效应与观测量顶点自身的直接修正，也解释为何换输入方案会改变单个 Wilson 系数前的数值。

## 21.1 输入方案的数学形式

取三个输入量
$$
\{\alpha,\,G_F,\,m_Z\}.
$$
在树级标准模型中
$$
\alpha={e^2\over4\pi},\qquad
G_F={1\over\sqrt2 v^2},\qquad
m_Z^2={g^2+g'^2\over4}v^2,
$$
且
$$
e={gg'\over\sqrt{g^2+g'^2}}.
$$
定义
$$
s^2={g'^2\over g^2+g'^2},\qquad
c^2={g^2\over g^2+g'^2}.
$$

**定义 21.1（输入修正）.** 设 SMEFT 直接修正三个输入关系为
$$
{\delta\alpha\over\alpha}
=2\left(s^2{\delta g\over g}+c^2{\delta g'\over g'}\right)+\epsilon_\alpha,
$$
$$
{\delta G_F\over G_F}
=-2{\delta v\over v}+\epsilon_G,
$$
$$
{\delta m_Z^2\over m_Z^2}
=2\left(c^2{\delta g\over g}+s^2{\delta g'\over g'}\right)
 +2{\delta v\over v}+\epsilon_Z.
$$
这里 $\epsilon_\alpha,\epsilon_G,\epsilon_Z$ 是具体 SMEFT 算符对输入定义的直接贡献。
它们是无量纲的 $p=2$ 量，典型地含
$v^2C_i^{(6)}(\mu)/\Lambda_{\rm ref}^2$；本节在线性维数六和树级电弱关系下工作，
忽略 $p\ge4$ 与未列出的圈修正。

**备注 21.2.** 在 Warsaw basis 常见树级分析中，$\epsilon_G$ 来自 muon decay 中的四轻子和 lepton-current 算符，$\epsilon_Z$ 接收 ${\cal O}_{HD}$、${\cal O}_{HWB}$ 及场重定义影响。不同文献的符号约定会改变 $\epsilon$ 的显式表达，但不改变本章线性代数。

## 21.2 固定输入量

输入量被实验固定，因此对输入关系取变分时设
$$
{\delta\alpha\over\alpha}
={\delta G_F\over G_F}
={\delta m_Z^2\over m_Z^2}=0.
$$
令
$$
x={\delta g\over g},\qquad
y={\delta g'\over g'},\qquad
z={\delta v\over v}.
$$
方程组为
$$
2(s^2x+c^2y)+\epsilon_\alpha=0,
$$
$$
-2z+\epsilon_G=0,
$$
$$
2(c^2x+s^2y)+2z+\epsilon_Z=0.
$$

**命题 21.3（参数位移解）.** 若 $c^2\ne s^2$，则
$$
z={\epsilon_G\over2},
$$
$$
x={s^2\epsilon_\alpha-c^2(\epsilon_Z+\epsilon_G)
\over 2(c^2-s^2)},
$$
$$
y={s^2(\epsilon_Z+\epsilon_G)-c^2\epsilon_\alpha
\over 2(c^2-s^2)}.
$$

**证明.** 第二个方程立即给出 $z=\epsilon_G/2$。前两个 gauge-coupling 方程为
$$
s^2x+c^2y=-{\epsilon_\alpha\over2},
$$
$$
c^2x+s^2y=-{\epsilon_Z+\epsilon_G\over2}.
$$
解这个 $2\times2$ 线性系统，并使用行列式 $s^4-c^4=-(c^2-s^2)$，即得结果。$\square$

## 21.3 $m_W$ 预测

在树级标准模型中
$$
m_W^2={g^2v^2\over4}.
$$
设 SMEFT 对 $m_W$ 关系还有直接项 $\epsilon_W$：
$$
{\delta m_W^2\over m_W^2}
=2{\delta g\over g}+2{\delta v\over v}+\epsilon_W.
$$
代入命题 21.3 得
$$
{\delta m_W^2\over m_W^2}
={s^2\epsilon_\alpha-c^2\epsilon_Z-s^2\epsilon_G\over c^2-s^2}
+\epsilon_W.
$$

**结论 21.4.** 一个 Wilson 系数若只修正输入量，也能改变 $m_W$ 预测。输入方案不是后处理细节，而是 SMEFT 预测的一部分。

## 21.4 与 Warsaw 算符的接口

在 Warsaw basis 下，常见树级路径为：

1.  从第十三章算符表选出会改写 muon decay、neutral gauge kinetic/mass terms 和 charged-current vertices 的算符；
2.  展开到破缺相并做 kinetic normalization；
3.  把结果写成 $\epsilon_\alpha,\epsilon_G,\epsilon_Z,\epsilon_W$；
4.  使用本章线性系统传播到预测量；
5.  再计算观测量自身的直接 vertex 或 contact 修正。

**警告 21.5.** 只给出一个“$C_i$ 对 $m_W$ 的系数”而不说明输入方案，是不完整结果。换用 $\{m_W,m_Z,G_F\}$ 或 $\{\alpha,m_Z,G_F\}$ 会改变中间 Wilson 组合。

## 21.5 从输入位移到可比较预测

固定 $\{\alpha,G_F,m_Z\}$ 后，$g,g',v$ 的位移不是额外自由参数，而是由三个输入方程唯一反解。$m_W$ 的响应同时含这些间接位移和 $\epsilon_W$ 代表的直接修正；换用 $\{m_W,m_Z,G_F\}$ 会重新分配二者，却不会改变用完整同阶关系计算的物理预测。因而 Wilson 系数前的单个数值只有连同输入集、场归一化、定义尺度与截断一起才可比较。

## 练习

**练习 21.1.** 从 $e=gg'/\sqrt{g^2+g'^2}$ 推导 $\delta\alpha/\alpha$ 的线性表达式。

**练习 21.2.** 在 $\epsilon_\alpha=\epsilon_W=0$ 时写出 $\delta m_W^2/m_W^2$。

**练习 21.3.** 解释为什么同一算符在不同输入方案下的数值限制不能直接比较。
