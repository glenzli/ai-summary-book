# 第七章：谐振子、升降算符与 Hermite 基

## 本章目标

本章计算一维量子谐振子的能谱、基态、升降算符和数算符结构。

## 依赖前置知识

需要正则对易关系、微分方程和 Hilbert 空间正交分解。

## 7.1 Hamiltonian 与升降算符

**定义 7.1.** 在 $\hbar=1$ 下，取
$\mathcal H=L^2(\mathbb R)$、$m,\omega>0$。位置与动量算子取其标准
自伴实现
$$
\mathcal D(X)=\{\psi\in L^2:x\psi\in L^2\},
\qquad
\mathcal D(P)=H^1(\mathbb R),
\qquad P=-i\frac d{dx}.
$$
令 $\mathcal D_1=\mathcal D(X)\cap\mathcal D(P)$，定义闭算子
$$
a=\sqrt{\frac{m\omega}{2}}X+\frac{i}{\sqrt{2m\omega}}P,\qquad
a^*=\sqrt{\frac{m\omega}{2}}X-\frac{i}{\sqrt{2m\omega}}P.
$$
二者的定义域均为 $\mathcal D_1$，并互为 Hilbert 空间伴随；这里的
$a^*$ 不是只在形式上把 $i$ 变号。Schwartz 空间
$\mathscr S(\mathbb R)$ 稠密、在 $X,P,a,a^*$ 下不变，并是
$a,a^*$ 的共同算子核心。因此它也是二者的共同不变稠密域，下面所有
代数计算都先在该域上进行。

这些定义域断言可由如下图范数计算检查。对
$\psi\in\mathscr S(\mathbb R)$ 分部积分得到
$$
\begin{aligned}
\|a\psi\|^2
&=\frac{m\omega}{2}\|x\psi\|^2
+\frac{1}{2m\omega}\|\psi'\|^2-\frac12\|\psi\|^2,\\
\|a^*\psi\|^2
&=\frac{m\omega}{2}\|x\psi\|^2
+\frac{1}{2m\omega}\|\psi'\|^2+\frac12\|\psi\|^2.
\end{aligned}
$$
把 $\|\psi\|^2$ 加到两式可见，相应图范数与
$$
\|\psi\|+\|x\psi\|+\|\psi'\|
$$
给出的范数等价。截断后再磨光表明 $\mathscr S(\mathbb R)$ 在
$\mathcal D_1$ 的这一范数中稠密；分部积分以及对伴随定义域中向量作
同样的截断--磨光逼近，便得到上述闭性、伴随关系和核心结论。

**命题 7.2.** 在共同不变核心上，
$$
[a,a^*]=I,
\qquad
\frac{P^2}{2m}+\frac12m\omega^2X^2
=\omega\left(a^*a+\frac12I\right).
$$
这里以及下文未另作说明的无界算子交换子都先在
$\mathscr S(\mathbb R)$ 上理解。

**证明.** 使用 $[X,P]=iI$。令 $\alpha=\sqrt{m\omega/2}$、$\beta=1/\sqrt{2m\omega}$。则
$$
[a,a^*]=[\alpha X+i\beta P,\alpha X-i\beta P]
=-i\alpha\beta[X,P]+i\alpha\beta[P,X]=2\alpha\beta I=I.
$$
展开 $a^*a$：
$$
a^*a=\alpha^2X^2+\beta^2P^2+i\alpha\beta(XP-PX)
=\frac{m\omega}{2}X^2+\frac{1}{2m\omega}P^2-\frac12I.
$$
乘以 $\omega$ 得 Hamiltonian 公式。$\square$

## 7.2 能级与数算符定义域

**定义 7.3.** 数算符为
$$
N=a^*a,
\qquad
\mathcal D(N)
=\{\psi\in\mathcal D(a):a\psi\in\mathcal D(a^*)\}.
$$
由于 $a$ 稠定且闭，$N=a^*a$ 是非负自伴算子。谐振子
Hamiltonian 定义为
$$
H=\omega\left(N+\frac12I\right),
\qquad \mathcal D(H)=\mathcal D(N).
$$
命题 7.2 表明它在 $\mathscr S(\mathbb R)$ 上等于通常的二阶微分
表达式。外部输入定理 7.10 给出 Hermite 完备性；推论 7.10A 再由
加权移位计算得到 $a,a^*,N,H$ 的精确定义域和共同不变核心。
若 $N\psi=n\psi$，称 $\psi$ 为数态。

**命题 7.4.** 若 $\psi\in\mathcal D(N)$、$N\psi=n\psi$ 且
$\|\psi\|=1$，则 $n\ge0$。若进一步有
$\psi\in\mathscr S(\mathbb R)$，则 $a\psi,a^*\psi\in\mathcal D(N)$，
并且：当 $a\psi\ne0$ 时，它是本征值 $n-1$ 的本征向量；当
$a^*\psi\ne0$ 时，它是本征值 $n+1$ 的本征向量。

**证明.** 首先
$$
n=\langle\psi,N\psi\rangle=\langle a\psi,a\psi\rangle=\|a\psi\|^2\ge0.
$$
在额外的 Schwartz 假设下，所有乘积都作用在共同不变域上。由
$[N,a]=-a$ 与 $[N,a^*]=a^*$，
$$
N(a\psi)=aN\psi-a\psi=(n-1)a\psi,
$$
并且
$$
N(a^*\psi)=a^*N\psi+a^*\psi=(n+1)a^*\psi.
$$
$\psi\in\mathcal D(N)$ 本身只足以推出非负性，并不自动保证
$a\psi$ 或 $a^*\psi$ 属于 $\mathcal D(N)$；这正是附加共同不变域
假设的作用。
$\square$

**命题 7.5（共同核心上的本征值量子化）.** 设
$\psi\in\mathscr S(\mathbb R)$，且 $N\psi=n\psi$、$\|\psi\|=1$。
则 $n\in\mathbb Z_{\ge0}$，相应能量只能为
$$
E_n=\omega\left(n+\frac12\right),\qquad n=0,1,2,\dots.
$$

**证明.** 由命题 7.4，$n\ge0$。对每个 $k\ge1$，反复使用
$aa^*=a^*a+I$ 得
$$
\|a^k\psi\|^2=n(n-1)\cdots(n-k+1).
$$
若 $n$ 不是非负整数，取 $k=\lfloor n\rfloor+2$，右端恰有一个负
因子而其余因子为正，矛盾于范数平方非负。因此 $n$ 是非负整数。
最后代入 $H=\omega(N+I/2)$。$\square$

## 7.3 基态波函数

**命题 7.6.** 基态满足 $a\psi_0=0$，在位置表象中可取
$$
\psi_0(x)=\left(\frac{m\omega}{\pi}\right)^{1/4}e^{-m\omega x^2/2}.
$$

**证明.** 在位置表象 $P=-i\,d/dx$，方程 $a\psi_0=0$ 化为
$$
\sqrt{\frac{m\omega}{2}}x\psi_0+\frac{1}{\sqrt{2m\omega}}\psi_0'=0.
$$
即 $\psi_0'=-m\omega x\psi_0$，解为 $Ce^{-m\omega x^2/2}$。归一化常数由 Gaussian 积分
$$
\int_{\mathbb R}e^{-m\omega x^2}\,dx=\sqrt{\frac{\pi}{m\omega}}
$$
确定。$\square$

**命题 7.6A（Hermite 数态）.** 对 $n\ge0$ 定义
$$
|n\rangle=\frac{(a^*)^n}{\sqrt{n!}}\psi_0.
$$
则 $|n\rangle\in\mathscr S(\mathbb R)$，并且
$$
N|n\rangle=n|n\rangle,
\qquad
a|n\rangle=\sqrt n\,|n-1\rangle,
\qquad
a^*|n\rangle=\sqrt{n+1}\,|n+1\rangle,
$$
其中 $a|0\rangle=0$。该族正交归一。

**证明.** $\psi_0$ 是 Schwartz 函数，而 $a^*$ 保持 Schwartz 空间，
故所有乘积均有定义。由 $a\psi_0=0$ 和 $aa^*=a^*a+I$ 归纳得到
$$
a(a^*)^n\psi_0=n(a^*)^{n-1}\psi_0,
\qquad
\|(a^*)^n\psi_0\|^2=n!.
$$
这给出归一化和两个升降公式，并由 $N=a^*a$ 得
$N|n\rangle=n|n\rangle$。$N$ 自伴，不同本征值的本征向量正交，
故该族正交归一。$\square$

## 7.4 相干态

**定义 7.7.** 谐振子的相干态是湮灭算符 $a$ 的本征态：
$$
a|\alpha\rangle=\alpha|\alpha\rangle,\qquad \alpha\in\mathbb C.
$$
具体地，下面的正交级数按 $L^2$ 范数收敛，并定义相干态
$$
|\alpha\rangle=e^{-|\alpha|^2/2}\sum_{n=0}^\infty\frac{\alpha^n}{\sqrt{n!}}|n\rangle.
$$

**命题 7.8.** 上述级数定义归一化态，并满足 $a|\alpha\rangle=\alpha|\alpha\rangle$。

**证明.** 正交归一性给出范数
$$
e^{-|\alpha|^2}\sum_{n=0}^\infty\frac{|\alpha|^{2n}}{n!}=1.
$$
令级数的第 $K$ 个部分和为 $\psi_K$。命题 7.6A 给出
$$
a\psi_K
=e^{-|\alpha|^2/2}\sum_{n=1}^K
\frac{\alpha^n\sqrt n}{\sqrt{n!}}|n-1\rangle
=\alpha e^{-|\alpha|^2/2}\sum_{m=0}^{K-1}
\frac{\alpha^m}{\sqrt{m!}}|m\rangle,
$$
右端收敛到 $\alpha|\alpha\rangle$，而
$\psi_K\to|\alpha\rangle$。由于 $a$ 闭，
$|\alpha\rangle\in\mathcal D(a)$ 且
$a|\alpha\rangle=\alpha|\alpha\rangle$。$\square$

**说明 7.9.** 相干态在谐振子演化下保持相干态形式，中心按经典谐振子轨道运动。因此它们是连接量子波包与经典极限的标准例子。

**外部输入定理 7.10（Hermite 函数完备性，QM-EXT-14）.**
命题 7.6A 的 Hermite 函数族 $\{|n\rangle\}_{n\ge0}$ 构成
$L^2(\mathbb R)$ 的正交归一基。来源定位为 Teschl,
*Mathematical Methods in Quantum Mechanics*, 2nd ed., §8.3,
Theorem 8.5 的一维部分；一般的 $m,\omega>0$ 由酉伸缩得到。本书不重证
Hermite 完备性。

**推论 7.10A（定义域与共同不变核心）.** 令
$$
\mathscr D_{\mathrm{fin}}
=\operatorname{span}\{|n\rangle:n\ge0\}.
$$
则该空间在 $a,a^*,N,H$ 下不变，并是这四个闭算子的共同核心。
若 $\psi=\sum_{n\ge0}c_n|n\rangle$，则
$$
\begin{aligned}
\mathcal D(a)
&=\left\{\sum_{n\ge0}c_n|n\rangle:
\sum_{n\ge0}n|c_n|^2<\infty\right\},\\
\mathcal D(a^*)
&=\left\{\sum_{n\ge0}c_n|n\rangle:
\sum_{n\ge0}(n+1)|c_n|^2<\infty\right\}
=\mathcal D(a),\\
\mathcal D(N)=\mathcal D(H)
&=\left\{\sum_{n\ge0}c_n|n\rangle:
\sum_{n\ge0}n^2|c_n|^2<\infty\right\}.
\end{aligned}
$$
这里每个集合都已隐含 $\sum_n|c_n|^2<\infty$，因为展开表示
$L^2$ 中的向量。

**证明.** 命题 7.6A 给出
$$
a|n\rangle=\sqrt n\,|n-1\rangle,
\qquad
a^*|n\rangle=\sqrt{n+1}\,|n+1\rangle.
$$
若 $\psi\in\mathcal D(a)$，由伴随关系逐项取 Hermite 系数并用
Parseval 等式，
$$
\|a\psi\|^2=\sum_{n\ge0}n|c_n|^2.
$$
反之，若右端有限，则部分和
$\psi_K=\sum_{n=0}^Kc_n|n\rangle$ 满足
$\psi_K\to\psi$ 且 $a\psi_K$ 收敛；$a$ 的闭性给出
$\psi\in\mathcal D(a)$。对 $a^*$ 作同样论证得到
$$
\|a^*\psi\|^2=\sum_{n\ge0}(n+1)|c_n|^2
$$
及所列定义域。再由
$\mathcal D(N)=\{\psi\in\mathcal D(a):a\psi\in\mathcal D(a^*)\}$
得到 $\sum n^2|c_n|^2<\infty$；$H=\omega(N+I/2)$ 与 $N$
具有同一定义域。对任一上述算子，Hermite 部分和还按相应图范数
收敛，所以 $\mathscr D_{\mathrm{fin}}$ 是共同核心；升降公式又说明它
共同不变。由于
$\mathscr D_{\mathrm{fin}}\subset\mathscr S(\mathbb R)$ 且
$\mathscr S(\mathbb R)$ 对四个算子共同不变，夹在核心与算子定义域
之间的准则也表明 $\mathscr S(\mathbb R)$ 是共同不变核心。$\square$

**推论 7.10B（完整谱）.** 一维谐振子自伴 Hamiltonian 的谱恰为
$\{\omega(n+\tfrac12):n\in\mathbb Z_{\ge0}\}$，每个本征值重数为一。

**证明.** 在外部输入定理 7.10 给出的正交归一基中，
$$
H\sum_{n\ge0}c_n|n\rangle
=\sum_{n\ge0}\omega\left(n+\frac12\right)c_n|n\rangle
$$
在其加权平方可和定义域上成立。因此 $H$ 是对角自伴算子，谱为对角
值集合的闭包。该集合没有有限聚点，故谱恰为所列集合；每个对角值只
对应一个基向量，重数为一。$\square$

## 本章小结

谐振子是量子力学中最重要的可解模型。升降算符把二阶微分算子谱问题转化为代数问题，给出等间距能级、零点能和 Hermite 函数基。

## 练习

**练习 7.1.** 证明 $[N,a]=-a$ 与 $[N,a^*]=a^*$。

**练习 7.2.** 恢复 $\hbar$，写出谐振子能级。
