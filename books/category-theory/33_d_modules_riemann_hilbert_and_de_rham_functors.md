# 第三十三章：$D$-modules、Riemann-Hilbert 与 de Rham 函子

## 本章目标

本章把 perverse sheaves 与代数微分方程连接起来。$D$-modules 给出“带微分算子作用的 sheaves”；Riemann-Hilbert correspondence 说明，在复光滑代数或解析语境中，正则 holonomic $D$-modules 与可构造 sheaves 或 perverse sheaves 是同一理论的两个模型。范畴论核心是：一个代数定义的稳定范畴通过 de Rham 或 solution functor 与 sheaf 型六操作范畴等价。

## 依赖前置知识

需要 sheaf、导出范畴、perverse t-结构、六操作、Verdier 对偶、闭幺半范畴、stable $\infty$-categories 和 derived Morita 的基本语言。

## 33.1 微分算子与 $D_X$-模

**定义 33.1.** 设 $X$ 为光滑复代数簇或复流形，$\mathcal O_X$ 为结构层，$\mathcal T_X$ 为切向量场层。微分算子层 $D_X$ 是由 $\mathcal O_X$ 与 $\mathcal T_X$ 生成的 filtered sheaf of rings，满足关系

$$
\xi f-f\xi=\xi(f)
$$

其中 $f\in\mathcal O_X$，$\xi\in\mathcal T_X$。

**定义 33.2.** 左 $D_X$-module 是 sheaf $M$，它是左 $D_X$-模。右 $D_X$-module 类似。$D^b(D_X)$ 表示左 $D_X$-模的有界导出范畴；合适有限性条件下写作 $D^b_{\operatorname{hol}}(D_X)$ 或 $D^b_{\operatorname{rh}}(D_X)$。

**命题 33.3.** 左 $D_X$-module 等价于 $\mathcal O_X$-module $M$ 连同 flat connection

$$
\nabla:M\to\Omega_X^1\otimes_{\mathcal O_X}M
$$

在 $M$ 局部自由或适当准凝聚条件下成立。

**证明.** 左 $D_X$-作用限制到 $\mathcal T_X$ 给出对每个向量场 $\xi$ 的算子 $\nabla_\xi:M\to M$，满足 Leibniz 公式

$$
\nabla_\xi(fm)=\xi(f)m+f\nabla_\xi(m).
$$

这等价于 connection。$D_X$ 中向量场的 Lie bracket 关系保证曲率为零，即 connection flat。反过来，flat connection 给出 $\mathcal T_X$ 作用，与 $\mathcal O_X$ 一起按 $D_X$ 的生成关系唯一延拓为 $D_X$-作用。$\square$

## 33.2 Holonomic 与 regular holonomic

**定义 33.4.** 对 coherent $D_X$-module $M$，取 good filtration，其 associated graded 是 $\operatorname{gr}D_X\simeq\operatorname{Sym}\mathcal T_X$ 上的模。其支撑

$$
\operatorname{Char}(M)\subseteq T^*X
$$

称为 characteristic variety。

**外部输入定理 33.5（Bernstein inequality）.** 若 $M$ 是非零 coherent $D_X$-module，則

$$
\dim\operatorname{Char}(M)\ge\dim X.
$$

若等号成立，称 $M$ holonomic。

**定义 33.6.** Holonomic $D_X$-module 称为 regular holonomic，若其奇点为 regular singularities。精确定义可由曲线测试、增长条件或 $V$-filtration 给出。

**命题 33.7.** 若 $X$ 是一点，则 $D_X\simeq\mathbb C$，regular holonomic $D_X$-modules 就是有限维复向量空间。

**证明.** 点空间上 $\mathcal O_X=\mathbb C$，切向量场为零，故 $D_X=\mathbb C$。Coherent、holonomic 和 regular 条件都退化为有限维 $\mathbb C$-向量空间条件。$\square$

## 33.3 de Rham 与 solution functors

**定义 33.8.** 对左 $D_X$-module $M$，其 de Rham complex 定义为

$$
\operatorname{DR}_X(M)=
\left[
M\to\Omega_X^1\otimes M\to\cdots\to\Omega_X^{\dim X}\otimes M
\right][\dim X],
$$

微分由 connection 给出。平移约定使正则 holonomic 对象落入 perverse heart。

**定义 33.9.** Solution complex 定义为

$$
\operatorname{Sol}_X(M)=\mathbb R\mathcal Hom_{D_X}(M,\mathcal O_X)[\dim X]
$$

或按相反平移约定给出。

**命题 33.10.** 若 $M=(\mathcal O_X,d)$ 是平凡 connection，则

$$
\operatorname{DR}_X(M)\simeq\mathbb C_X[\dim X].
$$

**证明.** 该 de Rham complex 是解析 de Rham complex

$$
\mathcal O_X\to\Omega_X^1\to\cdots\to\Omega_X^{\dim X}.
$$

Poincaré lemma 说明它解析局部 quasi-isomorphic 于常值 sheaf $\mathbb C_X$。再按定义平移 $[\dim X]$ 得结论。$\square$

**外部输入定理 33.11（Riemann-Hilbert correspondence）.** 对复光滑代数簇或复解析流形 $X$，de Rham 或 solution functor 给出正则 holonomic $D_X$-modules 的导出范畴与可构造 sheaves 的导出范畴之间的等价：

$$
D^b_{\operatorname{rh}}(D_X)\simeq D^b_c(X,\mathbb C).
$$

在 heart 层面，regular holonomic $D_X$-modules 对应 perverse sheaves。

## 33.4 六操作与 $D$-modules

**外部输入定理 33.12.** $D$-modules 具有六操作形式主义。对合适态射 $f:X\to Y$，存在

$$
f^!,\quad f_*,\quad f_+,\quad f^\dagger
$$

等变体；在不同左/右模和 de Rham/solution 约定下，它们与 sheaf 侧的 $f^*,f_*,f_!,f^!$ 相容。

**命题 33.13.** 若 $f:X\to Y$ proper，则 Riemann-Hilbert 等价把 $D$-module 直接像与 sheaf 直接像对应。

**证明.** 这是定理 33.11 与定理 33.12 的相容性特例。Proper 情形下 sheaf 侧 $f_!=f_*$，$D$-module 侧 proper direct image 保持 regular holonomic 对象。Riemann-Hilbert 等价是六操作相容的，因此两侧直接像在等价下对应。$\square$

## 33.5 Kashiwara 等价与闭嵌入

**外部输入定理 33.14（Kashiwara equivalence）.** 若 $i:Z\hookrightarrow X$ 是闭嵌入，则支撑在 $Z$ 上的 $D_X$-modules 范畴等价于 $D_Z$-modules 范畴。导出和 holonomic 版本同样成立。

**命题 33.15.** Kashiwara 等价是 recollement 中闭支撑部分的 $D$-module 版本。

**证明.** Sheaf 侧闭嵌入 $i$ 的 $i_*$ 把 $D(Z)$ 全忠实嵌入到支撑在 $Z$ 的对象。$D$-module 侧 Kashiwara 等价也把 $D_Z$-modules 识别为 $X$ 上支撑在 $Z$ 的 $D_X$-modules。Riemann-Hilbert 相容性把两种描述对应起来，因此它是 recollement 闭部分的代数模型。$\square$

## 33.6 本章小结

$D_X$-modules 把微分方程编码为微分算子层的模。Holonomic 条件是有限性条件；regular holonomic 条件控制奇点。de Rham 和 solution functors 把 $D$-modules 送到可构造 sheaves，Riemann-Hilbert correspondence 则给出 regular holonomic $D$-modules 与 perverse sheaves 的等价。Kashiwara 等价和六操作相容性说明 $D$-modules 是 sheaf 六操作理论的代数模型。

## 练习

**练习 33.1.** 定义微分算子层 $D_X$。

**练习 33.2.** 说明左 $D_X$-module 与 flat connection 的关系。

**练习 33.3.** 定义 characteristic variety。

**练习 33.4.** 陈述 Bernstein inequality 和 holonomic 条件。

**练习 33.5.** 解释一点空间上的 regular holonomic $D$-modules。

**练习 33.6.** 定义 de Rham complex $\operatorname{DR}_X(M)$。

**练习 33.7.** 计算平凡 connection 的 de Rham complex。

**练习 33.8.** 陈述 Riemann-Hilbert correspondence。

**练习 33.9.** 说明 regular holonomic $D$-modules 与 perverse sheaves 的 heart 层对应。

**练习 33.10.** 说明 proper direct image 与 Riemann-Hilbert 的相容性。

**练习 33.11.** 陈述 Kashiwara equivalence。

**练习 33.12.** 解释 Kashiwara equivalence 与 recollement 闭支撑部分的关系。
