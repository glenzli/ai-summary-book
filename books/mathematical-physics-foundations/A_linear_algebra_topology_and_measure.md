# 附录 A：线性代数、拓扑与测度工具

本附录收集正文反复使用但不宜在每章重证的基础工具。其目的不是替代线性代数、点集拓扑或测度论教材，而是固定本书的最低使用口径：哪些结论可在有限维内直接证明，哪些结论作为外部输入。

## 线性代数

**定义 A.1.** 复内积空间中，算符 $N$ 称为正规，如果 $NN^*=N^*N$。

**命题 A.1 (`P`, 有限维谱分解).** 有限维复内积空间上的正规算符存在标准正交本征基。

**证明.** 对维数归纳。维数为零时结论成立。若维数正，由复数域上的特征多项式有根，取单位本征向量 $v$，$Nv=\lambda v$。算符 $T=N-\lambda I$ 仍正规，而且对每个 $x$，
$$
\|T^*x\|^2-\|Tx\|^2
=\langle x,(TT^*-T^*T)x\rangle=0.
$$
所以 $Tv=0$ 蕴含 $T^*v=0$，即 $N^*v=\overline\lambda v$。若 $w\perp v$，则
$$
\langle Nw,v\rangle
=\langle w,N^*v\rangle
=\overline\lambda\langle w,v\rangle=0,
$$
故 $Nw\perp v$；同理 $N^*w\perp v$。因此 $v^\perp$ 同时在 $N,N^*$ 下不变，且限制算符仍正规。对维数少一的 $v^\perp$ 使用归纳假设，取得其标准正交本征基；与 $v$ 合并即得全空间的标准正交本征基。$\square$

**定义 A.2.** 对称双线性型 $B$ 的 signature 是正、负、零惯性指数三元组。Minkowski 度量在本书中取 mostly plus：
$$
\eta=\operatorname{diag}(-,+,\ldots,+).
$$

## 拓扑和上同调

**定义 A.3.** 链复形是对象列 $C_k$ 与边界算子 $\partial_k:C_k\to C_{k-1}$，满足 $\partial_{k-1}\partial_k=0$。同调为
$$
H_k(C)=\ker\partial_k/\operatorname{im}\partial_{k+1}.
$$

**定理 A.2 (`E`, de Rham 定理).** 光滑流形的 de Rham 上同调与实系数奇异上同调自然同构。

**外部输入边界.** 正文使用该定理解释闭形式积分的拓扑不变量含义；不证明奇异上同调与微分形式之间的链同伦构造。所用自然同构版本见 [SOURCES.md](SOURCES.md) 的 \`E-A.2\`。

**定义 A.4.** 若拓扑存在可数基，则空间称为第二可数。若每个开覆盖都有可数子覆盖，则空间称为 Lindelöf。第二可数空间必为 Lindelöf：对给定开覆盖，从每个被某个覆盖成员包含的基元素选择一个覆盖成员；所选集合至多可数并仍覆盖全空间。反向在一般拓扑空间中不成立。本书按通常定义默认流形 Hausdorff、第二可数；分割单位还要结合局部 Euclidean 性所推出的仿紧性使用。

## 测度和积分

**定义 A.5.** 测度空间 $(X,\Sigma,\mu)$ 上的 $L^p$ 空间由满足
$$
\|f\|_p=\left(\int_X |f|^p\,d\mu\right)^{1/p}<\infty
$$
的可测函数按几乎处处相等等价得到。

**外部输入 A.3 (`E`).** Fubini-Tonelli 定理、Radon-Nikodym 定理和 Riesz 表示定理作为测度论外部输入使用。

**外部输入边界.** 正文只在 Fourier 分析、Hilbert 空间和路径积分有限维模型中使用这些结论；无穷维路径积分不由本附录测度论自动推出。三个定理的假设与来源分别登记在 [SOURCES.md](SOURCES.md) 的 \`E-A.3a--c\`。

## 常用恒等式

1. Jacobi 恒等式：
   $$
   [X,[Y,Z]]+[Y,[Z,X]]+[Z,[X,Y]]=0.
   $$
2. 分部积分：
   $$
   \int_\Omega \partial_\mu V^\mu\,d^dx=\int_{\partial\Omega}V^\mu n_\mu\,d\Sigma.
   $$
3. Fourier 反演按 [NOTATION.md](NOTATION.md) 的约定使用。
