# 第二十章：Coulomb branches、BFN 构造与量子化

Affine Grassmannian 的卷积只记录 $G$-bundle 的相对位置；BFN 构造再加入 $G$-表示 $N$ 的 section，并要求它在修改前后都 regular。这个额外闭条件把每个 coweight 分支的几何改变掉，其 equivariant Borel--Moore homology 形成交换代数，谱定义为 Coulomb branch；加入 loop rotation 后，旋转参数 $\hbar$ 产生非交换量子化。构造发生在非有限型空间上，所以有限维近似与卷积 formalism 是不可省略的外部输入。为看见 matter 条件究竟做了什么，先完整计算 $G=\mathbb C^\times,N=\mathbb C$：第 $m$ 个分支上的 section 必须从次数 $\max(m,0)$ 开始，这一消失阶数正是纯 torus group-algebra 计算失效的来源。

## 20.1 BFN 输入数据

**定义 20.1.** BFN 构造的基本输入为复 reductive group $G$ 和有限维 $G$-representation $N$。令
$$
\mathcal O=\mathbb C[[z]],\qquad \mathcal K=\mathbb C((z)).
$$
定义 affine Grassmannian
$$
\operatorname{Gr}_G=G(\mathcal K)/G(\mathcal O).
$$

**定义 20.2.** BFN space $\mathcal R$ 的点可理解为三元组
$$
(\mathcal P,\varphi,s)
$$
其中 $\mathcal P$ 是 formal disk 上的 $G$-bundle，$\varphi$ 是 punctured disk 上的平凡化，$s$ 是 associated $N$-bundle 的 section，并在 $\varphi$ 下满足 regularity 条件。

**警告 20.3.** $\mathcal R$ 通常不是有限型 scheme。其 Borel-Moore homology 的定义需要有限维近似、limit formalism 或 BFN 的专门构造。

**定义 20.4.** 在平凡化后，可把 $\mathcal R$ 形式写成
$$
\mathcal R=\{(g,s)\in G(\mathcal K)\times N(\mathcal O)\mid g^{-1}s\in N(\mathcal O)\}/G(\mathcal O),
$$
其中 $g$ 表示 $G$-bundle 的 punctured disk 平凡化相对位置。这个公式用于直观和计算；严格定义应在相应 ind-scheme/prestack 模型中给出。

**例 20.5.** 若 $N=0$，regularity 条件消失，$\mathcal R$ 退化为 affine Grassmannian 型对象。此时 BFN 代数与 $G$ 的 affine Grassmannian equivariant Borel--Moore homology 相关。具体结果依 $G$ 和等变参数而变，不能仅由本例推出一般 Coulomb branch 的有限性。

**例 20.5.1.** 若 $G=T$ 是 algebraic torus 且 $N=0$，则
$$
\operatorname{Gr}_T=T(\mathcal K)/T(\mathcal O)\simeq X_\ast(T)
$$
为离散 ind-scheme。点 $\lambda\in X_\ast(T)$ 由 loop $z^\lambda$ 表示。此时 $\mathcal R=\operatorname{Gr}_T$，卷积对应 coweight 加法。

**命题 20.5.2.** 在例 20.5.1 中，若以 $u_\lambda$ 表示 $\lambda\in X_\ast(T)$ 对应连通分支的 equivariant fundamental class，则
$$
u_\lambda\star u_\mu=u_{\lambda+\mu}.
$$
因此作为 $H_T^\ast(\mathrm{pt})$-module，
$$
H_\ast^{T(\mathcal O)}(\mathcal R)
\simeq H_T^\ast(\mathrm{pt})\otimes_\mathbb C \mathbb C[X_\ast(T)]
$$
并且乘法在 group algebra 因子上由 lattice 加法给出。

**证明.** $T$ 交换，且
$$
z^\lambda z^\mu=z^{\lambda+\mu}.
$$
卷积 correspondence 在离散点集上退化为图
$$
X_\ast(T)\times X_\ast(T)\longrightarrow X_\ast(T),\qquad
(\lambda,\mu)\mapsto\lambda+\mu.
$$
每个分支都是一点，Borel-Moore fundamental class 的 pull-intersect-push 只把两个点的类推到其和。等变系数来自稳定群 $T(\mathcal O)$ 与 $T$ 同伦等价的标准识别。$\square$

**例 20.5.3.** 若 $G=\mathbb C^\times$，$N=\mathbb C$ 为标准权 $1$ 表示，则 $\operatorname{Gr}_G\simeq\mathbb Z$。对点 $(z^m,s)$，regularity 条件
$$
z^{-m}s\in \mathbb C[[z]]
$$
限制 $s$ 的可允许 vanishing order。这个例子说明 matter representation 会把纯 lattice convolution 改成带闭条件的 convolution；闭条件进入 Borel-Moore homology 后会改变代数关系。

**命题 20.5.4（权一 matter 的分支计算）.** 在例 20.5.3 中，固定 affine Grassmannian 的 coweight $m\in\mathbb Z$。在取 residual $G(\mathcal O)$-equivariant quotient 之前，该分支上可允许的 sections 构成线性空间
$$
\mathcal R_m^{\mathrm{sec}}
=\mathcal O\cap z^m\mathcal O
=z^{\max(m,0)}\mathcal O.
$$
在截断 $\mathcal O/z^d\mathcal O$ 中，它的像具有 codimension
$$
\min(\max(m,0),d).
$$

**证明.** 写 $s=\sum_{r\ge0}a_rz^r\in\mathcal O$。对 $g=z^m$，第二个 regularity 条件是
$$
g^{-1}s=z^{-m}s\in\mathcal O.
$$
若 $m\le0$，则 $z^{-m}=z^{|m|}$ 不产生负幂，条件自动成立。若 $m>0$，则所有 $a_0,\ldots,a_{m-1}$ 必须为零，也就是 $s\in z^m\mathcal O$。两种情形合并为第一式。模 $z^d$ 后，$z^r\mathcal O$ 的像由 $z^r,\ldots,z^{d-1}$ 张成；当 $r<d$ 时 codimension 为 $r$，当 $r\ge d$ 时像为零，故得到第二式。$\square$

例如 $m=2$ 时 $s=a_2z^2+a_3z^3+\cdots$，而 $m=-1$ 时任意 $s\in\mathcal O$ 都可取。纯 gauge 情形每个 coweight 只贡献一个离散分支；加入 matter 后，正 coweights 带有逐渐增大的有限截断 codimension。卷积中的 Gysin/Euler 因子会记住这些 codimensions，因此命题 20.5.2 的裸 group-algebra 乘法不能原样沿用。

## 20.2 卷积代数

**外部输入定理 20.6.** BFN 定义
$$
\mathcal A=H_\ast^{G(\mathcal O)}(\mathcal R)
$$
上的 convolution product，使其成为交换 graded algebra，并定义 Coulomb branch
$$
\mathcal M_C(G,N)=\operatorname{Spec}\mathcal A.
$$
在带 loop rotation 的等变版本中得到 quantized Coulomb branch algebra $\mathcal A_\hbar$。

命题 20.5.4 只计算了构成 $\mathcal R$ 的线性条件，没有自行定义无限维 Borel--Moore fundamental classes。把这些有限截断稳定地组织起来并证明卷积交换，正是定理 20.6 的外部内容。

**资料入口 20.7.** Braverman-Finkelberg-Nakajima, arXiv:1601.03586 说明在 cotangent type 假设下把 Coulomb branch 定义为带 $\mathbb C^\times$-作用的 affine algebraic variety，并用 affine Grassmannian 型 convolution 构造乘法。

**定义 20.8.** 卷积的 correspondence 由三元组组成：两个可复合的点
$$
(g_1,s_1),\qquad (g_2,s_2)
$$
满足中间 section 的 regularity 相容条件。乘法把相对位置合成为 $g_1g_2$。形式上，它与 affine Grassmannian 的
$$
G(\mathcal K)\times^{G(\mathcal O)}G(\mathcal K)/G(\mathcal O)
$$
卷积相同，但多了 $N$-section 的闭条件。

**命题 20.9.** 若接受 BFN convolution formalism，则 $\mathcal A$ 的乘法结合。

**证明.** BFN 卷积由与 affine Grassmannian convolution 同型的三重 correspondence 控制。两次乘法的两种加括号方式都由三重 $\mathcal R$-型空间拉回、交叉并推前得到。correspondence associativity 和 Borel-Moore homology 的 projection formula 给出两者相等。关键几何和同调构造属于外部输入定理 20.6。$\square$

**定义 20.9.1.** loop rotation $\mathbb C^\times_\hbar$ 作用在 $\mathcal K=\mathbb C((z))$ 上：
$$
a\cdot z=az.
$$
加入该等变作用后定义
$$
\mathcal A_\hbar=H_\ast^{G(\mathcal O)\rtimes\mathbb C^\times_\hbar}(\mathcal R).
$$
参数 $\hbar$ 是
$$
H^2_{\mathbb C^\times_\hbar}(\mathrm{pt})\simeq\mathbb C\hbar
$$
的生成元。

**命题 20.9.2.** 若 BFN 的等变卷积构造适用，则 $\mathcal A_\hbar/(\hbar)\simeq\mathcal A$，且 $\mathcal A_\hbar$ 给出 $\mathcal A$ 的 filtered quantization。

**证明.** 忘记 loop rotation 等变性对应沿
$$
H^\ast_{\mathbb C^\times_\hbar}(\mathrm{pt})=\mathbb C[\hbar]\longrightarrow \mathbb C,\qquad \hbar\mapsto0
$$
作基变换。等变 Borel-Moore homology 对这个基变换的相容性给出模 $\hbar$ 的普通卷积代数。非交换性由旋转等变参数记录；$\hbar=0$ 后恢复交换的经典 Coulomb branch algebra。严格的平坦性与 filtered quantization statement 属于 BFN 外部输入。$\square$

## 20.3 与几何表示论的关系

**边界说明 20.10.** Coulomb branches 与 affine Grassmannian slices、shifted Yangians、symplectic duality、3d mirror symmetry、KLR/Yangian/quantum loop algebra 表示以及 categories $\mathcal O$ for quantized symplectic resolutions 有已知或预期联系。每个联系都需要独立定理，不得由 BFN 定义自动推出。

**命题 20.10.1.** 第十九章的 conical symplectic singularity 语言不能自动套用于任意 BFN 输入 $(G,N)$；必须先知道 $\mathcal M_C(G,N)$ 是 finite type affine Poisson variety，并且其 smooth locus 上 Poisson tensor 非退化。

**证明.** 第十九章定义的 symplectic singularity 要求 normal affine variety、smooth locus 上的 symplectic form、以及任意 resolution 上的 extension 条件。BFN 定义首先给出的是 convolution algebra 的 spectrum。finite generation、reducedness、normality 和 symplectic singularity 性质均不是形式定义的直接结果，必须由 BFN 定理或后续识别定理提供。$\square$

**模型条件 20.11.** 使用某个 Coulomb branch 结果前必须记录：

1. 输入 pair $(G,N)$；
2. 是否 cotangent type；
3. 使用 ordinary 还是 equivariant Borel-Moore homology；
4. 是否包含 loop rotation；
5. $\mathcal M_C$ 是否已知为 reduced、normal、symplectic singularity 或有 resolution；
6. 是否使用与 affine Grassmannian slice 或 shifted Yangian 的外部同构。

Pure torus 时 coweight components 按格点相加；权一 matter 则在第 $m$ 个分支施加 $\max(m,0)$ 阶消失条件，有限截断已能看出卷积会获得额外 Euler 因子。BFN formalism 把这些非有限型分支的等变 Borel--Moore homology 组织成交换代数，loop rotation 再给出 $\hbar$-量子化。是否得到 symplectic singularity 或 resolution 仍需对具体 $(G,N)$ 另证。下一章改用短正合列 stack 作 correspondence，从 Hall multiplication 进入 CoHA 与 DT 理论。

## 练习

**练习 20.1.** 对 $N=0$ 的 pure gauge theory，查阅 BFN 中 Coulomb branch 的已知例子，并记录所需假设。

**练习 20.2.** 解释 loop rotation 如何引入量子化参数 $\hbar$。

**练习 20.3.** 比较 affine Grassmannian convolution、Steinberg convolution 和 BFN convolution 的共同形式。

**练习 20.4.** 对 torus $G=T$ 和 $N=0$，查阅 BFN 构造给出的 Coulomb branch，并说明它与 $T^\vee$ 或 cocharacter lattice 的关系需要哪些外部定理。

**练习 20.5.** 对 $G=\mathbb C^\times$、$N=\mathbb C$，把 regularity 条件写成 $s=\sum_{i\ge0}a_iz^i$ 的系数限制，并比较 $m\ge0$ 与 $m<0$ 的差异。

**练习 20.6.** 证明例 20.5.1 的卷积单位是 $u_0$，其中 $0\in X_\ast(T)$ 是零 coweight。

**练习 20.7.** 对 $G=\mathbb C^\times$ 和权 $r>0$ 的一维表示 $N$，把命题 20.5.4 推广为
$$
\mathcal R_m^{\mathrm{sec}}=z^{\max(rm,0)}\mathcal O,
$$
并计算其模 $z^d$ 的 codimension。
