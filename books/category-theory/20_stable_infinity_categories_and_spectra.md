# 第二十章：稳定 $\infty$-范畴与谱

## 本章目标

本章定义稳定 $\infty$-范畴、纤维/余纤维序列、三角结构和谱。稳定 $\infty$-范畴是现代同调代数和稳定同伦论的共同语言。

## 依赖前置知识

需要 $\infty$-范畴中的有限极限、有限余极限、零对象和映射空间。

## 20.1 指向对象与零对象

**定义 20.1.** $\infty$-范畴 $C$ 的零对象是既终又始的对象。若 $C$ 有零对象，则称 $C$ 是 pointed。

**定义 20.2.** 在 pointed $\infty$-范畴中，态射 $f:X\to Y$ 的纤维是拉回

$$
\operatorname{fib}(f)=X\times_Y 0,
$$

余纤维是推出

$$
\operatorname{cofib}(f)=0\sqcup_X Y.
$$

## 20.2 稳定性

**定义 20.3.** $\infty$-范畴 $C$ 称为稳定（stable），若：

1. $C$ 有零对象；
2. $C$ 有所有有限极限和有限余极限；
3. $C$ 中一个方块是推出，当且仅当它是拉回。

**例子 20.4.** 链复形的导出 $\infty$-范畴、谱的 $\infty$-范畴 $\mathbf{Sp}$、环谱上的模 $\infty$-范畴都是稳定 $\infty$-范畴。

**外部输入定理 20.5.** 稳定 $\infty$-范畴的同伦范畴 $hC$ 带有自然三角范畴结构。

平移函子由悬挂 $\Sigma X=\operatorname{cofib}(X\to0)$ 给出。态射 $f:X\to Y$ 产生余纤维序列

$$
X\to Y\to\operatorname{cofib}(f)\to\Sigma X.
$$

稳定性保证纤维和余纤维兼容，并给出三角公理。该定理的完整证明见 Lurie *Higher Algebra* 中稳定 $\infty$-范畴的旋转、八面体和同伦推出-拉回演算。

## 20.3 谱

**定义 20.6.** 谱的 $\infty$-范畴 $\mathbf{Sp}$ 可定义为 pointed spaces 的稳定化：

$$
\mathbf{Sp}=\operatorname{Stab}(\mathcal S_*).
$$

等价地，可用序列 $(E_n)_{n\ge0}$ 连同等价 $E_n\simeq\Omega E_{n+1}$ 的 $\Omega$-谱模型描述。

**外部输入定理 20.7.** $\mathbf{Sp}$ 是 presentable stable $\infty$-category，并且对任意 presentable $\infty$-category $C$，稳定化 $\operatorname{Stab}(C)$ 满足相应泛性质。

**定义 20.A.** 一个 sequential prespectrum 是 pointed spaces 的序列

$$
E_0,E_1,E_2,\dots
$$

连同结构映射

$$
\sigma_n:\Sigma E_n\to E_{n+1}\qquad(n\ge0).
$$

其伴随映射记为

$$
\tilde\sigma_n:E_n\to\Omega E_{n+1}.
$$

若每个 $\tilde\sigma_n$ 都是 spaces 中的等价，则称 $E$ 为 $\Omega$-谱。顺序谱模型中，稳定对象由对 prespectra 作合适局部化得到；其中 fibrant 对象可取为 $\Omega$-谱。

**例子 20.B.** sphere spectrum $\mathbb S$ 的第 $n$ 项为

$$
\mathbb S_n=S^n,
$$

结构映射

$$
\Sigma S^n\cong S^{n+1}\to S^{n+1}
$$

取标准同构。它是稳定同伦论中的单位对象；在严格模型中需要先取 fibrant replacement 才得到 $\Omega$-谱。

**定义 20.C.** 谱 $E$ 的稳定同伦群定义为

$$
\pi_k(E)=\operatorname*{colim}_{n}\pi_{k+n}(E_n),
$$

其中过渡映射由结构映射的伴随 $E_n\to\Omega E_{n+1}$ 诱导。若 $E$ 是 $\Omega$-谱，则对足够按定义合法的指标，有

$$
\pi_k(E)\cong \pi_{k+n}(E_n)
$$

在稳定范围内由结构等价识别。

## 20.4 谱富化、映射谱与 smash product

**外部输入定理 20.D.** 每个稳定 $\infty$-范畴 $C$ 都典范地富化于谱。也就是说，对任意 $X,Y\in C$，存在映射谱

$$
\operatorname{Map}^{\operatorname{Sp}}_C(X,Y)\in\mathbf{Sp}
$$

满足

$$
\Omega^\infty\operatorname{Map}^{\operatorname{Sp}}_C(X,Y)
\simeq
\operatorname{Map}_C(X,Y),
$$

并且

$$
\pi_n\operatorname{Map}^{\operatorname{Sp}}_C(X,Y)
\cong
hC(\Sigma^nX,Y)
$$

在通常符号约定下成立。其构造依赖稳定 $\infty$-范畴中悬挂与环路互逆以及谱的泛性质，完整证明见 Lurie *Higher Algebra*。

**命题 20.E.** 若 $F:C\to D$ 是稳定 $\infty$-范畴之间的正合函子，则 $F$ 诱导映射谱之间的自然态射

$$
\operatorname{Map}^{\operatorname{Sp}}_C(X,Y)\to
\operatorname{Map}^{\operatorname{Sp}}_D(FX,FY).
$$

**证明.** 正合函子保持零对象、有限极限、有限余极限，并与悬挂相容。因此它把 $C$ 中表示映射谱各层和结构映射的纤维/环路数据送到 $D$ 中相应数据。由映射谱的泛性质，得到自然的谱态射。$\square$

**外部输入定理 20.F.** $\mathbf{Sp}$ 带有闭对称幺半结构

$$
(E,F)\longmapsto E\wedge F
$$

称为 smash product，单位为 sphere spectrum $\mathbb S$。存在内部 Hom 谱 $\underline{\operatorname{Hom}}(F,G)$，满足自然等价

$$
\operatorname{Map}_{\mathbf{Sp}}(E\wedge F,G)
\simeq
\operatorname{Map}_{\mathbf{Sp}}(E,\underline{\operatorname{Hom}}(F,G)).
$$

此外，$E\wedge-$ 与 $-\wedge E$ 保持余极限和正合三角。

**定义 20.G.** 环谱是 $\mathbf{Sp}$ 的幺半 $\infty$-范畴中的 $E_1$-代数。交换环谱是 $E_\infty$-代数。若 $R$ 是环谱，则 $R$-模构成稳定 $\infty$-范畴 $\operatorname{Mod}_R$；其构造将在第二十二章用高阶代数语言描述。

## 20.5 悬挂、环路与正合函子

**定义 20.8.** 在 pointed $\infty$-范畴中定义

$$
\Sigma X=\operatorname{cofib}(X\to0),\qquad
\Omega X=\operatorname{fib}(0\to X).
$$

它们分别称为悬挂和环路对象。

**命题 20.9.** 若 $C$ 是稳定 $\infty$-范畴，则 $\Sigma:C\to C$ 与 $\Omega:C\to C$ 互为等价。

**证明.** 对任意 $X$，按定义有推出方块

$$
\begin{matrix}
X&\longrightarrow&0\\
\downarrow&&\downarrow\\
0&\longrightarrow&\Sigma X.
\end{matrix}
$$

由于 $C$ 稳定，该方块同时是拉回方块。拉回泛性质说明 $X$ 是 $0\to\Sigma X$ 的纤维，即

$$
X\simeq\Omega\Sigma X.
$$

对偶地，定义 $\Omega X$ 的拉回方块

$$
\begin{matrix}
\Omega X&\longrightarrow&0\\
\downarrow&&\downarrow\\
0&\longrightarrow&X
\end{matrix}
$$

同时是推出方块，所以

$$
\Sigma\Omega X\simeq X.
$$

这些等价由相应泛性质自然给出，故 $\Sigma$ 与 $\Omega$ 互为逆等价。$\square$

**定义 20.10.** 稳定 $\infty$-范畴之间的函子 $F:C\to D$ 称为正合（exact），若它保持零对象、有限极限和有限余极限。等价地，它保持有限拉回-推出方块和零对象。

**命题 20.11.** 正合函子保持纤维、余纤维和余纤维序列。

**证明.** 设 $f:X\to Y$。纤维是拉回

$$
\operatorname{fib}(f)=X\times_Y0.
$$

因 $F$ 保持零对象和有限极限，

$$
F\operatorname{fib}(f)\simeq FX\times_{FY}0\simeq\operatorname{fib}(Ff).
$$

同理，余纤维是推出

$$
\operatorname{cofib}(f)=0\sqcup_XY,
$$

而 $F$ 保持零对象和有限余极限，所以

$$
F\operatorname{cofib}(f)\simeq0\sqcup_{FX}FY\simeq\operatorname{cofib}(Ff).
$$

因此由 $f$ 产生的序列

$$
X\to Y\to\operatorname{cofib}(f)\to\Sigma X
$$

被送到 $Ff$ 的余纤维序列。$\square$

## 20.6 t-结构

**定义 20.12.** 稳定 $\infty$-范畴 $C$ 上的 t-结构由全子范畴 $C_{\ge0}$ 与 $C_{\le0}$ 组成，满足平移闭性、正交性和截断三角存在性。其 heart

$$
C^\heartsuit=C_{\ge0}\cap C_{\le0}
$$

是阿贝尔范畴。

**外部输入定理 20.13.** t-结构的 heart 是阿贝尔范畴，且许多经典导出范畴的标准 t-结构恢复原来的阿贝尔范畴。

**定义 20.H.** 与 t-结构相伴的截断函子记为

$$
\tau_{\ge0}:C\to C_{\ge0},\qquad
\tau_{\le0}:C\to C_{\le0}.
$$

一般地，

$$
C_{\ge n}=\Sigma^n C_{\ge0},\qquad C_{\le n}=\Sigma^n C_{\le0},
$$

并定义

$$
\tau_{\ge n},\tau_{\le n}
$$

为相应截断。对象 $X$ 的第 $n$ 个 cohomology object 定义为

$$
H^n(X)=\tau_{\le0}\tau_{\ge0}(\Sigma^{-n}X)\in C^\heartsuit.
$$

**命题 20.I.** 若 $f:A\to B$ 是 heart $C^\heartsuit$ 中的态射，则其核和余核可由稳定结构及截断给出：

$$
\ker(f)=H^0(\operatorname{fib}(f)),\qquad
\operatorname{coker}(f)=H^0(\operatorname{cofib}(f)).
$$

**证明.** 纤维和余纤维存在，因为 $C$ 稳定。虽然 $\operatorname{fib}(f)$ 与 $\operatorname{cofib}(f)$ 一般不在 heart 中，但取 $H^0$ 会把它们投影回 $C^\heartsuit$。对任意 $K\in C^\heartsuit$，映射空间

$$
\operatorname{Map}_{C^\heartsuit}(K,H^0\operatorname{fib}(f))
$$

由截断伴随性等同于那些 $K\to A$ 且复合到 $B$ 为零的映射；这正是核的泛性质。余核对偶，由从 $B$ 到 heart 对象且杀掉 $A$ 的映射刻画。$\square$

**外部输入定理 20.J.** 上述核、余核、coimage 和 image 在 heart 中满足阿贝尔范畴公理；特别地，canonical map $\operatorname{coim}(f)\to\operatorname{im}(f)$ 是同构。完整证明使用 t-结构正交性、截断三角和稳定范畴中的纤维-余纤维演算。

**命题 20.P.** heart $C^\heartsuit$ 是加性范畴，其有限 biproduct 由 $C$ 中的有限积/余积给出。

**证明.** 稳定 $\infty$-范畴有零对象，故 $C^\heartsuit$ 继承零对象。若 $A,B\in C^\heartsuit$，稳定范畴中的有限积和有限余积一致；记共同对象为 $A\oplus B$。由于 $C_{\ge0}$ 与 $C_{\le0}$ 对有限极限和有限余极限在相应截断范围内封闭，$A\oplus B$ 仍在二者交中。映射空间的 $\pi_0$ 给出

$$
hC(K,A\oplus B)\cong hC(K,A)\times hC(K,B),
$$

以及对偶的余积泛性质。因此在 heart 中 $A\oplus B$ 同时为积和余积。加法由 biproduct 标准公式

$$
f+g=\nabla_B\circ(f\oplus g)\circ\Delta_A
$$

定义，其中 $\Delta_A:A\to A\oplus A$ 与 $\nabla_B:B\oplus B\to B$ 是对角和余对角。结合律、交换律和零元律化为 biproduct 的泛性质验证。$\square$

**推论 20.Q.** 对 heart 中态射 $f:A\to B$，下列条件等价：

1. $f$ 是 monomorphism；
2. $\ker(f)\simeq0$；
3. $H^0(\operatorname{fib}(f))\simeq0$。

对偶地，$f$ 是 epimorphism 当且仅当 $\operatorname{coker}(f)\simeq0$，也当且仅当 $H^0(\operatorname{cofib}(f))\simeq0$。

**证明.** 在任意有核的加性范畴中，$f$ 为 monomorphism 当且仅当其核为零对象：若 $f$ 为 monomorphism，则核 $k:K\to A$ 满足 $fk=0=f0$，由单性得 $k=0$，核的泛性质迫使 $K\simeq0$。反过来若核为零，且 $fu=fv$，则 $f(u-v)=0$，所以 $u-v$ 经零核唯一分解，故 $u=v$。第三个条件由命题 20.I 的核公式给出。余核陈述对偶。$\square$

**外部输入定理 20.R（cohomology 长正合列）.** 若

$$
X\to Y\to Z\to\Sigma X
$$

是带 t-结构稳定 $\infty$-范畴中的纤维-余纤维序列，则存在 heart 中的长正合列

$$
\cdots\to H^n(X)\to H^n(Y)\to H^n(Z)
\xrightarrow{\partial}
H^{n+1}(X)\to H^{n+1}(Y)\to\cdots.
$$

边界映射 $\partial$ 由三角中的连接态射 $Z\to\Sigma X$ 经截断得到。正合性指每一项处 image 等于 kernel。该定理依赖外部输入定理 20.J 中 heart 的阿贝尔性；其余构造由稳定范畴中的旋转三角和截断函子给出。

**例子 20.S.** 若 $C=D(\mathcal A)$ 是阿贝尔范畴 $\mathcal A$ 的导出 $\infty$-范畴并取标准 t-结构，则

$$
D(\mathcal A)^\heartsuit\simeq\mathcal A.
$$

对象 $X$ 的 $H^n(X)$ 就是链复形的通常第 $n$ 个上同调对象。态射 $f:A\to B$ 位于 heart 中时，$\operatorname{fib}(f)$ 与 $\operatorname{cofib}(f)$ 分别由两项复形表示，命题 20.I 恢复 $\mathcal A$ 中的通常核和余核。

**定义 20.K.** 设 $X$ 是带递增或递减滤过的对象，例如递增滤过

$$
0=F_{-1}X\to F_0X\to F_1X\to\cdots\to X.
$$

其第 $p$ 个 associated graded 定义为

$$
\operatorname{gr}_pX=\operatorname{cofib}(F_{p-1}X\to F_pX).
$$

在带 t-结构的稳定 $\infty$-范畴中，滤过对象的 cohomology 由各 $\operatorname{gr}_pX$ 的 cohomology 组织成谱序列。

**外部输入定理 20.L（滤过对象谱序列）.** 在适当有界性或收敛性假设下，滤过对象 $X$ 给出谱序列

$$
E_1^{p,q}=H^{p+q}(\operatorname{gr}_pX)
\Longrightarrow
H^{p+q}(X).
$$

微分和收敛由滤过三角的 exact couple 构造给出。

**定义 20.M.** exact couple 由一对双次数对象 $D,E$ 和三个次数固定的态射

$$
D\xrightarrow{i}D,\qquad D\xrightarrow{j}E,\qquad E\xrightarrow{k}D
$$

组成，使得三角形

$$
D\xrightarrow{j}E\xrightarrow{k}D\xrightarrow{i}D
$$

在 heart 中正合。由 exact couple 定义微分

$$
d=jk:E\to E,
$$

并通过取 homology 得到导出 exact couple；反复导出产生谱序列。

**构造 20.N.** 滤过对象

$$
\cdots\to F_{p-1}X\to F_pX\to F_{p+1}X\to\cdots
$$

给出余纤维三角

$$
F_{p-1}X\to F_pX\to\operatorname{gr}_pX\to\Sigma F_{p-1}X.
$$

对这些三角取 cohomology 并按 $p$ 排列，可得到 exact couple。其 $E_1$ 页为

$$
E_1^{p,q}=H^{p+q}(\operatorname{gr}_pX),
$$

第一个微分来自相邻余纤维三角的连接态射。

**定义 20.O.** 谱序列称为强收敛到 $H^*(X)$，若其极限页 $E_\infty$ 与 $H^*(X)$ 上由滤过诱导的 associated graded 对象同构，并且不存在隐藏的无限扩张问题。常用充分条件包括滤过有界、完备且 Hausdorff，或在每个总次数上只有有限多个非零项。

**定义 20.T.** 递增滤过对象 $F_\bullet X$ 称为有限滤过，若存在整数 $a\le b$，使得

$$
F_pX\simeq0\quad(p<a),\qquad F_pX\simeq X\quad(p\ge b).
$$

此时 $X$ 只由有限多个 graded pieces $\operatorname{gr}_aX,\dots,\operatorname{gr}_bX$ 组成。

**定理 20.U（有限滤过的收敛）.** 设 $C$ 带有 t-结构，$X$ 为有限滤过对象。由构造 20.N 得到的谱序列在每个总次数上强收敛到 $H^*(X)$ 上的有限滤过；更精确地，

$$
E_\infty^{p,q}\cong
\operatorname{gr}_pH^{p+q}(X).
$$

**证明.** 固定总次数 $n=p+q$。由于滤过有限，只有 $a\le p\le b$ 的有限多个 graded pieces 可能对 $n$ 次 cohomology 有贡献。因此 exact couple 的导出过程在该总次数上经过有限步后不再出现来自任意远处 filtration degree 的微分。

余纤维三角

$$
F_{p-1}X\to F_pX\to\operatorname{gr}_pX\to\Sigma F_{p-1}X
$$

经外部输入定理 20.R 给出长正合列。把这些长正合列沿 $p$ 排列得到 exact couple；其导出页记录由 filtration 上一步一步取 kernel 与 quotient 后留下的部分。有限性保证该过程在每个 $n$ 上稳定，稳定值正是

$$
\operatorname{im}\bigl(H^n(F_pX)\to H^n(X)\bigr)/
\operatorname{im}\bigl(H^n(F_{p-1}X)\to H^n(X)\bigr).
$$

这就是 $H^n(X)$ 的 induced filtration 的第 $p$ 个 associated graded。由于 filtration 有限，不存在无限下降或无限上升链造成的 $\lim^1$ 型障碍；因此得到强收敛。$\square$

**定义 20.V.** 递增滤过 $F_\bullet X$ 称为 exhaustive，若自然映射

$$
\operatorname*{colim}_pF_pX\to X
$$

是等价。称为 separated，若

$$
\lim_pF_pX\simeq0
$$

在由 $p\to-\infty$ 的反向系统意义下成立。称为 complete，若 $X$ 等价于其由 quotients 或 truncations 给出的完成对象；在稳定 $\infty$-范畴中常写为

$$
X\simeq\lim_p X/F_pX
$$

或相应递减滤过版本。具体公式依赖滤过方向，本书只在明确方向时使用。

**定义 20.W.** 带 t-结构的稳定 $\infty$-范畴称为 left complete，若每个对象 $X$ 都由其 Postnikov tower 恢复：

$$
X\simeq\lim_n\tau_{\le n}X.
$$

称为 right complete，若

$$
X\simeq\operatorname*{colim}_n\tau_{\ge -n}X
$$

对所有对象成立。

**外部输入定理 20.X（完备滤过的条件收敛）.** 设 $C$ 为带 t-结构的稳定 presentable $\infty$-范畴，且 t-结构与相关极限/余极限相容。若滤过 $F_\bullet X$ exhaustive、complete，并满足每个总次数上的有界性或 Mittag-Leffler 型条件，则构造 20.N 的谱序列条件收敛或强收敛到 $H^*(X)$。其 $E_\infty$ 页给出目标 cohomology 上诱导滤过的 associated graded；若 filtration Hausdorff 且 complete，则隐藏无限扩张由完成条件控制。

该定理包含许多版本：经典 Boardman 收敛定理、filtered derived category 的谱序列收敛、以及稳定 $\infty$-范畴中的 t-structure compatible convergence。本书把一般形式作为外部输入，只在有限滤过情形给出书内证明。

**例子 20.Y（Postnikov 谱序列）.** 对带 t-结构且 left complete 的稳定 $\infty$-范畴，Postnikov tower

$$
\cdots\to\tau_{\le n}X\to\tau_{\le n-1}X\to\cdots
$$

给出由 cohomology objects $H^n(X)$ 控制的滤过。若 tower 收敛到 $X$，则相应谱序列把对象的层状 cohomology 数据组织为可计算的 exact couple。对 sheaf cohomology、Adams 型分辨率和 filtered complexes，这一结构是许多计算谱序列的共同来源。

**命题 20.Z.** 若滤过 $F_\bullet X$ 有界下方且 exhaustive，并且对每个总次数 $n$ 只有有限多个 $p$ 使 $H^{n}(\operatorname{gr}_pX)$ 非零，则相应谱序列在总次数 $n$ 上不存在无限进入或离开该次数的微分。

**证明.** 固定 $n$。微分 $d_r$ 改变 bidegree，其源或靶位于同一总次数附近的有限集合中。按假设，可能非零的 $E_1^{p,n-p}$ 只有有限多个 $p$。随着 $r$ 增大，任何给定项可能接收或发出的微分只能来自这些有限位置；超过最大距离后不存在源或靶。因此在该总次数上页数稳定，不会有无限长微分链。$\square$

## 20.7 本章小结

稳定 $\infty$-范畴把三角范畴提升到保留高阶映射空间的环境。谱是 pointed spaces 的稳定化，是稳定同伦论的基本对象。t-结构把稳定高阶范畴与阿贝尔范畴和同调代数连接起来；有限滤过和完备滤过对象进一步给出可计算的谱序列。

## 练习

**练习 20.1.** 证明 pointed 普通范畴中的零态射来自零对象。

**练习 20.2.** 在链复形中写出映射锥，并比较余纤维。

**练习 20.3.** 说明为什么三角范畴本身不足以唯一恢复映射空间。

**练习 20.4.** 查阅谱的 sequential spectrum 模型，并写出结构映射。

**练习 20.5.** 解释 heart 为阿贝尔范畴为何需要稳定性和 t-结构公理。

**练习 20.6.** 用稳定性证明 $\Omega\Sigma X\simeq X$。

**练习 20.7.** 设 $F:C\to D$ 是正合函子。证明 $F(\Sigma X)\simeq\Sigma F(X)$。

**练习 20.8.** 写出 sequential prespectrum 的伴随结构映射 $E_n\to\Omega E_{n+1}$。

**练习 20.9.** 对 sphere spectrum 解释为什么结构映射 $\Sigma S^n\to S^{n+1}$ 是自然的。

**练习 20.10.** 用外部输入定理 20.D 解释为什么稳定 $\infty$-范畴比三角范畴多记录映射谱。

**练习 20.11.** 证明正合函子诱导 $hC(\Sigma^nX,Y)\to hD(\Sigma^nFX,FY)$。

**练习 20.12.** 说明为什么 ring spectrum 应被看作 $\mathbf{Sp}$ 中的 $E_1$-代数。

**练习 20.13.** 解释为什么 $H^0(\operatorname{fib}(f))$ 落在 heart 中。

**练习 20.14.** 对 heart 中态射 $f:A\to B$，写出余核的泛性质并与 $H^0(\operatorname{cofib}(f))$ 比较。

**练习 20.15.** 对两步滤过 $0\to F_0X\to X$，写出 associated graded 对象。

**练习 20.16.** 在 exact couple 中证明 $d^2=0$。

**练习 20.17.** 对有限滤过说明为什么每个总次数上只有有限多个 $E_1^{p,q}$ 非零。

**练习 20.18.** 解释 $E_\infty$ 页为什么通常只给出目标对象的 associated graded，而不是直接给出目标对象本身。

**练习 20.19.** 在任意有 biproduct 的范畴中，写出两个态射 $f,g:A\to B$ 的和 $f+g$ 的定义。

**练习 20.20.** 证明加性范畴中若 $\ker(f)=0$，则 $f$ 是 monomorphism。

**练习 20.21.** 对导出范畴中由短正合列 $0\to A\to B\to C\to0$ 给出的三角，写出外部输入定理 20.R 的长正合列。

**练习 20.22.** 对三步滤过 $0\to F_0X\to F_1X\to X$，列出所有 $\operatorname{gr}_pX$。

**练习 20.23.** 说明有限滤过为什么排除谱序列中的无限扩张问题。

**练习 20.24.** 在标准 t-结构下，解释为什么复形的通常上同调对象属于 heart。

**练习 20.25.** 解释 exhaustive filtration 的含义。

**练习 20.26.** 比较 separated 与 complete 两个条件。

**练习 20.27.** 写出 left complete t-结构的定义。

**练习 20.28.** 说明为什么有限滤过自动满足完备性问题中的大多数收敛障碍。

**练习 20.29.** 对 Postnikov tower，指出其 graded pieces 与 cohomology objects 的关系。

**练习 20.30.** 证明命题 20.Z 中“有限多个非零项”如何排除无限微分链。
