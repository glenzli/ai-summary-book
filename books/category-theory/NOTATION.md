# 符号约定

本文档记录《范畴论》的固定符号。后续章节不得随意更改。

## 集合论与大小

- 固定 Grothendieck universes
  $$
  \mathcal U\in\mathcal V\in\mathcal W.
  $$
  若不特别说明，“集合”指 $\mathcal U$-小集合，即与 $\mathcal U$ 的某个元素双射的集合。
- $\mathbf{Set}_{\mathcal U}$：以 $\mathcal U$ 中集合为对象的集合范畴。它在 $\mathcal V$ 层小、局部 $\mathcal U$-小，但不在 $\mathcal U$ 层小。
- $\mathbf{Cat}_{\mathcal U}$：$\mathcal U$-小范畴和函子构成的范畴；它同样在 $\mathcal V$ 层小且局部 $\mathcal U$-小。
- 一个范畴称为 $\mathcal U$-小，若其对象集和总态射集都是 $\mathcal U$-小集合；称为局部 $\mathcal U$-小，若每个 Hom 是 $\mathcal U$-小集合。
- 未另行声明的普通范畴默认在 $\mathcal V$ 层小且局部 $\mathcal U$-小。“小范畴”“小极限”“小余极限”中的“小”默认指 $\mathcal U$-小。
- “本质 $\mathcal U$-小”表示等价于一个 $\mathcal U$-小范畴；它不表示给定对象集本身属于 $\mathcal U$。

## 普通范畴论

- 范畴通常记为 $\mathcal C,\mathcal D,\mathcal E$。
- 对象写作 $X\in\mathcal C$ 或 $X\in\operatorname{Ob}(\mathcal C)$。
- Hom 集写作
  $$
  \mathcal C(X,Y)=\operatorname{Hom}_{\mathcal C}(X,Y).
  $$
- 态射 $f\in\mathcal C(X,Y)$ 写作 $f:X\to Y$。
- 恒等态射写作 $\operatorname{id}_X:X\to X$。
- 复合 $X\xrightarrow{f}Y\xrightarrow{g}Z$ 写作 $g\circ f:X\to Z$。
- 反范畴写作 $\mathcal C^{\operatorname{op}}$。
- 函子写作 $F:\mathcal C\to\mathcal D$。
- 自然变换写作 $\alpha:F\Rightarrow G$，其在 $X$ 处的分量写作 $\alpha_X:F(X)\to G(X)$。
- 函子范畴写作
  $$
  \operatorname{Fun}(\mathcal C,\mathcal D).
  $$
- 自然变换集合写作
  $$
  \operatorname{Nat}(F,G).
  $$
  若源范畴 $\mathcal C$ 为 $\mathcal U$-小且目标局部 $\mathcal U$-小，则该集合为 $\mathcal U$-小；若 $\mathcal C$ 只在 $\mathcal V$ 层小，则一般只保证它为 $\mathcal V$-小。

## 预层、Yoneda 与表示性

- 对 $\mathcal U$-小范畴 $\mathcal C$，$\widehat{\mathcal C}$ 表示 $\mathcal U$-小集合值预层范畴：
  $$
  \widehat{\mathcal C}=\operatorname{Fun}(\mathcal C^{\operatorname{op}},\mathbf{Set}_{\mathcal U}).
  $$
- Yoneda 嵌入写作
  $$
  y:\mathcal C\to\widehat{\mathcal C},\qquad
  X\mapsto h_X=\mathcal C(-,X).
  $$
- 协变可表函子写作 $\mathcal C(X,-)$，反变可表函子写作 $\mathcal C(-,X)$。
- 泛性质优先写成自然同构，例如
  $$
  \mathcal C(X,\lim D)\cong \operatorname{Cone}(X,D).
  $$

## 极限、伴随与 Kan 延拓

- 小图形范畴通常记为 $\mathcal J$。
- 图形写作 $D:\mathcal J\to\mathcal C$。
- 极限和余极限写作 $\lim_{\mathcal J}D$ 与 $\operatorname{colim}_{\mathcal J}D$。
- 对 $U:\mathcal I\to\mathcal J$，共尾性使用逗号范畴 $j/U$，始性使用 $U/j$。
- 伴随写作
  $$
  F:\mathcal C\rightleftarrows\mathcal D:G,\qquad F\dashv G.
  $$
- 单位和余单位写作
  $$
  \eta:\operatorname{id}_{\mathcal C}\Rightarrow GF,\qquad
  \varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}.
  $$
- 沿 $K:\mathcal C\to\mathcal D$ 的左、右 Kan 延拓写作
  $$
  \operatorname{Lan}_K F,\qquad \operatorname{Ran}_K F.
  $$
- 逗号范畴写作 $K/d$、$d/K$、$\mathcal C/X$ 或 $X/\mathcal C$。
- 元素范畴写作 $\int_{\mathcal C}P$。
- 反射子范畴通常写作
  $$
  L:\mathcal C\rightleftarrows\mathcal A:I,\qquad L\dashv I,
  $$
  其中 $I:\mathcal A\hookrightarrow\mathcal C$ 为包含；余反射子范畴写作 $I\dashv R$。
- 单子写作 $(T,\eta,\mu)$。其 Kleisli 范畴写作 $\mathcal C_T$，Eilenberg-Moore 范畴写作 $\mathcal C^T$，自由-遗忘伴随写作
  $$
  F^T:\mathcal C\rightleftarrows\mathcal C^T:U^T,
  $$
  Kleisli 伴随写作 $J:\mathcal C\rightleftarrows\mathcal C_T:G_T$。

## 幺半、富范畴和 coend

- 幺半单位对象写作 $\mathbb 1$。
- 内部 Hom 写作 $[X,Y]$。
- end 与 coend 写作
  $$
  \int_C H(C,C),\qquad \int^C H(C,C).
  $$
- Day 卷积写作 $P\star Q$。
- 富范畴的基幺半范畴通常写作 $\mathcal V$。
- $\mathcal V$-富范畴的 Hom 对象写作 $\mathcal A(A,B)\in\mathcal V$。
- 富自然变换对象写作 $\operatorname{Nat}_{\mathcal V}(F,G)$。
- 富函子范畴写作 $\operatorname{Fun}_{\mathcal V}(\mathcal A,\mathcal B)$。
- 富范畴中对象 $A$ 被 $V\in\mathcal V$ 张量写作 $V\odot A$；被 $V$ 余张量写作 $A^V$。

## Topos 与可表现范畴

- 正则基数通常写作 $\kappa$。
- $\kappa$-紧对象全子范畴写作 $\mathcal C_\kappa$。
- Ind 完备化写作 $\operatorname{Ind}_\kappa(\mathcal A)$。
- 站点写作 $(\mathcal C,J)$。
- sheaf 范畴写作 $\operatorname{Sh}(\mathcal C,J)$。
- $\infty$-sheaf 范畴写作 $\operatorname{Sh}_\infty(\mathcal C,J)$。
- spaces 的 $\infty$-范畴写作 $\mathcal S$。
- 覆盖或超覆盖的增广单纯对象通常写作 $U_\bullet\to U$。
- 超覆盖的 matching object 写作 $M_n(U_\bullet)$。
- $\infty$-topos 常写作 $\mathcal X,\mathcal Y$。
- 小 $\infty$-范畴 $C$ 的 space 值预层范畴写作
  $$
  \mathcal P(C)=\operatorname{Fun}(C^{op},\mathcal S).
  $$
- $\infty$-Yoneda 嵌入写作 $j:C\to\mathcal P(C)$。
- $\infty$-范畴的 Ind 完备化写作 $\operatorname{Ind}_\kappa(C)$。
- presentable $\infty$-categories 与左伴随组成的 $\infty$-范畴写作 $\operatorname{Pr}^L$；与右伴随组成的写作 $\operatorname{Pr}^R$。
- $n$-截断对象全子范畴写作 $\mathcal X_{\le n}$。
- $n$-截断函子写作 $\tau_{\le n}$。
- $\infty$-topos $\mathcal X$ 的 hypercompletion 写作 $\widehat{\mathcal X}$。
- 几何态射写作 $f:\mathcal X\to\mathcal Y$，其伴随写作
  $$
  f^*:\mathcal Y\rightleftarrows\mathcal X:f_*.
  $$
- 谱的 $\infty$-范畴写作 $\mathbf{Sp}$。
- heart 写作 $C^\heartsuit$。

## 高阶范畴论

- $\Delta$：单纯形范畴，对象为有限非空全序集 $[n]=\{0<\cdots<n\}$，态射为保序映射。
- $\mathbf{sSet}_{\mathcal U}$：$\mathcal U$-小单纯集范畴
  $$
  \operatorname{Fun}(\Delta^{\operatorname{op}},\mathbf{Set}_{\mathcal U}).
  $$
- 标准 $n$-单纯形写作 $\Delta^n$。
- 第 $i$ 个面写作 $d_i:\Delta^{n-1}\to\Delta^n$；第 $i$ 个退化写作 $s_i:\Delta^{n+1}\to\Delta^n$。
- 第 $i$ 个角（horn）写作 $\Lambda_i^n\subseteq\Delta^n$。
- 单纯集的 join 写作 $X\star Y$。
- slice quasi-category 写作 $C_{/p}$ 或 $C_{p/}$。
- marked simplicial set 的常用标记写作 $C^\natural,C^\sharp,C^\flat$。
- scaled nerve 写作 $N^{sc}(\mathcal B)$。
- walking adjunction 的 $2$-范畴写作 $\operatorname{Adj}$。
- quasi-category 指满足所有内角填充条件的单纯集。
- $\infty$-范畴若不特别说明即指 quasi-category。
- 普通范畴 $\mathcal C$ 的 nerve 写作 $N(\mathcal C)$。
- $\mathcal{Cat}_\infty$ 表示小 $\infty$-范畴组成的 $\infty$-范畴，具体 universe 层级由上下文决定。
- Cartesian fibration 和 coCartesian fibration 分别记作 $\operatorname{CartFib}_{/S}$ 与 $\operatorname{coCartFib}_{/S}$。
- Cartesian fibration 中沿 $\alpha:s\to t$ 的限制函子写作 $\alpha^*:X_t\to X_s$。
- Cartesian sections 组成的 $\infty$-范畴写作 $\operatorname{Sect}^{Cart}_S(X)$。

## 稳定、高阶代数与 Morita 理论

- 稳定 $\infty$-范畴中的映射谱写作 $\operatorname{Map}^{\operatorname{Sp}}_C(X,Y)$。
- 谱的 smash product 写作 $E\wedge F$。
- t-结构的截断函子写作 $\tau_{\ge n},\tau_{\le n}$；cohomology object 写作 $H^n(X)$。
- 滤过对象写作 $F_\bullet X$，associated graded 写作 $\operatorname{gr}_pX$。
- $E_r$ 页谱序列写作 $E_r^{p,q}$。
- 左、右、双模 $\infty$-范畴分别写作 $\operatorname{LMod}_A(C)$、$\operatorname{RMod}_A(C)$、${}_{A}\operatorname{BMod}_{B}(C)$。
- 相对张量积写作 $M\otimes_A N$。
- 双边 bar 构造写作 $\operatorname{Bar}_\bullet(M,A,N)$。
- Morita $\infty$-范畴写作 $\operatorname{Mor}_1(C)$。
- 代数 $A$ 的中心写作 $Z(A)$。
- 因子化同调写作 $\int_M A$。
- Profunctor 记作 $P:\mathcal C\nrightarrow\mathcal D$，即 $P:\mathcal C^{op}\times\mathcal D\to\mathbf{Set}$。
- 函子 $F:\mathcal C\to\mathcal D$ 对应的 companion/conjoint profunctors 写作 $F_*$ 与 $F^*$。
- Cauchy/Karoubi 完备化写作 $\operatorname{Kar}(\mathcal C)$。
- 加权余极限写作 $W\star D$。
- $\mathcal V$-profunctor 也写作 $M:\mathcal A\nrightarrow\mathcal B$，即 $M:\mathcal A^{op}\otimes\mathcal B\to\mathcal V$。
- 富 profunctor 双范畴写作 $\mathbf{Prof}_{\mathcal V}$。
- compact objects 子范畴常写作 $C^\omega$。
- localizing subcategory 常写作 $L\subseteq C$，Verdier quotient 写作 $C/L$。
- Bousfield localization 通常写作 $L:C\to C$；$L$-acyclic fiber 写作 $A_X$。
- $k$-模链复形范畴写作 $\operatorname{Ch}(k)$。
- dg category 常写作 $\mathcal A,\mathcal B$；其同伦范畴写作 $H^0(\mathcal A)$。
- 右 dg 模范畴写作 $\operatorname{Mod}_{\mathcal A}$，导出模 $\infty$-范畴写作 $D(\mathcal A)$。
- 可表右 dg 模写作 $h_a=\mathcal A(-,a)$。
- perfect dg modules 子范畴写作 $\operatorname{Perf}(\mathcal A)$。
- dg nerve 写作 $N_{\operatorname{dg}}(\mathcal A)$。
- dg bimodule 写作 $M:\mathcal A\nrightarrow\mathcal B$ 或 $M:\mathcal A^{op}\otimes\mathcal B\to\operatorname{Ch}(k)$。
- dg bimodule 的导出相对张量积写作 $M\otimes^{\mathbb L}_{\mathcal B}N$。
- Hochschild chains 写作 $HH(\mathcal A)$。
- 六操作系数范畴常写作 $\mathcal D(X)$。
- 态射 $f:X\to Y$ 的六操作写作
  $$
  f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-).
  $$
- $\mathcal D(X)$ 的单位对象写作 $\mathbb 1_X$。
- Cartesian 方块中上横箭头常写作 $g'$，左竖箭头写作 $f'$，并与 $f,g$ 满足 $fg'=gf'$。
- 结构态射写作 $p_X:X\to *$，dualizing object 写作 $\omega_X=p_X^!\mathbb 1_*$。
- Verdier 对偶函子写作 $\mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X)$。
- Relative category 写作 $(\mathcal C,W)$，其中 $W$ 为 weak equivalences。
- $\infty$-categorical localization 写作 $\mathcal C[W^{-1}]$。
- Simplicial category 的 Hom 单纯集写作 $\operatorname{Map}_{\mathcal A}(x,y)$，同伦范畴写作 $\pi_0\mathcal A$。
- Dwyer-Kan simplicial localization 写作 $L(\mathcal C,W)$。
- Coherent nerve 写作 $N_{\operatorname{hc}}$。
- Complete Segal space 常简写为 CSS；simplicial space 写作 $X:\Delta^{op}\to\mathcal S$。
- 小幂等完备稳定 $\infty$-范畴组成的 $\infty$-范畴写作 $\operatorname{Cat}^{\operatorname{perf}}_\infty$。
- Additive 和 localizing motives 分别写作 $\operatorname{Mot}_{\operatorname{add}}$ 与 $\operatorname{Mot}_{\operatorname{loc}}$，普遍函子写作 $U_{\operatorname{add}}$ 与 $U_{\operatorname{loc}}$。
- 代数 $K$-理论、Hochschild homology 和 topological Hochschild homology 分别写作 $K(C)$、$HH(C)$、$THH(C)$。
- 可构造导出范畴写作 $D^b_c(X,\Lambda)$ 或 $D^b_c(X)$。
- stratum 嵌入常写作 $i_\alpha:S_\alpha\hookrightarrow X$。
- Perverse t-结构两半写作 ${}^pD^{\le0}(X)$ 与 ${}^pD^{\ge0}(X)$。
- Perverse sheaves 的 heart 写作 $\operatorname{Perv}(X)$。
- 开嵌入 $j:U\hookrightarrow X$ 的中间延拓写作 $j_{!*}$。
- Nearby cycles 与 vanishing cycles 写作 $\psi_f$ 与 $\phi_f$。
- 谱 $E$ 的 Bousfield class 写作 $\langle E\rangle$。
- Morava $K$-theory 写作 $K(n)$，Johnson-Wilson 型局部化常写作 $L_n$。
- Telescope 谱写作 $T(n)$，有限局部化写作 $L_n^f$。
- $K(n)$-localization 写作 $L_{K(n)}$。
- 微分算子层写作 $D_X$，regular holonomic $D_X$-modules 的导出范畴写作 $D^b_{\operatorname{rh}}(D_X)$。
- de Rham 和 solution functors 写作 $\operatorname{DR}_X$ 与 $\operatorname{Sol}_X$。
- Characteristic variety 写作 $\operatorname{Char}(M)\subseteq T^*X$。
- Connective $E_\infty$-rings 的 $\infty$-范畴写作 $\operatorname{CAlg}^{cn}$，派生仿射范畴写作 $\operatorname{dAff}$。
- Derived prestack 写作 $F:\operatorname{dAff}^{op}\to\mathcal S$。
- 派生栈的 quasi-coherent complexes 写作 $\operatorname{QCoh}(X)$，perfect complexes 写作 $\operatorname{Perf}(X)$。
- Cotangent complex 写作 $L_A,L_{B/A},L_X$；点 $x$ 处切复形写作 $T_xX$。
- Ind-coherent sheaves 写作 $\operatorname{IndCoh}(X)$，带奇异支撑条件的子范畴写作 $\operatorname{IndCoh}_{\mathcal N}(X)$。
- 在 $\infty$-范畴语境中，monad 通常写作 $T:C\to C$，其代数 $\infty$-范畴写作 $\operatorname{Alg}_T(C)$。
- Cech nerve 写作 $U_\bullet\to X$，totalization 写作 $\operatorname{Tot}$。
- Neutral Tannakian category 的 fiber functor 写作 $\omega:\mathcal C\to\operatorname{Vect}_k^{fd}$。
- 张量自同构群函子写作 $\operatorname{Aut}^{\otimes}(\omega)$。
- Classifying stack 写作 $BG$，表示范畴写作 $\operatorname{Rep}(G)$。
- Tensor triangular category 常写作 $T$，Balmer spectrum 写作 $\operatorname{Spc}(T)$。
- 对象 $x\in T$ 的 Balmer support 写作 $\operatorname{supp}(x)$。
- Prime thick tensor ideal 常写作 $\mathfrak p$。
- Topological Hochschild homology 写作 $THH(C)$ 或 $THH(R)$。
- Topological cyclic homology 写作 $TC(C)$ 或 $TC(R)$。
- 圆群写作 $\mathbb T=S^1$，Tate construction 写作 $X^{tC_p}$。
- Cyclotomic trace 写作 $\operatorname{tr}_{cycl}:K(C)\to TC(C)$。
- Goodwillie $n$-excisive approximation 写作 $P_nF$，homogeneous layer 写作 $D_nF$。
- Cross-effect 写作 $\operatorname{cr}_nF$，Goodwillie derivative 写作 $\partial_nF$。
- $S$ 上光滑概形范畴写作 $\operatorname{Sm}_S$，motivic spaces 写作 $\mathbf H(S)$。
- Stable motivic homotopy category 写作 $\mathbf{SH}(S)$。
- Tate sphere 写作 $T=\mathbb A^1/(\mathbb A^1\setminus0)$，Thom space 写作 $\operatorname{Th}(V)$。
- Motivic Eilenberg-Mac Lane spectrum 写作 $H\mathbb Z$。
- 子对象偏序写作 $\operatorname{Sub}_{\mathcal C}(X)$，存在量词左伴随写作 $\exists_f$。
- Slice 上替换、依赖和与依赖积写作
  $$
  \Sigma_f\dashv f^*\dashv\Pi_f.
  $$
- Comprehension category 的上下文扩张写作 $\Gamma.A\to\Gamma$。
- Universe 写作 $\mathcal U$，univalence 中的等价类型写作 $\operatorname{Equiv}(A,B)$。
- 小圆盘范畴写作 $\operatorname{Disk}_n$，流形范畴写作 $\operatorname{Mfld}_n$。
- 因子化同调写作 $\int_MA$，圆盘 overcategory 写作 $\operatorname{Disk}_{n/M}$。
- $E_n$-代数范畴写作 $\operatorname{Alg}_{E_n}(C)$。
- $n$-重 delooping 写作 $B^nA$，紧支撑映射空间写作 $\operatorname{Map}_c(M,B^nA)$。
- Profinite sets 范畴写作 $\operatorname{ProFin}$。
- Condensed sets 与 condensed abelian groups 写作 $\operatorname{Cond}(\mathbf{Set})$ 与 $\operatorname{Cond}(\mathbf{Ab})$。
- 离散集合或拓扑空间的凝聚化写作 $\underline A$ 或 $\underline T$。
- Solidification 写作 $(-)^{\mathrm{solid}}$，solid tensor product 写作 $\otimes^{\mathrm{solid}}$。
- Solid $A$-modules 写作 $\operatorname{Mod}^{\mathrm{solid}}_A$，solid derived category 写作 $D_{\mathrm{solid}}(A)$。
- 理论 $T$ 的语法范畴写作 $\mathcal C_T^{\operatorname{syn}}$，分类 topos 写作 $\mathcal E_T$，泛模型写作 $U_T$。
- 几何态射范畴写作 $\operatorname{Geom}(\mathcal F,\mathcal E)$，$T$-模型范畴写作 $\operatorname{Mod}_T(\mathcal F)$。
- Tripos 常写作 $P:\mathcal C^{op}\to\mathbf{Heyt}$，generic predicate 的分类对象写作 $\Omega$。
- 关系写作 $R:X\nrightarrow Y$，态射 $f$ 的图像关系写作 $\Gamma_f$，反向关系写作 $R^\circ$。
- Regular 和 exact completions 写作 $\mathcal C_{\mathrm{reg}}$ 与 $\mathcal C_{\mathrm{ex}}$。
- Cohesive 伴随串写作 $\Pi\dashv\operatorname{Disc}\dashv\Gamma\dashv\operatorname{Codisc}$。
- Cohesive 模态写作 $\int,\flat,\sharp$；de Rham shape 写作 $X_{\mathrm{dR}}$。
- 层化空间的 exit-path $\infty$-category 写作 $\operatorname{Exit}(X)$。
- Constructible sheaves 写作 $\operatorname{Shv}_{cbl}(X;\mathcal S)$。
- $n$-重 Morita 范畴写作 $\operatorname{Alg}_n(C)$。
- Morita trace 写作 $\operatorname{Tr}(\operatorname{id}_A)$。
- 增广 $E_n$-代数的 Koszul dual 写作 $A^!$，$n$-重 bar 构造写作 $\operatorname{Bar}^n(A)$。
- Derivator 写作 $\mathbb D:\mathbf{Cat}^{op}\to\mathbf{CAT}$；限制函子写作 $u^*$，同伦 Kan 延拓写作 $u_!,u_*$。
- 由 $\infty$-category $C$ 诱导的 derivator 写作 $\mathbb D_C(I)=h\operatorname{Fun}(N(I),C)$。
- Stack 的 descent category 写作 $\operatorname{Desc}(F,U_\bullet)$；classifying stack 写作 $BG$。
- Descent morphism $p:E\to B$ 的 descent category 写作 $\operatorname{Desc}(p)$。
- Galois 群写作 $\operatorname{Gal}(L/K)$。
- 多项式函子用图 $I\xleftarrow{s}E\xrightarrow{p}B\xrightarrow{t}J$ 表示，并写作 $P=\Sigma_t\Pi_ps^*$。
- Species 的解析函子写作 $\widehat F(X)=\sum_{n\ge0}F[n]\times_{\Sigma_n}X^n$。
- W-type 通常写作 $W$，其结构映射写作 $\alpha:P(W)\to W$。
- $\infty$-cosmos 写作 $\mathcal K$，其 homotopy 2-category 写作 $\mathcal K_2$，映射 quasi-category 写作 $\operatorname{map}_{\mathcal K}(A,B)$。
- 态射正交写作 $f\perp g$；态射类正交写作 ${}^\perp\mathcal S$ 与 $\mathcal S^\perp$。
- 正交因子化系统写作 $(\mathcal E,\mathcal M)$，弱因子化系统写作 $(\mathcal L,\mathcal R)$。
- Sketch 通常写作 $\mathcal S$，理论常写作 $\mathbb T$。
- Karoubi 包络写作 $\operatorname{Kar}(\mathcal C)$。

## 证明用语

- “同构”保留给普通范畴中的可逆态射或严格同构。
- “等价”在普通范畴中通常指范畴等价；在 $\infty$-范畴中指相应映射空间中的等价边，必须由上下文说明。
- “自然”必须指明相对于哪些变量自然。
- “唯一”若在高阶环境中使用，必须说明是严格唯一、至多唯一、或在可缩空间中唯一。
