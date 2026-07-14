# 第二章：导出范畴、完美复形与 B-side 增强

符号 $\mathrm D^b\operatorname{Coh}(X)$ 容易造成一种错觉：似乎 B-side 已经由一个三角范畴完整确定。实际上，Fourier--Mukai 核在链级复合，Hochschild 不变量依赖增强，而当 $X$ 奇异时，完美复形与有界 coherent 复形甚至不是同一个范畴。本章沿着这三处差异展开：先由 quasi-isomorphism 的局部化得到导出范畴，再分离 $\operatorname{Perf}(X)$ 与 $\mathrm D^b\operatorname{Coh}(X)$，随后给出 dg enhancement、核函子和 matrix factorization 的可比较模型。读者需要复形、abelian 范畴、局部自由层及导出拉推张量的基本语言；一般导出代数几何不作前置输入。

## 2.1 从复形到导出范畴

**定义 2.1.** 设 $\mathcal A$ 是 $k$-线性 abelian category。复形范畴 $\operatorname{Ch}(\mathcal A)$ 的对象是 cochain complexes
$$
\cdots\to C^{i-1}\xrightarrow{d^{i-1}}C^i\xrightarrow{d^i}C^{i+1}\to\cdots,
\qquad d^{i+1}d^i=0.
$$
链映射为保持次数并与微分交换的 morphisms。

**定义 2.2.** 链映射 $f:C\to D$ 称为 quasi-isomorphism，若对所有 $i$，诱导映射
$$
H^i(f):H^i(C)\to H^i(D)
$$
为同构。

**定义 2.3.** $\mathcal A$ 的导出范畴 $\mathrm D(\mathcal A)$ 是同伦范畴 $K(\mathcal A)$ 对所有 quasi-isomorphisms 的 Verdier localization：
$$
\mathrm D(\mathcal A)=K(\mathcal A)[\operatorname{Qis}^{-1}].
$$
若只允许 bounded complexes，则写作 $\mathrm D^b(\mathcal A)$。

**警告 2.4.** $\mathrm D(\mathcal A)$ 是三角范畴。HMS 中只给出三角范畴 $\mathrm D^b\operatorname{Coh}(X)$ 仍不够，因为 Hochschild invariants、Fourier-Mukai kernels 和 Morita theory 需要 dg 或 stable $\infty$ enhancement。

## 2.2 完美复形

**定义 2.5.** 设 $X$ 是 $k$-scheme。一个 $\mathcal O_X$-module 复形 $E$ 称为 perfect，若 Zariski 局部上 $E$ quasi-isomorphic 于有限长的有限秩局部自由 $\mathcal O_X$-module 复形。perfect complexes 的增强范畴记为 $\operatorname{Perf}(X)$。

**定义 2.6.** 若 $X$ 是 locally noetherian scheme，$\operatorname{Coh}(X)$ 表示 coherent sheaves 构成的 abelian category，$\mathrm D^b\operatorname{Coh}(X)$ 表示其 bounded derived category。

**外部输入定理 2.7（regular scheme 上 perfect 与 coherent 的比较）.** 设
$X$ 是 regular noetherian $k$-scheme；特别地，可取光滑有限型
$k$-scheme。则 $\mathrm D(\mathcal O_X)$ 中的 full subcategories 满足
$$
\mathrm D_{\mathrm{perf}}(X)=\mathrm D^b_{\mathrm{Coh}}(\mathcal O_X).
$$
因此对后文固定的 dg models 有
$H^0\operatorname{Perf}_{\mathrm{dg}}(X)\simeq
\mathrm D^b\operatorname{Coh}(X)$。

**证明路线（外部输入）.** Perfectness 是局部性质。Regular local rings 的
有限 global dimension 使 bounded coherent complexes 局部具有有限长
finite-free resolutions；反向包含来自 perfect complexes 的 bounded
coherent cohomology。完整结论引用 Stacks Project, Tag 0FDC
(Lemma 36.11.8)，本书不重证 regular-local-ring 输入。

**警告 2.8.** 若 $X$ 奇异，$\operatorname{Perf}(X)$ 与 $\mathrm D^b\operatorname{Coh}(X)$ 一般不同。奇异性正由 quotient
$$
\mathrm D_{\mathrm{sg}}(X)=\mathrm D^b\operatorname{Coh}(X)/\operatorname{Perf}(X)
$$
测量，这也是 Landau-Ginzburg B-side 和 matrix factorization 进入 HMS 的原因之一。

**反例 2.8A（dual numbers）.** 令
$R=k[\varepsilon]/(\varepsilon^2)$、$X=\operatorname{Spec}R$，并把 residue
field $k=R/(\varepsilon)$ 看成 coherent sheaf。它有无限周期自由分解
$$
\cdots\xrightarrow{\varepsilon}R
\xrightarrow{\varepsilon}R
\xrightarrow{\varepsilon}R\longrightarrow k\longrightarrow0.
$$
与 $k$ 张量后所有 differential 都为零，所以
$\operatorname{Tor}_i^R(k,k)\cong k$ 对每个 $i\ge0$ 都非零。若 $k$ 是
perfect $R$-complex，则它有 bounded finite-projective resolution，从而高阶
Tor 必为零，矛盾。因此
$k\in\mathrm D^b\operatorname{Coh}(X)$ 但
$k\notin\operatorname{Perf}(X)$；警告 2.8 的差异已经出现在零维奇异
scheme 上。

## 2.3 dg enhancement

**定义 2.9.** 三角范畴 $\mathcal T$ 的 dg enhancement 是一个 pretriangulated dg category $\mathcal C$ 和一个三角等价
$$
H^0(\mathcal C)\xrightarrow{\sim}\mathcal T.
$$

**约定 2.10（本书固定的 B-side dg model）.** 设 $X$ 是 quasi-compact、
quasi-separated $k$-scheme。记
$\mathrm C_{\mathrm{dg}}(\operatorname{QCoh}X)$ 为 quasi-coherent sheaf
complexes 的 dg category；其 morphism complex 是 graded sheaf maps，
微分为 graded commutator。一个复形 $I$ 称为 h-injective，若对每个
acyclic complex $A$，complex
$\operatorname{hom}(A,I)$ acyclic。令 $\mathcal I_{\mathrm{dg}}(X)$ 为
h-injectives 的 full dg subcategory，并定义：

$$
\operatorname{Perf}_{\mathrm{dg}}(X)
=\{I\in\mathcal I_{\mathrm{dg}}(X):[I]\text{ 在 }\mathrm D_{\mathrm{qc}}(X)
\text{ 中 perfect}\},
$$
$$
\mathrm D^b_{\mathrm{dg}}\operatorname{Coh}(X)
=\{I\in\mathcal I_{\mathrm{dg}}(X):H^i(I)\text{ coherent 且仅有限多个非零}\}.
$$
第二行只在 $X$ noetherian 时使用。每次取一个 $\mathcal U$-小 skeleton，
以满足第一章的小性约定。相应 stable $\infty$ enhancement 可取
$\operatorname{Perf}_{\mathrm{dg}}(X)$ 的 dg nerve；本书不会在同一公式中
把 dg category 与其 dg nerve 混为一个对象。

**外部输入定理 2.11（h-injective enhancement 的存在与比较）.** 对上述
$X$，每个 quasi-coherent complex 有 h-injective replacement，并且
localization 诱导三角等价
$$
H^0\mathcal I_{\mathrm{dg}}(X)\simeq\mathrm D(\operatorname{QCoh}X).
$$
因此
$$
H^0\operatorname{Perf}_{\mathrm{dg}}(X)
\simeq\mathrm D_{\mathrm{perf}}(X).
$$
若 $X$ regular noetherian，则定理 2.7 进一步说明 full dg inclusion
$$
\operatorname{Perf}_{\mathrm{dg}}(X)
\hookrightarrow\mathrm D^b_{\mathrm{dg}}\operatorname{Coh}(X)
$$
是 quasi-equivalence。

**证明路线（外部输入）.** H-injective replacements 的存在属于
Grothendieck abelian categories 上的 unbounded derived-category 理论。
固定该模型后，最后一句的 quasi-fully-faithfulness 因 inclusion 为 full
而成立，essential surjectivity 则正是外部输入定理 2.7。来源定位见
Keller、标准 h-injective 理论与 Stacks Project Tag 0FDC。

**警告 2.11A.** 三角等价
$H^0(\mathcal C)\simeq\mathrm D^b\operatorname{Coh}(X)$ 本身不指定
$\mathcal C$ 的 quasi-equivalence 或 Morita 类型。本书通过约定 2.10 固定
模型；除非另引 enhancement uniqueness theorem，不从三角影子反推任意
两个 enhancements 等价。

## 2.4 Fourier-Mukai transforms

**定义 2.12.** 设 $X,Y$ 是光滑 proper 有限型 $k$-schemes，$p_X,p_Y$ 为 $X\times_kY$ 到两因子的投影。对 kernel
$$
K\in\operatorname{Perf}(X\times Y),
$$
其 triangulated Fourier--Mukai transform 是
$$
H^0\Phi_K:H^0\operatorname{Perf}_{\mathrm{dg}}(X)
\to H^0\operatorname{Perf}_{\mathrm{dg}}(Y),\qquad
\Phi_K(E)=\mathbf R p_{Y*}(\mathbf Lp_X^\ast E\otimes^{\mathbf L}K).
$$
所谓其 enhanced Fourier--Mukai transform，是约定 2.10 的 models 之间一个
dg quasi-functor $\Phi_K^{\mathrm{dg}}$，其 $H^0$ 与上式自然同构。由于
$p_X$ 具有有限 Tor-dimension、$p_Y$ proper 且 perfect，上式把 perfect
complexes 送到 perfect complexes；若删去这些保持性假设，公式未必以
$\operatorname{Perf}(Y)$ 为值域。

**外部输入定理 2.12A（kernel transform 的增强）.** 在定义 2.12 的
假设下，derived pullback、tensor 与 pushforward 可在 h-flat/h-injective
models 上导出，给出 $\Phi_K^{\mathrm{dg}}$；其在 dg/Morita homotopy category
中的同构类只依赖 $K$ 在 $\operatorname{Perf}(X\times Y)$ 中的同构类。
Kernel convolution 对应 quasi-functor composition。来源：Keller/Toen 的
derived Morita 理论与 Huybrechts 的 Fourier--Mukai 形式主义；卷积的
对象层计算见附录 D。

**命题 2.13.** 在定义 2.12 的假设下，对角嵌入 $\Delta:X\hookrightarrow X\times_kX$ 是 regular closed immersion，故 $\mathcal O_\Delta\in\operatorname{Perf}(X\times X)$；该 kernel 给出恒等 Fourier-Mukai transform。

**证明.** 令 $\Delta:X\to X\times X$ 为对角嵌入。由定义
$$
\Phi_{\mathcal O_\Delta}(E)=\mathbf R p_{2*}(\mathbf Lp_1^\ast E\otimes^{\mathbf L}\Delta_\ast\mathcal O_X).
$$
光滑性使对角嵌入 regular，因而 $\Delta_*\mathcal O_X$ 是 perfect。对 closed immersion 使用导出投影公式给出
$$
\mathbf Lp_1^\ast E\otimes^{\mathbf L}\Delta_\ast\mathcal O_X
\simeq \Delta_\ast(\mathbf L\Delta^*\mathbf Lp_1^\ast E)
\simeq \Delta_\ast E.
$$
因为 $p_2\circ\Delta=\operatorname{id}_X$，所以
$\mathbf R p_{2*}\Delta_\ast E\simeq E$。这些同构来自 derived functors 的
自然变换，故定理 2.12A 把它们提升为 enhanced functors 的自然等价。证毕。

**外部输入定理 2.14（Orlov 表示性，本文使用的窄版本）.** 设 $k$ 为
特征零代数闭域，$X,Y$ 为 smooth projective $k$-varieties。设
$k$-linear、exact、fully faithful functor
$$
F:\mathrm D^b\operatorname{Coh}(X)
\longrightarrow\mathrm D^b\operatorname{Coh}(Y)
$$
同时有 left 与 right adjoints。则存在按同构唯一的
$K\in\mathrm D^b\operatorname{Coh}(X\times_kY)$，使 $F$ 同构于
Fourier--Mukai transform $\Phi_K$；在此光滑情形 $K$ 也是 perfect。该
表示性是外部输入，不由“$F$ exact”或“$F$ fully faithful”在未核查
adjoints 时单独推出。来源：Huybrechts, *Fourier--Mukai Transforms in
Algebraic Geometry*, Theorem 5.14 (Orlov)。

**边界 2.14A.** 定理 2.14 不声称缺少所列 adjoints 的 functor、任意
non-fully-faithful exact functor、singular/nonproper scheme 上的 functor
或抽象三角 functor 都由 kernel 表示。HMS 若使用这些语境，必须引用相应
扩展，不能仍标作“Orlov representability”。

**解释 2.15.** Fourier-Mukai kernel 是 B-side 等价的具体候选函子。HMS 中若能把 A-side 生成对象的 endomorphism algebra 与 B-side 某个生成对象的 endomorphism algebra 识别，就常能得到 Morita 层面的等价；若还能构造 kernel，则得到更几何的 B-side 解释。

## 2.5 Landau-Ginzburg B-side 与 matrix factorizations

**定义 2.16（affine dg model）.** 设 $R$ 是 commutative $k$-algebra，
$w\in R$。有限秩 matrix factorization 是有限生成 projective
$\mathbb Z/2$-graded $R$-module $E=E_0\oplus E_1$ 与奇 endomorphism
$d_E$，满足
$$
d_E^2=w\operatorname{id}_E.
$$
对两个对象 $E,F$，令 morphism space 为
$\operatorname{Hom}_R^{\mathbb Z/2}(E,F)$，微分为
$$
\delta(f)=d_Ff-(-1)^{|f|}fd_E.
$$
因为
$$
\delta^2(f)=d_F^2f-fd_E^2=wf-fw=0,
$$
这些数据构成 $\mathbb Z/2$-graded dg category
$\operatorname{MF}^{\mathrm{fr}}_{\mathrm{dg}}(R,w)$。本书的
$\operatorname{MF}(\operatorname{Spec}R,w)$ 默认指其 pretriangulated、
idempotent-complete Morita envelope；若使用 raw finite-rank model 会显式
写上标 $\mathrm{fr}$。

对 nonaffine scheme $X$，局部 matrix factorizations 需要 sheafified mapping
complex、descent/absolute derived quotient 等额外模型选择。本书不把 affine
公式无说明地升级为全局定义。

**例 2.17.** 若 $w=0$，定义 2.16 正是 finite projective
$2$-periodic complexes。若 $w$ 是 non-zero-divisor，zero fiber
$\operatorname{Spec}(R/(w))$ 才进入下面的 hypersurface singularity
定理；“$w$ 有孤立临界点”不是定义 dg category 的必要条件，只影响其
properness、生成性和可计算性。

**外部输入定理 2.18（affine hypersurface 的 Orlov 比较）.** 设 $R$ 为
finite Krull-dimensional regular noetherian commutative ring，
$w\in R$ 为 non-zero-divisor，并令 $S=R/(w)$。Cokernel construction 把
finite-rank matrix factorization
送到 $S$-module，并在 idempotent completion 后诱导三角等价
$$
H^0\operatorname{MF}^{\mathrm{fr}}_{\mathrm{dg}}(R,w)^\pi
\xrightarrow{\sim}
\left(
\mathrm D^b(\operatorname{mod}_{\mathrm{fg}}S)
/\operatorname{Perf}(S)
\right)^\pi.
$$
该定理不覆盖 $w$ 为 zero-divisor、nonaffine descent 或 equivariant/graded
factorizations；这些版本需要各自的 Orlov/relative-singularity 输入。

## 2.6 B-side 的 HMS 检查项

一个 B-side 数据包必须至少给出：

1. 几何对象 $X$ 或 $(X,W)$；
2. 采用的增强模型：例如约定 2.10 的
   $\operatorname{Perf}_{\mathrm{dg}}(X)$、
   $\mathrm D^b_{\mathrm{dg}}\operatorname{Coh}(X)$，或明确写出 affine、
   nonaffine、graded/equivariant 哪一种 $\operatorname{MF}$；
3. 是否光滑、适当、奇异、栈化或非紧；
4. 生成对象或 tilting object；
5. 候选 Fourier--Mukai kernel 或 full generating subcategory；若压缩为
   endomorphism algebra，记录对象 idempotents 与全部 $A_\infty$ 运算；
6. 与 A-side 比较的系数、分次和 grading 数据；
7. 比较 raw enhancement、pretriangulated envelope 还是 perfect/Morita
   completion。

于是 B-side 不再是一个未注明模型的三角影子。Regular noetherian 情形允许把完美复形与有界 coherent 复形比较，dual numbers 则说明奇异情形不能沿用这一识别；Fourier--Mukai 核给出增强函子的链级来源，而 matrix factorization 把势函数的曲率写入微分平方。下一步要构造的 A-side 与这些对象在形式上截然不同，但比较时必须达到同样的增强精度。

## 练习

**练习 2.1.** 证明 quasi-isomorphism 在导出范畴中变成同构。

**练习 2.2.** 对仿射 scheme $X=\operatorname{Spec}A$，说明 perfect complexes 与 perfect dg $A$-modules 的关系。

**练习 2.3.** 逐步验证命题 2.13 中使用的投影公式和对角线恒等式。

**练习 2.4.** 给出一个奇异 scheme 的例子，并解释为什么 $\operatorname{Perf}(X)$ 与 $\mathrm D^b\operatorname{Coh}(X)$ 不应混同。
