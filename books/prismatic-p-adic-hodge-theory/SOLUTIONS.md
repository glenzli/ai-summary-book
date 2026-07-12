# 习题解答与提示

本文档给出《Prismatic / p-adic Hodge Theory》正式教材稿的习题解答。解答目标是覆盖正文和技术附录中的全部章末练习，并对每道题给出可检查的对象、结构和推理边界。

## 序章

**练习 0.1 解答.** 若只说 $A$ 是 $p$-完备环，可能指 ordinary inverse limit
$$
A\simeq\varprojlim_n A/p^n
$$
也可能指 derived $p$-complete。非 noetherian 或有高阶 torsion 时二者不同。Prismatic theory 中 base change、flatness 和 cohomology 都在 derived category 中运行，因此必须声明 derived completion。

**练习 0.2 解答.** 设 $D$ 是 filtered vector space。一个 vector-space isomorphism $f:D\to D'$ 不必满足 $f(\operatorname{Fil}^iD)=\operatorname{Fil}^iD'$。Classical de Rham comparison 要求 filtered $B_{\mathrm{dR}}$-linear isomorphism，不能只给裸同构。

**练习 0.3 解答.** 可取如下三类外部输入。Classical layer：Fontaine-Faltings-Tsuji comparison theorems，连接 de Rham、crystalline、semistable representations 与 period rings。Integral layer：Bhatt-Morrow-Scholze $A_{\inf}$ integral comparison theorem，输出带 Frobenius 和 lattice 信息的 $A_{\inf}$-complex。Prismatic layer：Bhatt-Scholze prismatic comparison theorem，给出 Hodge-Tate、de Rham、crystalline 和 etale specialization 的统一来源。若再选一个 prismatic representation-theoretic 输入，可取 Bhatt-Scholze prismatic $F$-crystal classification theorem。

## 第一章

**练习 1.1 解答.** 由 $0=x+(-x)$，
$$
0=\delta(x)+\delta(-x)+\frac{x^p+(-x)^p}{p}.
$$
若 $p$ 奇，则 $x^p+(-x)^p=0$，故 $\delta(-x)=-\delta(x)$。若 $p=2$，则
$$
\delta(-x)=-\delta(x)-x^2.
$$
统一写法由定义 1.1 的加法公式给出。

**练习 1.2 解答.** 若 $\phi$ 是 Frobenius lift，则
$$
\delta(x^2)=\frac{\phi(x)^2-x^{2p}}{p}
=2x^p\delta(x)+p\delta(x)^2.
$$
同理
$$
\delta(x^3)=3x^{2p}\delta(x)+3px^p\delta(x)^2+p^2\delta(x)^3.
$$

**练习 1.3 解答.** 令 $q=1+t$。则
$$
[p]_q=\frac{q^p-1}{q-1}=1+q+\cdots+q^{p-1}.
$$
在 $q=1$ 处取值为 $p$。这说明 $[p]_q$ 是 $p$ 的 $q$-变形；其常数项控制 crystalline 极限。但要说明它是 distinguished element，还必须检查相应 $\delta$-结构下 $\delta([p]_q)$ 为单位，这不是单靠 $q=1$ 的取值完成的。

## 第二章

**练习 2.1 解答.** 若 $d$ distinguished，则 $\delta(d)$ 是单位。由
$$
\phi(d)=d^p+p\delta(d)
$$
可解得
$$
p=\delta(d)^{-1}\phi(d)-\delta(d)^{-1}d^p\in(\phi(d))+(d).
$$

**练习 2.2 解答.** 对 $(A,(p))$，prism 第四条要求
$$
p\in(p)+\phi((p))A.
$$
因为右侧包含 $(p)$，该包含成立。

**练习 2.3 解答.** 在 $(X/A)_\Delta$ 中，对象可写为 prism probe
$(B,IB)$ 连同 map $\operatorname{Spf}(B/IB)\to X$ over
$\operatorname{Spf}(A/I)$。引理 2.4A 保证 probe ideal 必为 $IB$。一个态射
$$
(B,IB)\to(B',IB')
$$
需要是与底 prism $(A,I)$ 相容的 $\delta$-环态射，并使诱导的
$$
\operatorname{Spf}(B'/IB')\to\operatorname{Spf}(B/IB)\to X
$$
等于目标对象给定的结构态射。也就是说，$\delta$-结构、Cartier divisor ideal、底 prism map 和到 $X$ 的 quotient map 四者必须同时交换。

## 第三章

**练习 3.1 解答.** $\Delta_{R/A}\otimes_A^L A/I$ 是直接 modulo $I$ 的 Hodge-Tate specialization。De Rham specialization 先沿 $\phi_A$ pullback：
$$
A\otimes_{A,\phi_A}^L\Delta_{R/A},
$$
再 modulo $I$。Frobenius pullback 改变 $A$-module structure，因此两个
construction 不同。严格地说，de Rham comparison 还对所得 tensor product
作 derived $p$-completion，而 Hodge--Tate specialization 的定义没有这个
额外步骤。

**练习 3.2 解答.** 形式推论 3.6 使用 conjugate filtration。若 filtration 的 associated graded 只有有限多项，且每个
$$
R\Gamma(X,\Omega^i)[-i]\{-i\}
$$
是 perfect $A/I$-complex，则从最高非零 filtration 层向下归纳即可。每一步有 exact triangle
$$
F^{i+1}\to F^i\to \operatorname{gr}^i(F),
$$
其中右项 perfect，归纳假设左项 perfect。Perfect complexes 在 exact triangle 中满足 two-out-of-three，故 $F^i$ perfect。最后得到 $\overline\Delta_{X/A}$ perfect。

**练习 3.3 解答.** 错误写法：
$$
R\Gamma_\Delta(X/A)\cong R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z_p).
$$
左侧是 $A$-complex 带 Frobenius，右侧是 $\mathbf Z_p$-complex。正确比较需要 perfect prism、invert $I$、derived Frobenius fixed points 和 modulo $p^n$/inverse limit。

## 第四章

**练习 4.1 解答.** 约定 $\mathbf Q_p(1)$ 由 cyclotomic character 给出。Hodge-Tate grading 的正负依文献 convention 可能相反；本书只要求读者说明 convention。若采用常见 convention，$\mathbf Q_p(n)$ 的 Hodge-Tate weight 为 $-n$，则 $D_{B_{\mathrm{HT}}}(\mathbf Q_p(n))$ 集中在对应的第 $-n$ 层。

**练习 4.2 解答.** Crystalline representation 是 $B_{\mathrm{cris}}$-admissible，de Rham representation 是 $B_{\mathrm{dR}}$-admissible。包含 $\operatorname{Rep}_{\mathrm{cris}}\subset\operatorname{Rep}_{\mathrm{dR}}$ 依赖 period rings 之间的结构和 Fontaine admissibility theorem，不是定义本身。

**练习 4.3 解答.** 在
$$
B_{\mathrm{dR}}\otimes_K H^n_{\mathrm{dR}}(X/K)
\simeq
B_{\mathrm{dR}}\otimes_{\mathbf Q_p}H^n_{\mathrm{et}}(X_{\overline K},\mathbf Q_p)
$$
中，filtration 位于 $B_{\mathrm{dR}}$ 及 de Rham 侧的 tensor product 上；$G_K$ 作用在右侧通过 $B_{\mathrm{dR}}$ 和 etale cohomology 共同给出。比较同构必须同时尊重这些结构。

## 第五章

**练习 5.1 解答.** $R\Gamma_{A_{\inf}}(\mathfrak X)$ 是 $A_{\inf}$-complex，带 Frobenius 和 integral torsion/lattice 信息。若只看成 $\mathbf Z_p$-complex，会遗忘 $\theta$-specialization、crystalline specialization 和 Frobenius module structure。

**练习 5.2 解答.** Perfect prism $(A_{\inf},\ker\theta)$ 的 quotient 是 $\mathcal O_C$。Breuil-Kisin prism $(\mathfrak S,(E(u)))$ 的 quotient 是 $\mathcal O_K$。前者适合 perfectoid base 和 $A_{\inf}$-cohomology，后者适合 discretely valued base 和 Breuil-Kisin cohomology。

**练习 5.3 解答.** 形式推论 5.11 使用两个外部输入：第一，BMS $A_{\inf}$ object 与 prismatic cohomology over $(A_{\inf},\ker\theta)$ 的识别，属于 integral layer；第二，Bhatt-Scholze prismatic comparison theorem，属于 prismatic layer。二者组合后才得到统一 specialization 解释。

## 第六章

**练习 6.1 解答.** Crystal condition 要求沿 prismatic site 中的 morphism pullback 后有指定同构，并满足 cocycle compatibility。普通 sheaf 只给 restriction maps；crystal 要求这些 restriction 在结构层变换后表现为刚性 base change。

**练习 6.2 解答.** 若该 probe 的 ideal $J=(d)$，则
$$
\mathcal E[1/J]=\mathcal E[1/d].
$$
若换生成元 $d'=ud$，其中 $u$ 是单位，则 inverting $d'$ 与 inverting $d$ 给出同一局部化。

**练习 6.3 解答.** 定理 6.9 的左侧是带 morphisms、tensor operations
和 descent 的 $F$-crystals 范畴，右侧是 crystalline $\mathbf Z_p$-lattices
及其 $G_K$-equivariant maps 的范畴。Etale realization 不只给每个对象一个
cohomology group；它给 functor，并且定理同时断言 full faithfulness 与
essential surjectivity。因此结论是范畴等价，而不是两个 cohomology
complexes 的拟同构。

## 第七章

**练习 7.1 解答.** 若 $x\in N^{\ge i+1}_{\mathrm{naive}}M$，则 $\varphi(x)\in d^{i+1}M\subset d^iM$，所以 $x\in N^{\ge i}_{\mathrm{naive}}M$。

**练习 7.2 解答.** Semilinear Frobenius 的 $A$-linear source 是沿
$\phi_A$ 的 scalar pullback；在 complete category 中应使用
$C^{(1)}=C\widehat\otimes_{A,\phi_A}^LA$。定理 7.3 的同构
$C^{(1)}\simeq L\eta_I C$ 把 Frobenius 的 divisibility 编码进
$L\eta_I$；随后 $L\eta_I C\to C$ 才给原 Frobenius linearization。若把
filtration 直接放在 $C$ 上，就丢掉这个 source twist。

**练习 7.3 解答.** 定义 7.5 的 source 是
$\mathcal N^{\ge i}\widehat\Delta_S\{i\}$，两张 maps 的共同 target 都是
$\widehat\Delta_S\{i\}$，fibre 位于 $p$-complete
$D(\mathbf Z_p)$；derived modulo $p^r$ 后才位于
$D(\mathbf Z/p^r)$。$\widehat\Delta_S$ 是 BMS2 经 quasisyntomic descent
得到的 Nygaard-complete object，而 relative $R\Gamma_\Delta(X/A)$ 的
Nygaard filtration 位于其 completed Frobenius twist。把两者凭记号相似直接
替换，会遗漏这项深 comparison 及其 hypotheses。

## 第八章

**练习 8.1 解答.** Prismatization 把 prismatic probes 和 crystals 组织为 stack-theoretic language。它解释 $\mathcal D_{qc}(\mathrm{WCart})$ 与 crystals 的关系，但仍依赖 prismatic data、完备性和 quasisyntomic/lci 假设。因此它不是把 prismatic cohomology 换成普通 stack cohomology 的无条件替代定义。

**练习 8.2 解答.** $F$-crystal 是 crystal 加 Frobenius after inverting prism ideal；$F$-gauge 通常还编码 filtration 或 gauge data。前者偏向 vector bundle with Frobenius，后者偏向 filtered/Frobenius package，常出现在 prismatization 或 display 语境。

**练习 8.3 解答.** 例如 syntomic Steenrod operations 属于 2025 研究边界。它依赖 spectral syntomic and operations 结构，不能用于证明基础 prismatic comparison theorem；否则会把后续增强结构倒用到基础理论中。

## 第九章

**练习 9.1 解答.** 若 $X$ relative dimension 为 $d$ 且 smooth，则 $\Omega^i_{X/(A/I)}=0$ for $i>d$。因此 associated graded 中 $i>d$ 的项为零，$i<0$ 也不出现。

**练习 9.2 解答.** 命题 9.7 的谱序列可写为
$$
E_1^{i,j}=H^{i+j}\left(R\Gamma(X,\Omega^i)[-i]\{-i\}\right)
\Rightarrow H^{i+j}(\overline\Delta_{X/A}).
$$
因为 shift $[-i]$ 满足 $H^m(C[-i])=H^{m-i}(C)$，所以
$$
E_1^{i,j}\cong H^j(X,\Omega^i)\{-i\}.
$$
在该记号下，第二指标 $j$ 记录原始 sheaf cohomology degree，总次数仍为目标的 $i+j$。

**练习 9.3 解答.** 错误论证可以写成：定理 9.4 给出 $\overline\Delta_{X/A}$ 的 conjugate filtration，其 graded pieces 由 $\Omega^i[-i]\{-i\}$ 控制；因此 de Rham complex $R\Gamma_{\mathrm{dR}}(X/(A/I))$ 的 Hodge filtration 也有相同 strictness。错误发生在第一步之后：conjugate filtration 的对象是 Hodge-Tate specialization $\overline\Delta_{X/A}$，而 Hodge filtration 的对象是 $p$-completed de Rham specialization $\phi_A^\ast R\Gamma_\Delta(X/A)\widehat\otimes_A^L A/I$。定理 9.9 只是 unfiltered comparison；没有额外 filtered comparison 时，不能从一个 specialization 的滤过性质推出另一个 specialization 的滤过性质。

## 第十章

**练习 10.1 解答.** 在 crystalline prism $(A,(p))$ 中，quotient 为 $A/p$。对任意 $\bar x\in A/p$，都有
$$
p\bar x=0.
$$
因此
$$
(A/p)[p^\infty]=A/p=(A/p)[p],
$$
也就是所有 $p$-power torsion 已经由一次乘以 $p$ 杀死。故 boundedness 条件中的 $p^\infty$-torsion 有界，且界可取 $N=1$。

**练习 10.2 解答.** 标准相容关系之一是
$$
FV=VF=p,\qquad F\,d\,V=d
$$
在 de Rham-Witt formalism 中成立。这里 $F$ 是 Frobenius，$V$ 是 Verschiebung，$d$ 是 differential。其来源是 Illusie 的 de Rham-Witt complex 理论；本书第十章只把这些关系作为外部输入定义的一部分使用，不在内部重建 de Rham-Witt complex。

**练习 10.3 解答.** 对 $f(T)=T^n$，
$$
\nabla_q(f)=\frac{q^nT^n-T^n}{qT-T}
=\frac{q^n-1}{q-1}T^{n-1}
=[n]_qT^{n-1}.
$$

## 第十一章

**练习 11.1 解答.** 对两项 complex
$[M\xrightarrow{u-1}M]$，其 $H^0$ 为 $\ker(u-1)$，$H^1$ 为
$\operatorname{coker}(u-1)$。在 derived fixed point 中，后者记录 ordinary
fixed points 看不见的 obstruction。

**练习 11.2 解答.** 若 $\varphi(am)=\phi_A(a)\varphi(m)$，则
$$
(\varphi-1)(am)=\phi_A(a)\varphi(m)-am,
$$
一般不等于 $a(\varphi-1)(m)$，所以 $\varphi-1$ 不是 $A$-linear。定理
11.2 先 modulo $p^r$，再把系数限制到被 Frobenius 固定的
$\mathbf Z/p^r$；此时 $\varphi-1$ 是 $\mathbf Z/p^r$-linear，fibre 位于
$D(\mathbf Z/p^r)$。

**练习 11.3 解答.** 定理 11.8 (2) 的 source 只识别
$\mathbf Z/p^r(i)$ 与 $\tau^{\le i}R\psi_*\mathbf Z/p^r(i)$。去掉
$\tau^{\le i}$ 会额外断言 nearby-cycles complex 在 degrees $>i$ 的部分也由
syntomic complex 控制；这既不由原拟同构形式推出，也不是 BMS2,
Theorem 10.1 的结论。

## 第十二章

**练习 12.1 解答.** 定义 12.4 的 datum 本身就是
$$
(\phi_{\mathfrak S}^*M)[1/E(u)]\xrightarrow{\sim}M[1/E(u)],
$$
所以 localized linearization 可逆是定义条件，不是由 finite generation
推出的结论。这个 localization 没有 invert $p$，故不能推出 $M[1/p]$
finite projective、Frobenius 在 $M[1/p]$ 上可逆，或 $p$-power torsion
消失。Rank one 时若 map 为乘以 $a$，条件只说明
$a\in\mathfrak S[1/E(u)]^\times$。

**练习 12.2 解答.** Breuil-Kisin module 的底环是
$\mathfrak S=W(k)[[u]]$，定义只要求有限生成，并要求 linearized Frobenius
在 invert $E(u)$ 后为同构。BKF module 的底环是 $A_{\inf}$，要求有限
呈示、$N[1/p]$ finite free，并有
$N[1/\xi]\simeq N[1/\phi(\xi)]$ 的 semilinear Frobenius。二者的底环、
反演 divisor 与 finiteness 条件均不同。

**练习 12.3 解答.** Rational filtered $\varphi$-module 通常是在张量 $\mathbf Q_p$ 或 invert 相应 period 元素后得到的对象。Localization 会遗忘 integral lattice 和 torsion。最简单的线性代数例子是同一个 $V=\mathbf Q_p$ 中的两个 lattice
$$
\mathbf Z_p\subset V,\qquad p\mathbf Z_p\subset V.
$$
二者张量 $\mathbf Q_p$ 后都等于 $V$，但作为 integral lattice 不同。因此只知道 rational filtered $\varphi$-module，不能唯一恢复 Galois lattice、BK module 或 BKF module 中的 integral 信息。

## 第十三章

**练习 13.1 解答.** 系数 crystal $\mathcal E$ 给出每个 prismatic probe 上的 module，并要求 pullback compatibility。若取 $\mathcal E=\mathcal O_\Prism$，带系数 cohomology 退化为普通 prismatic cohomology。

**练习 13.2 解答.** 三个差异如下。第一，Hodge-Tate prismatic crystal 通常位于 Hodge-Tate specialization 或 $\overline{\mathcal O}_\Delta$ 层面，而 prismatic $F$-crystal 位于 integral prismatic site 上。第二，Hodge-Tate prismatic crystal 的核心结构常与 rationalization、connections 或 Higgs-type data 相关；prismatic $F$-crystal 的核心结构是 Frobenius after inverting prism ideal。第三，前者在本书中属于 non-abelian/rational 研究边界，后者用于 crystalline Galois lattices 的基础分类输入。

**练习 13.3 解答.** 若省略 finite projective 条件，系数对象可能不再表现为 vector bundle，pullback 后也可能不保 perfectness。这样 comparison theorem 中的 base change、duality 和 tensor operations 都可能失效。

## 第十四章

**练习 14.1 解答.** Artin stack 方向需要控制 cotangent complex、smooth atlases 和 descent。若只把 scheme 的 prismatic cohomology 逐字套到 stack 上，可能丢失 atlas independence 和 higher stabilizer 的贡献。

**练习 14.2 解答.** Shimura varieties 的 prismatic realization 需要 integral model、level structure、comparison functor 和 crystalline realization 的兼容性。它不是单纯的 cohomology group 计算，而是 functorial realization problem。

**练习 14.3 解答.** 按命题 14.9，至少需要如下数据：finite flat group scheme 的高度条件，通常这里是 height one；基底的 characteristic，Mondal-Olsson 条目处在 positive characteristic 语境；所使用的 crystalline site 或 prismatic site；目标 prismatic $F$-gauge 的定义范畴；以及 Frobenius 和 Verschiebung 数据如何对应。若缺少 height 或 $F/V$ 数据，就无法判断输出是否真能恢复 Dieudonne module。

## 第十五章

**练习 15.1 解答.** 一个章节若只列定义和定理名，没有对象构造、假设说明、证明或错误边界，就仍是大纲。收口章节至少应能让读者检查每个对象的输入、输出和使用范围。

**练习 15.2 解答.** 判定步骤如下。第一，按约定 14.10 检查是否为一手文献，版本是否已核查，额外 hypotheses 是否明确。第二，按说明 15.11 检查 locator、编号、convention 和 production 状态。第三，判断它是否只属于应用边界，还是可进入基础定理链。一个新的预印本若只给出前沿应用、没有被独立核查到稳定版本，或其 hypotheses 依赖新范畴尚未在书内定义，则只能进入研究边界；不能用于证明 prism、comparison theorem 或 $F$-crystal classification 的基础结论。

**练习 15.3 解答.** 一个合格 locator 表项可写为：

| 字段 | 内容 |
| --- | --- |
| 本书编号 | 定理 11.8 |
| Source | Bhatt-Morrow-Scholze, syntomic sheaves and nearby cycles 所在论文或正式出版版本 |
| Version | arXiv/出版版本号与下载日期 |
| External theorem | syntomic-to-etale comparison, including Frobenius/Nygaard fibre convention |
| Hypotheses | smooth/proper 或 quasisyntomic 条件、base prism、mod $p^r$ 或 derived complete convention |
| 本书用途 | 支撑第十一章 syntomic tower 与 etale comparison 的 fixed-point/fibre construction |
| 风险 | twist sign、truncation、nearby cycles 版本需与附录 F 逐项核对 |

这样的 locator 不只写定理名，而是记录 source、版本、假设和本书使用方式。

## 附录 A

**练习 A.1 解答.** 对 $A=\mathbf Z_p$、$J=(p)$、$M=A$，
$$
M^{\wedge,L}_J=R\varprojlim_nK_{\mathbf Z_p}(p^n).
$$
因为 $p^n$ 在 $\mathbf Z_p$ 中是 nonzerodivisor，
$K_{\mathbf Z_p}(p^n)\simeq\mathbf Z_p/p^n$。Transition maps 满射，故
没有 $\varprojlim^1$ 项，并得到
$$
R\varprojlim_n\mathbf Z_p/p^n\simeq\mathbf Z_p.
$$
故 $\mathbf Z_p$ 已 derived $p$-complete。

**练习 A.2 解答.** Transition map
$$
M/p^{n+1}M\to M/p^nM
$$
总是满射，因为任意 $m\bmod p^nM$ 可由同一个 $m\bmod p^{n+1}M$ 提升。满射 inverse system 满足 Mittag-Leffler 条件。无 $p$-torsion 不是满射性的必要条件，但它保证这些 quotients 对 $p$-adic filtration 的解释没有隐藏 $p$-torsion 干扰。

**练习 A.3 解答.** 按附录 A 的 convention，
$K_A(f,g)=K_A(f)\otimes_AK_A(g)$，每个单变量 Koszul complex 的右端
$A$ 位于 degree $0$。因此三项 complex 为
$$
0\to A\xrightarrow{d^{-2}}A\oplus A\xrightarrow{d^{-1}}A\to0,
$$
degree 分别为 $-2,-1,0$。若 degree $-1$ 的两个 summands 依次对应
$K_A(f)^{-1}\otimes K_A(g)^0$ 与
$K_A(f)^0\otimes K_A(g)^{-1}$，则可取
$$
d^{-2}(a)=(-ga,fa),\qquad d^{-1}(b,c)=fb+gc.
$$
直接计算得 $d^{-1}d^{-2}(a)=-fga+gfa=0$。同时改变 degree $-1$
某一 summand 的基会得到等价的符号 convention。

## 附录 B

**练习 B.1 解答.** 对 $A=W(k)$，crystalline prism 为 $(W(k),(p))$，quotient 为 $W(k)/p\simeq k$。在 $k$ 中 $p=0$，所以
$$
k[p^\infty]=k=k[p].
$$
因此 $p^\infty$-torsion 由一次乘以 $p$ 杀死，boundedness 成立。

**练习 B.2 解答.** 若 $E(u)$ 是 Eisenstein polynomial，则可写为
$$
E(u)=u^e+p\,a(u)
$$
其中 $a(u)$ 的常数项为单位。由此 $E(u)\in(p,u^e)$，并且 $u^e=E(u)-p\,a(u)\in(E(u),p)$。因此 $(p,E(u))$ 和 $(p,u^e)$ 给出相同的完备拓扑；而 $(p,u^e)$-adic topology 与 $(p,u)$-adic topology 等价。

**练习 B.3 解答.** 令 $q=1+t$，则
$$
[p]_q=\frac{(1+t)^p-1}{t}
=p+\binom p2t+\binom p3t^2+O(t^3).
$$
因此到 $t^2$ 项为
$$
p+\binom p2t+\binom p3t^2,
$$
常数项为 $p$。

## 附录 G

**练习 G.1 解答.** 对 $A=\mathbf Z_p[[T]]$，其 $p$-adic truncations 为
$$
\operatorname{Spec}(A/p^nA)=\operatorname{Spec}\left(\mathbf Z_p[[T]]/p^n\right).
$$
这些 schemes 随 $n$ 由自然 quotient maps 组成 inverse system，形式极限即 $\operatorname{Spf}(\mathbf Z_p[[T]])$。

**练习 G.2 解答.** 若覆盖族只有一个同构 $V\xrightarrow{\sim}U$，则
$$
V\times_UV\simeq V.
$$
Sheaf equalizer 条件变成
$$
F(U)\to F(V)\rightrightarrows F(V)
$$
为 equalizer。两个平行箭头都由同一个同构的两个投影诱导，因而相等；同时 presheaf 对同构给出双射 $F(U)\simeq F(V)$。所以 equalizer 条件自动成立。

**练习 G.3 解答.** Prismatic cohomology 是 site 上结构层或系数 complex 的 cohomology。一般 sheaf 的普通 global sections 不是 exact functor，且覆盖 descent 会产生高阶 Cech 或 hypercover cohomology。若只取 $\Gamma((X/A)_\Delta,\mathcal O_\Delta)$，会丢失高阶 glueing obstruction。因此必须定义为
$$
R\Gamma((X/A)_\Delta,\mathcal O_\Delta).
$$

## 附录 H

**练习 H.1 解答.** 若 $p=2$，
$$
C_2(X,Y)=\frac{X^2+Y^2-(X+Y)^2}{2}=-XY.
$$
若 $p=3$，
$$
C_3(X,Y)=-(X^2Y+XY^2).
$$

**练习 H.2 解答.** 给定 Frobenius lift $\phi$ 并令 $\delta(x)=(\phi(x)-x^p)/p$。则
$$
\begin{aligned}
p\delta(x+y)
&=\phi(x+y)-(x+y)^p\\
&=\phi(x)+\phi(y)-(x+y)^p\\
&=x^p+y^p+p\delta(x)+p\delta(y)-(x+y)^p\\
&=p\delta(x)+p\delta(y)+x^p+y^p-(x+y)^p.
\end{aligned}
$$
除以 $p$，得到
$$
\delta(x+y)=\delta(x)+\delta(y)+\frac{x^p+y^p-(x+y)^p}{p}.
$$
整数性由命题 H.1 保证；$p$-torsionfree 保证除以 $p$ 的等式在 $A$ 中唯一。

**练习 H.3 解答.** 因为 $d$ distinguished，$\delta(d)$ 是单位。由
$$
\phi(d)=d^p+p\delta(d)
$$
在 $A/(d)$ 中得到
$$
\overline{\phi(d)}=p\,\overline{\delta(d)}.
$$
而 $\overline{\delta(d)}$ 仍是 $A/(d)$ 中的单位。因此 $\phi(d)$ 在 $A/(d)$ 中与 $p$ 相差单位倍。

## 附录 I

**练习 I.1 解答.** Sheaf condition 处理覆盖方向：给定覆盖 $U_i\to U$，要求局部截面在交叠上一致时可唯一粘合。Crystal condition 处理 thickening 或 probe morphism 方向：对 $u:T'\to T$，要求
$$
u^\ast\mathcal E(T)\xrightarrow{\sim}\mathcal E(T')
$$
为同构。前者是 glueing，后者是沿增厚方向的 rigidity。

**练习 I.2 解答.** 若 $P$ 是 finite projective $A$-module，则存在 $Q$ 和整数 $n$ 使
$$
P\oplus Q\simeq A^n.
$$
对任意 $A\to B$ 张量 $B$ 得
$$
(P\otimes_AB)\oplus(Q\otimes_AB)\simeq B^n.
$$
因此 $P\otimes_AB$ 是 finite free module 的 direct summand，故 finite projective。

**练习 I.3 解答.** 在 $V=\mathbf Q_p$ 中，
$$
T_1=\mathbf Z_p,\qquad T_2=p\mathbf Z_p
$$
都是 lattice。它们不同，因为 $1\in T_1$ 但 $1\notin T_2$。然而
$$
T_1\otimes_{\mathbf Z_p}\mathbf Q_p\simeq\mathbf Q_p\simeq T_2\otimes_{\mathbf Z_p}\mathbf Q_p.
$$
所以 rationalization 不唯一决定 integral lattice。

## 附录 J

**练习 J.1 解答.** 若 $F:M\to M$ 满足
$$
F(am)=\phi_A(a)F(m),
$$
则其 linearized map 为
$$
A\otimes_{A,\phi_A}M\to M,\qquad a\otimes m\mapsto aF(m).
$$
该 map 是 $A$-linear，因为 semilinearity 已被 tensor product 的右侧 $A$-module structure 吸收。

**练习 J.2 解答.** 令 $D=k e_1\oplus k e_2$，定义
$$
\operatorname{Fil}^0D=D,\qquad
\operatorname{Fil}^1D=k e_1,\qquad
\operatorname{Fil}^2D=0.
$$
则
$$
\operatorname{gr}^0D=D/\operatorname{Fil}^1D\simeq k e_2,\qquad
\operatorname{gr}^1D=\operatorname{Fil}^1D/\operatorname{Fil}^2D\simeq k e_1,
$$
其余 graded pieces 为零。

**练习 J.3 解答.** 在 $V=\mathbf Q_p^2$ 中可取
$$
T_1=\mathbf Z_p e_1\oplus\mathbf Z_p e_2,\qquad
T_2=\mathbf Z_p e_1\oplus p\mathbf Z_p e_2.
$$
二者都是 lattice，且 $T_2\subsetneq T_1$。张量 $\mathbf Q_p$ 后都给出 $V$，但 integral lattice 不同。

## 附录 K

**练习 K.1 解答.** 对 $R=(A/I)\langle T_1,T_2\rangle$，有
$$
\Omega^0=R,\quad
\Omega^1=R\,dT_1\oplus R\,dT_2,\quad
\Omega^2=R\,dT_1\wedge dT_2.
$$
因此 Hodge-Tate associated graded 只出现在 $i=0,1,2$ 三层，分别带 twist $\{0\},\{-1\},\{-2\}$ 和 shifts $0,-1,-2$。

**练习 K.2 解答.** 对 $C=\mathbf Z_p$、$\varphi=p$，map $\varphi-1$ 是乘以 $p-1$。由于 $p-1\in\mathbf Z_p^\times$，该映射为同构，因此
$$
C^{\varphi=1}\simeq0.
$$

**练习 K.3 解答.** 是否 invert $E(u)$ 后 $p$ 为单位取决于 $p$ 与 $E(u)$ 的关系。在 $\mathfrak S[1/E(u)]$ 中只强制 $E(u)$ 可逆，不自动强制 $p$ 可逆。由于 $E(u)$ Eisenstein，$p$ 与 $u^e$ 在拓扑上相关，但 $p$ 不因此成为 $\mathfrak S[1/E(u)]$ 的单位。
