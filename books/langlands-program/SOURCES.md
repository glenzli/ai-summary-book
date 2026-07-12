# 资料源

本文档记录《Langlands 纲领》的主要资料源。正文不得逐句翻译资料源；资料源用于校验定义、定理边界、标准术语和历史归属。

## 总览与 Langlands 纲领

- Joseph Bernstein and Stephen Gelbart, eds., *An Introduction to the Langlands Program*, Birkhauser/Springer, 2003. 作为全书路线和术语总览的主要参考。
- Robert P. Langlands, *Problems in the Theory of Automorphic Forms*, 1970. 作为原始纲领背景参考。
- James Arthur, *The Endoscopic Classification of Representations: Orthogonal and Symplectic Groups*, AMS, 2013. 用于后续 Arthur 参数和 endoscopy 章节。

## 代数数论、adeles 和类域论

- J. W. S. Cassels and A. Frohlich, eds., *Algebraic Number Theory*, Academic Press, 1967.
- Serge Lang, *Algebraic Number Theory*, Springer.
- Jurgen Neukirch, *Algebraic Number Theory*, Springer.
- J. S. Milne, *Class Field Theory*, online notes. 用于类域论陈述和 Artin reciprocity 归一化。
- Andre Weil, *Basic Number Theory*, Springer, especially Chapter IV. 用于 adeles、ideles、$C_K^1$ 紧性、Hecke quasi-character 的 unitary twist 分解和 Tate thesis 背景；也用于核对 $\mathbb Q$ 上 Dirichlet character 的 idelic 分解、无穷符号分量与 Euler convention。函数域中 degree/norm 的离散像必须单独保留。
- John Tate, class field theory articles in Cassels-Frohlich. 用于 class formations、Tate cohomology 和 reciprocity maps。
- John Tate, *Number Theoretic Background*, in *Automorphic Forms, Representations and L-functions*, Proc. Symp. Pure Math. 33, Part 2, AMS, 1979, pp. 3-26. 用于局部/全局 Weil 群和类域论归一化接口。

## Tate thesis 与 `GL(1)`

- John Tate, *Fourier Analysis in Number Fields and Hecke's Zeta-Functions*, thesis, 1950; reprinted in Cassels-Frohlich. 用于 Hecke character 的 idelic 定义和局部 Euler factors；练习 2.2/V.3 的 convention 特别固定 finite-unit restriction $\widehat\chi^{-1}$、由 $\chi(-1)$ 决定的无穷分量，以及 $\omega_{\chi,p}(p)=\chi(p)$。
- Dinakar Ramakrishnan and Robert J. Valenza, *Fourier Analysis on Number Fields*, Springer.
- Pierre Deligne, *Les constantes des equations fonctionnelles des fonctions L*, in *Modular Functions of One Variable II*, Lecture Notes in Mathematics 349, Springer, 1973, pp. 501-597. 用于 Weil-Deligne 表示、局部常数和加法特征/测度依赖。

## 自守形式、自守表示和 `GL(n)`

- Stephen Gelbart, *Automorphic Forms on Adele Groups*, Princeton University Press.
- Armand Borel and Harish-Chandra, *Arithmetic Subgroups of Algebraic Groups*, Annals of Mathematics 75 (1962), pp. 485-535；Gunter Harder, *Chevalley groups over function fields and automorphic forms*, Annals of Mathematics 100 (1974), pp. 249-306. 用于“无 proper rational parabolic”和 adelic quotient modulo split center 紧性的判准；正文只调用 13.8.1 的接口形式。
- Daniel Bump, *Automorphic Forms and Representations*, Cambridge University Press.
- Dorian Goldfeld and Joseph Hundley, *Automorphic Representations and L-Functions for the General Linear Group*, Cambridge University Press.
- Herve Jacquet and Robert P. Langlands, *Automorphic Forms on GL(2)*, Springer.
- Daniel Flath, *Decomposition of representations into tensor products*, in *Automorphic Forms, Representations and L-functions*, Proc. Symp. Pure Math. 33, Part 1, AMS, 1979, pp. 179-183. 用于不可约可容许自守表示的 restricted tensor product 分解。
- Roger Godement and Herve Jacquet, *Zeta Functions of Simple Algebras*, Lecture Notes in Mathematics 260, Springer, 1972. 用于 `GL(n)` 标准 L 函数、局部 zeta 积分和函数方程。
- Herve Jacquet, Ilya Piatetski-Shapiro, and Joseph Shalika, *Rankin-Selberg Convolutions*, American Journal of Mathematics 105 (1983), pp. 367-464. 用于 `GL(n)\times GL(m)` Rankin-Selberg L 函数和局部-整体因子。
- Herve Jacquet and Joseph Shalika, *On Euler products and the classification of automorphic representations I, II*, American Journal of Mathematics 103 (1981), pp. 499-558 and 777-815. 用于 Euler 乘积与强重数一。
- Ilya Piatetski-Shapiro and James Cogdell, works on converse theorems for `GL(n)`. 用于 `GL(n)` converse theorem 和函子性检测。
- I. N. Bernstein and A. V. Zelevinsky, works on induced representations of reductive p-adic groups. 用于 `GL(n)` 的局部分类和 Langlands quotient 接口。
- A. V. Zelevinsky, works on the classification of irreducible representations of `GL(n)` over non-Archimedean local fields. 用于 segments 和 multisegments。
- Marko Tadic, works on classification of unitary representations of `GL(n)` over local fields. 用于 tempered/unitary classification 背景。
- James Arthur and Laurent Clozel, *Simple Algebras, Base Change, and the Advanced Theory of the Trace Formula*, Princeton University Press. 用于 solvable base change 和 automorphic induction。
- Stephen Gelbart and Herve Jacquet, works on symmetric square lifting for `GL(2)`. 用于 symmetric square functoriality。
- Henry Kim and Freydoon Shahidi, works on symmetric power and tensor product functoriality. 用于低阶 symmetric power lifts 和若干 tensor product lifts。
- Dinakar Ramakrishnan, works on tensor product transfer for `GL(2)\times GL(2)`. 用于张量积函子性例子。
- Henry Kim, works on exterior square transfer. 用于 `GL(4)` exterior square lift 等低阶外方幂例子。
- Freydoon Shahidi, works on the Langlands-Shahidi method. 用于一般还原群中由抛物子群 adjoint action 得到的 L 函数。
- Freydoon Shahidi, *Eisenstein Series and Automorphic L-Functions*, AMS Colloquium Publications. 用于 Langlands-Shahidi local coefficient、$\gamma$ 因子和全局函数方程。
- Henry Kim and Freydoon Shahidi, works using the Langlands-Shahidi method for symmetric powers and functorial lifts. 用于低阶函子性与解析性质的实例。
- James Arthur, *An Introduction to the Trace Formula*, in standard trace formula references. 用于后续离散谱、稳定迹和全局 packet 背景。
- James Arthur, *The Trace Formula in Invariant Form*, Annals of Mathematics 114 (1981), pp. 1-74. 用于截断、加权轨道积分及不变迹公式的外部接口；稳定化仍需后续 Arthur-endoscopy 文献。
- Harish-Chandra and Robert Langlands, works on Eisenstein series and spectral decomposition. 用于残余谱、常数项和全局 L 函数解析性质背景。
- Robert Langlands, *On the Functional Equations Satisfied by Eisenstein Series*, Springer Lecture Notes. 用于 Eisenstein series、constant term formula、intertwining operators 和谱分解背景。
- C. Moeglin and J.-L. Waldspurger, works on residual spectrum and Eisenstein series. 用于 `GL(n)` residual spectrum 和 Arthur 参数背景。
- Laurent Lafforgue, *Chtoucas de Drinfeld et correspondance de Langlands*, Inventiones Mathematicae. 用于函数域 `GL(n)` 全局 Langlands。
- Laurent Clozel, works on algebraic automorphic representations. 用于 regular algebraic automorphic representations 和 Galois 表示接口。
- Michael Harris, Richard Taylor, Laurent Clozel, Richard Taylor, Peter Scholze, Ana Caraiani, and Harris-Lan-Taylor-Thorne works on Galois representations attached to automorphic forms. 用于数域 regular algebraic `GL(n)` 表示的 Galois 表示构造。

## 代数群、根资料和 L 群

- Armand Borel, *Linear Algebraic Groups*, Springer. 用于 connected reductive groups、Borel subgroup、maximal torus、root datum 和结构定理。
- James E. Humphreys, *Linear Algebraic Groups*, Springer. 用于 split reductive groups、根系统和 Weyl group 的基础结构。
- T. A. Springer, *Linear Algebraic Groups*, Birkhauser/Springer. 用于 reductive group 结构和 root datum 分类背景。
- J. S. Milne, *Algebraic Groups*, online notes. 用于代数群、tori、characters、cocharacters 和 reductive groups 的定义核对。
- Conrad, Gabber, and Prasad, *Pseudo-reductive Groups*, Cambridge University Press. 用于一般域上 reductive 与 pseudo-reductive 现象的边界提醒。
- A. Borel, *Automorphic L-functions*, in *Automorphic Forms, Representations and L-functions*, Proc. Symp. Pure Math. 33. 用于 L 群、非分歧参数和自守 L 函数接口。

## 局部 Langlands 和表示论

- Colin J. Bushnell and Guy Henniart, *The Local Langlands Conjecture for GL(2)*, Springer. 用于附录 AE 的 normalized principal series、Steinberg special parameter、supercuspidal 不可约 Weil 参数和标准局部因子；与本书 $|\operatorname{Fr}_F|=q^{-1}$ 合用时，非分歧 Steinberg twist 给出 $q^{-s-1/2}$。
- Michael Harris and Richard Taylor, *The Geometry and Cohomology of Some Simple Shimura Varieties*, Princeton University Press.
- Guy Henniart, works on local Langlands for `GL(n)`.
- Guy Henniart, *Une preuve simple des conjectures de Langlands pour GL(n) sur un corps p-adique*, Inventiones Mathematicae. 用于 `GL(n)` 局部 Langlands 的定理性依据。
- Peter Scholze, *The local Langlands correspondence for GL_n over p-adic fields*, Inventiones Mathematicae. 用于 `GL(n)` 局部 Langlands 的几何证明路线参考。
- Robert P. Langlands, *On the classification of irreducible representations of real algebraic groups*. 用于 Archimedean 局部 Langlands 和 Langlands 分类背景。
- David Vogan, works on the local Langlands conjecture and representations of real reductive groups. 用于 L-packet 和 component group 参数化背景。
- Jeffrey Adams, Dan Barbasch, and David Vogan, *The Langlands Classification and Irreducible Characters for Real Reductive Groups*, Birkhauser. 用于实还原群的局部对应和 character 理论。
- Tasho Kaletha, works on rigid inner forms and refined local Langlands correspondence. 用于 enhanced parameters、inner forms 和 normalization 的现代接口。
- Robert Kottwitz, works on stable trace formula, isocrystals, and classification of inner forms. 用于 Kottwitz 符号、内形式和稳定迹公式接口。
- Diana Shelstad, works on endoscopic transfer and real groups. 用于 transfer factors、stable characters 和 endoscopic character identities。
- Robert Langlands and Diana Shelstad, works on endoscopy and transfer factors. 用于 endoscopic datum 和 transfer factor 归一化。
- James Arthur, *The Endoscopic Classification of Representations: Orthogonal and Symplectic Groups*, AMS. 用于 classical groups 的 endoscopic packet 和局部-全局接口。
- Chung Pang Mok, works on endoscopic classification for unitary groups. 用于 unitary groups 的局部 packet 与全局分类接口。
- Jean-Loup Waldspurger, works on endoscopy and transfer. 用于 local endoscopic transfer 与基本引理背景。
- Labesse and Langlands, works on `SL(2)` packets. 用于 $\operatorname{SL}_2$ 局部 packet 例子。
- Jacquet and Langlands, *Automorphic Forms on GL(2)*. 也用于 quaternion algebra 内形式和 local Jacquet-Langlands 对应。
- A. Borel, *Automorphic L-functions*, in *Automorphic Forms, Representations and L-functions*, Proc. Symp. Pure Math. 33.
- Harish-Chandra, works on harmonic analysis on reductive p-adic groups. 用于 characters、Plancherel 和 tempered representations。
- Joseph Bernstein, Pierre Deligne, and David Kazhdan, works on admissible representations and Paley-Wiener theory. 用于 Bernstein center、blocks 和 local Paley-Wiener。
- J.-L. Waldspurger, works on harmonic analysis and trace formula local terms. 用于局部字符、Plancherel 和 endoscopy 局部接口。
- Francois Bruhat and Jacques Tits, works on reductive groups over local fields and buildings. 用于 buildings、parahoric 和 hyperspecial subgroups。
- Allen Moy and Gopal Prasad, works on filtrations of p-adic groups. 用于 Moy-Prasad filtrations 和 depth。
- I. N. Bernstein and A. V. Zelevinsky, works on representations of reductive p-adic groups.
- Ichiro Satake, original works on spherical functions and Hecke algebras. 用于 Satake 同构历史来源。
- I. G. Macdonald, *Spherical Functions on a Group of p-adic Type*. 用于球函数、Cartan 分解和 Satake transform。
- Satake isomorphism references through Borel and standard automorphic forms texts.

## 模形式、椭圆曲线和费马大定理

- Fred Diamond and Jerry Shurman, *A First Course in Modular Forms*, Springer.
- Toshitsune Miyake, *Modular Forms*, Springer.
- A. O. L. Atkin and Joseph Lehner, works on Hecke operators and Atkin-Lehner involutions.
- Wen-Ching Winnie Li, works on newforms and Atkin-Lehner theory.
- A. O. L. Atkin and Wen-Ching Winnie Li, works on twists of newforms and old/new theory. 用于 Atkin-Lehner-Li 分解。
- William Casselman, *On some results of Atkin and Lehner*, Mathematische Annalen 201 (1973), pp. 301-314. 用于 `GL(2)` 局部 newvector theorem、导子和固定向量维数公式。
- Jean-Pierre Serre, *A Course in Arithmetic*, Springer.
- Joseph H. Silverman, *The Arithmetic of Elliptic Curves*, Springer.
- Joseph H. Silverman, *Advanced Topics in the Arithmetic of Elliptic Curves*, Springer.
- Andre Neron, works on Neron models. 用于 Neron mapping property 和 Neron model existence 背景。
- Kodaira and Neron classification of singular fibers. 用于 Kodaira symbols、components 和 reduction types。
- Andrew Ogg, works on conductors of elliptic curves. 用于 Ogg conductor formula。
- John Tate, *Algorithm for determining the type of a singular fiber in an elliptic pencil*, in *Modular Functions of One Variable IV*.
- Gerhard Frey and Yves Hellegouarch, works associating elliptic curves to Fermat-type equations. 用于 Frey-Hellegouarch curve、半稳定性和判别式接口。
- Gary Cornell, Joseph H. Silverman, and Glenn Stevens, eds., *Modular Forms and Fermat's Last Theorem*, Springer.
- Andrew Wiles, *Modular elliptic curves and Fermat's Last Theorem*, Annals of Mathematics, 1995.
- Richard Taylor and Andrew Wiles, *Ring-theoretic properties of certain Hecke algebras*, Annals of Mathematics, 1995.
- Kenneth Ribet, works on level lowering and the epsilon conjecture.
- Kenneth Ribet, *On modular representations of $\operatorname{Gal}(\overline{\mathbb Q}/\mathbb Q)$ arising from modular forms*, Inventiones Mathematicae 100 (1990), pp. 431-476. 用于第十章所调用的权二 level-lowering/epsilon-conjecture 接口。
- Jean-Pierre Serre, *Sur les representations modulaires de degre 2 de Gal(\overline Q/Q)*, Duke Mathematical Journal, 1987.
- Chandrashekhar Khare and Jean-Pierre Wintenberger, works proving Serre's modularity conjecture.
- Christophe Breuil, Brian Conrad, Fred Diamond, and Richard Taylor, *On the modularity of elliptic curves over Q*, Journal of the AMS, 2001.
- Barry Mazur, *Deforming Galois representations*, in *Galois Groups over Q*.
- Barry Mazur, *Rational isogenies of prime degree*, Inventiones Mathematicae 44 (1978), pp. 129-162，以及 Frey-Serre-Ribet 的后续论证。用于 Frey residual representation 不可约性接口；低指数的独立处理必须与一般 $p$ 的论证分开登记。
- Jean-Pierre Serre, *Abelian l-adic representations and elliptic curves*, Benjamin/Addison-Wesley.
- Pierre Deligne, *Formes modulaires et representations $\ell$-adiques*, Seminaire Bourbaki, exp. 355, Lecture Notes in Mathematics 179, Springer, 1971, pp. 139-172；以及 *Formes modulaires et representations de $\operatorname{GL}(2)$*, in Lecture Notes in Mathematics 349, Springer, 1973, pp. 55-105. 用于归一化本征形式的 $\ell$-adic 表示及好素数 Frobenius 多项式。
- Henri Carayol, *Sur les representations $\ell$-adiques associees aux formes modulaires de Hilbert*, Annales scientifiques de l'Ecole Normale Superieure 19 (1986), pp. 409-468. 用于坏素数处 Weil-Deligne 局部-整体相容接口；正文只调用其中明确列出的经典模形式特例。
- Goro Shimura and Martin Eichler, works on Eichler-Shimura theory. 用于模曲线上同调和权二情形。
- Gerd Faltings and Jean-Marc Fontaine, works on p-adic Hodge theory. 用于 de Rham/crystalline comparison 和 Hodge-Tate weights。
- Michael Harris and Richard Taylor, *The Geometry and Cohomology of Some Simple Shimura Varieties*. 用于数域 `GL(n)` Galois 表示构造。
- Laurent Clozel, Richard Taylor, Peter Scholze, Ana Caraiani, and Harris-Lan-Taylor-Thorne works on Galois representations and local-global compatibility. 用于 cohomological automorphic Galois representations。
- Barnet-Lamb, Gee, Geraghty, and Taylor works on automorphy lifting. 用于 RAECSDC automorphy lifting 接口。

## 几何 Langlands

- Edward Frenkel, *Langlands Correspondence for Loop Groups*, Cambridge University Press.
- Edward Frenkel and Dennis Gaitsgory, works and surveys on geometric Langlands. 用于 Hecke eigensheaves 和几何 Langlands 基本形式。
- Ivan Mirkovic and Kari Vilonen, *Geometric Langlands duality and representations of algebraic groups over commutative rings*, Annals of Mathematics 166 (2007), pp. 95-143; arXiv:math/0401222v5. 用于几何 Satake；`GSAT-1` 定位为主等价 (1.1) 和 Theorem 12.1，承担 Tannaka 群的还原性与 dual root datum identification；§§5--6 定位 fusion 与 parity modification。有限域函数迹比较还需固定 Weil structure、几何 Frobenius 和半 Tate twist；权零 IC 的开胞腔 trace 按本书约定为 $(-1)^{d_\lambda}q^{-d_\lambda/2}$。
- Victor Ginzburg, works on perverse sheaves on affine Grassmannian and Langlands duality. 用于几何 Satake 背景。
- Alexander Beilinson and Vladimir Drinfeld, works on geometric Langlands and chiral algebras.
- Alexander Beilinson and Vladimir Drinfeld, works on Beilinson-Drinfeld Grassmannians and factorization. 用于 BD Grassmannian、fusion 和 Hecke factorization。
- Dennis Gaitsgory and Sam Raskin, *Proof of the geometric Langlands conjecture I: construction of the functor*, arXiv:2405.03599；D. Arinkin et al., *Proof of the geometric Langlands conjecture II: Kac-Moody localization and the FLE*, arXiv:2405.03648；Justin Campbell et al., *Proof of the geometric Langlands conjecture III: compatibility with parabolic induction*, arXiv:2409.07051；D. Arinkin et al., *Proof of the geometric Langlands conjecture IV: ambidexterity*, arXiv:2409.08670；Dennis Gaitsgory and Sam Raskin, *Proof of the geometric Langlands conjecture V: the multiplicity one theorem*, arXiv:2409.09856. 作为第二十一章和附录 O 的特征零外部 preprint theorem 来源。Paper I §0.1.1--0.1.2、§1.1 明确把自动侧定义为普通 $\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)$，并构造 automorphic $\to$ $\operatorname{IndCoh}_{\mathcal N}$ 的 $\mathbb L_G$；从谱侧返回自动侧时使用证明所得逆等价 $\mathbb L_G^{-1}$。Paper I §1.6 另行定义 $\operatorname{DMod}_{1/2}(\operatorname{Bun}_G)_{\operatorname{ren}}$、$\operatorname{ren}\dashv\operatorname{un\text{-}ren}$ 及指向全部 $\operatorname{IndCoh}$ 的 $\mathbb L_{G,\operatorname{ren}}$；这个 companion 不取代主定理的普通自动端。
- Dennis Gaitsgory and Sam Raskin, *Geometric Langlands in positive characteristic from characteristic zero*, arXiv:2508.02237. 只用于正特征 $\ell$-adic 情形中 automorphic nilpotent category 与参数栈某些连通分支之并上的 $\operatorname{IndCoh}$ 范畴之部分 preprint theorem，不作为 unrestricted 正特征等价引用。
- Masaki Kashiwara, works on D-modules and Riemann-Hilbert correspondence. 用于 D-modules、constructible sheaves 和正则 holonomic 口径。
- Ryoshi Hotta, Kiyoshi Takeuchi, and Toshiyuki Tanisaki, *D-Modules, Perverse Sheaves, and Representation Theory*. 用于 D-module 六运算、Riemann-Hilbert 和表示论接口。
- David Nadler, works on geometric Satake and sheaf-theoretic representation theory.
- Dennis Gaitsgory and Jacob Lurie, *Weil's Conjecture for Function Fields*, available through the authors' project pages.
- Dennis Gaitsgory and Nick Rozenblyum, *A Study in Derived Algebraic Geometry*, AMS.
- Dima Arinkin and Dennis Gaitsgory, works on singular support and geometric Langlands. 用于 $\operatorname{IndCoh}_{\mathcal N}$ 谱侧。
- Peter Scholze, works on perfectoid spaces and diamonds. 用于 diamonds、pro-etale geometry 和 local Shimura varieties。
- Laurent Fargues and Jean-Marc Fontaine, *Courbes et fibrés vectoriels en théorie de Hodge $p$-adique*, Astérisque 406 (2018), especially Théorème 8.2.10. 用于完备代数闭 perfectoid field 上 FF 曲线的 vector bundle 分类；一般基底只经相对曲线和 descent 使用。
- Laurent Fargues, *G-torseurs en théorie de Hodge $p$-adique*, Compositio Mathematica 156 (2020), no. 10, pp. 2076--2110. 用于完备代数闭 perfectoid field 上 $G$-bundles 与 $B(G)$ 的分类。
- Peter Scholze and Jared Weinstein, *Berkeley Lectures on $p$-adic Geometry*, Annals of Mathematics Studies 207, Princeton University Press, 2020, especially Lectures 23--24. 用于 local shtuka diamonds、local Shimura varieties 及 Weil descent。Lecture 24 采用的 $B(G,\mu^{-1})$/relative-position convention 与本书比较时必须同时反转 $\mu$ 和 modification 方向。
- Michael Rapoport and Eva Viehmann, *Towards a theory of local Shimura varieties*, Münster Journal of Mathematics 7 (2014), pp. 273--326, especially Definition 5.1. 用于 local Shimura datum 的 minuscule 条件、acceptable Kottwitz 类和 reflex field。
- Laurent Fargues and Peter Scholze, *Geometrization of the local Langlands correspondence*, arXiv:2102.13459v4 (accepted version), especially Theorem II.2.14、Theorem III.2.2、I.9、IX.3、IX.4、IX.6 and X. II.2.14 与 III.2.2 分别用于完备代数闭非 Archimedean 几何点上的 vector bundles 和 $G$-bundles 同构类分类；相对 $\operatorname{Bun}_G$ 只按 v-stack/descent 口径使用。IX.3 用于核对 $b\in B(G,\mu)$、reflex field $E_\mu$ 与 $W_{E_\mu}$ 作用；后三处用于 spectral action 和 semisimple LLC map。该来源不给出正文未声明的 monodromy operator 或完整 enhanced packet 分类。
- Vladimir Drinfeld, works on shtukas and `GL(2)` function field Langlands.
- Laurent Lafforgue, works on shtukas and `GL(n)` function field Langlands.
- Vincent Lafforgue, works on excursion operators and the global Langlands parameterization over function fields. 用于一般还原群的函数域参数化接口。
- Bao Chau Ngo, *Le lemme fondamental pour les algèbres de Lie*, Publications Mathématiques de l'IHES. 用于 fundamental lemma 和 Hitchin fibration.

## 使用规则

- 对本书内“外部输入定理”的每次使用，必须能在本文件中定位到相应资料源。
- 若某章使用尚未列出的资料源，先更新本文件，再写正文。
- 若不同资料源采用不同归一化，例如 Frobenius、局部 Artin 映射或 Haar 测度，正文必须显式选择一种归一化。
