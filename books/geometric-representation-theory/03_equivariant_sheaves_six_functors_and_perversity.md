# 第三章：Equivariant sheaves、六函子与 perverse t-structure

Schubert variety 往往有奇点，普通局部系统只能描述光滑 stratum，普通上同调又不能在 proper pushforward 后同时保留各层的维数信息。需要一种对象既能沿开、闭嵌入做六函子运算，又把“复维数 $d$ 的光滑层应放在次数 $-d$”编码进心范畴；perverse sheaf 正是这种重新校准后的层。为避免把拓扑层、代数表示和不同系数理论混在一起，以下固定复解析 Betti 模型，并把 equivariance 解释为 quotient stack 上的 coherent descent。这个语言随后会同时承载 Schubert IC 层、Springer sheaf 和 Satake category。

## 3.1 Constructible derived categories

**约定 3.1（本章的 Betti 模型）.** 本章固定 $k=\mathbb C$，并固定特征为 $0$ 的系数域 $E$。除非另行声明，$X,Y$ 是分离、有限型复代数 scheme，层实际定义在解析空间 $X^{\mathrm{an}}$ 的 classical topology 上；“constructible”总指对某个有限代数分层 constructible。所有 local systems 都要求 stalk 为有限维 $E$-向量空间。

本章的结论因此不自动包含 $\ell$-adic 版本。若后文改用 $\ell$-adic sheaves，必须重新声明底域、素数 $\ell\ne\operatorname{char}k$、系数域和 Tate twist，并引用该模型的六函子与 perverse formalism；两种模型之间没有未声明的 comparison functor。

**定义 3.2.** 给定有限代数分层
$$
X=\coprod_{\alpha\in A}X_\alpha
$$
，其中每个 $X_\alpha$ 是光滑、连通、locally closed subvariety，令
$$
D^b_{\mathscr S}(X,E)\subset D^b(E_{X^{\mathrm{an}}})
$$
为这样的 full triangulated category：$\mathcal F$ 的每个 cohomology sheaf 在每个 $X_\alpha^{\mathrm{an}}$ 上局部常值、stalk 有限维，且只有有限多个 cohomological degrees 非零。$D^b_c(X,E)$ 表示允许有限分层加细后所得的 constructible category。符号 $H^i(\mathcal F)$ 表示 standard t-structure 的 cohomology sheaf，$\mathbb H^i(X,\mathcal F)$ 表示 hypercohomology；二者不得混写。

**定义 3.3.** 令复代数群 $H$ 左作用于 $X$ 并保持分层。本书取 Bernstein--Lunts finite-dimensional approximation 与 quotient-stack construction 相容的 Betti 模型，并记
$$
D^b_H(X,E)=D^b_c([X/H],E)
$$
为 $H$-equivariant constructible derived category。沿 atlas $u:X\to[X/H]$ 的拉回给出 forgetful functor
$$
\operatorname{For}_H:D^b_H(X,E)\longrightarrow D^b_c(X,E).
$$
在普通 1-categorical 记号中，一个等变对象可缩写为 $\mathcal F\in D^b_c(X,E)$ 连同作用图
$$
a:H\times X\to X,\qquad p:H\times X\to X
$$
上的 descent isomorphism
$$
a^\ast\mathcal F\simeq p^\ast\mathcal F
$$
及 cocycle condition；严格地说，导出对象需要沿 nerve $H^\bullet\times X$ 的 coherent descent datum，单独写出一个同构并不足以定义对象。后文所有 pullback、pushforward 和卷积都在这个已固定模型中进行。

**命题 3.4.** 若 $X=H/K$，则在附录 A 的假设下，$D^b_H(X,E)\simeq D^b(BK,E)$。

**证明.** 命题 A.8 给出 quotient stack 的等价
$$
[(H/K)/H]\simeq BK.
$$
constructible derived category 对 stack equivalence 不变，故拉回给出所述三角范畴等价，其准逆由反向 stack equivalence 的拉回给出；两个复合由 stack 等价的 2-isomorphisms 自然同构于恒等函子。$\square$

**类型警告 3.4.1.** 命题 3.4 的右侧是 sheaves on $BK$，不是 algebraic representations of $K$。例如 Betti 模型中若 $K$ 连通，则 $\pi_1(BK)\simeq\pi_0(K)=1$，所以 $BK$ 上的不可约 local system 只有常值秩一对象，而连通 reductive $K$ 通常有许多非平凡代数表示。有限稳定子情形才直接恢复有限群的表示；详见推论 A.9。

Equivariant category 的类型确定以后，几何构造归结为沿映射拉回、推前和取对偶。接下来的六函子定理是这些操作的共同基础；正文只证明由它形式推出的命题，不把整个 formalism 伪装成局部计算。

## 3.2 六函子和 Verdier duality

**约定 3.5.** 对分离、有限型 morphism $f:X\to Y$，本书把
$$
f^\ast,\ f_\ast,\ f_!,\ f^!
$$
均视为 Betti constructible categories 上的 derived functors；若需要强调，可写 $Rf_\ast$ 与 $Rf_!$。存在自然比较 morphism $f_!\to f_\ast$，当 $f$ proper 时它是自然同构。未导出 functor 不使用相同符号。

**外部输入定理 3.6（Betti 六函子）.** 在约定 3.1 和 3.5 的范围内，六函子保持 constructibility，并满足 adjunctions
$$
f^\ast\dashv f_\ast,\qquad f_!\dashv f^!,
$$
base change、projection formula、Kunneth compatibility 和 Verdier duality compatibility。若使用 $\ast$-base-change 的 proper 版本，则相应竖直 morphism 必须 proper；$!$-base-change 按六函子 formalism 的版本调用。这里采用 BBD 与 Kashiwara--Schapira 的 Betti constructible 版本，不重建六函子 formalism。

**定义 3.7.** Verdier duality 定义为
$$
\mathbb D_X(\mathcal F)=R\mathcal Hom(\mathcal F,\omega_X^\bullet).
$$
它是 contravariant functor
$$
\mathbb D_X:D^b_c(X,E)^{op}\to D^b_c(X,E).
$$

**命题 3.8.** 若 $f:X\to Y$ proper，则存在自然同构
$$
\mathbb D_Y f_\ast\simeq f_\ast \mathbb D_X.
$$

**证明.** 由 properness，$f_\ast\simeq f_!$。Verdier duality 的六函子相容性给出
$$
\mathbb D_Y f_!\simeq f_\ast\mathbb D_X.
$$
代入 $f_!=f_\ast$ 得到结论。该证明的实质输入是外部输入定理 3.6。$\square$

## 3.3 Perverse t-structure

**定义 3.9.** 对约定 3.1 中的复代数簇 $X$，middle perversity 的 aisle 和 co-aisle 由条件
$$
{}^pD^{\le0}(X)=
\{\mathcal F\mid \dim\{x\in X\mid H^i(i_x^\ast\mathcal F)\ne0\}\le -i,\ \forall i\}
$$
和
$$
{}^pD^{\ge0}(X)=
\{\mathcal F\mid \dim\{x\in X\mid H^i(i_x^!\mathcal F)\ne0\}\le i,\ \forall i\}
$$
刻画。其心记为
$$
\operatorname{Perv}(X,E)={}^pD^{\le0}(X)\cap{}^pD^{\ge0}(X).
$$

更常用的等价 stratum-wise 表述为：若 $j_\alpha:X_\alpha\hookrightarrow X$ 是维数 $d_\alpha$ 的 stratum，则要求
$$
H^i(j_\alpha^\ast\mathcal F)=0\quad(i>-d_\alpha),
$$
以及
$$
H^i(j_\alpha^!\mathcal F)=0\quad(i< -d_\alpha).
$$

**约定 3.9.1（等变 perversity）.** 对有限维复代数群 $H$ 的作用，本书定义
$$
\operatorname{Perv}_H(X,E)
=\{\mathcal F\in D^b_H(X,E)\mid
\operatorname{For}_H(\mathcal F)\in\operatorname{Perv}(X,E)\}.
$$
也就是说，forgetful functor 按定义 t-exact，不额外加入 $\dim H$ shift。这与直接按 stack dimension 给 $[X/H]$ 归一化不是同一句话：atlas $X\to[X/H]$ 光滑、相对维数 $\dim H$，而 intrinsic stack convention 通常使 $u^\ast[\dim H]$ t-exact。第十二章的 pro-algebraic $L^+G$-equivariance 将在有限维支撑和有限商上采用本约定。

**外部输入定理 3.10.** 上述条件定义 $D^b_c(X,E)$ 上的 t-structure，其心 $\operatorname{Perv}(X,E)$ 是 artinian and noetherian abelian category，在有限分层和有限系数条件下有有限长度。  
来源：BBD。

**命题 3.11.** 若 $X$ 光滑连通纯维数 $d$，则局部系统 $\mathcal L$ 的 shift $\mathcal L[d]$ 是 perverse sheaf。

**证明.** 对唯一光滑 stratum $X$，$j=\operatorname{id}_X$。有
$$
H^i(j^\ast\mathcal L[d])=H^{i+d}(\mathcal L),
$$
仅当 $i=-d$ 时可能非零，因此满足 $i>-d$ 的 vanishing。又因为 $X$ 光滑，$j^!=j^\ast$ 对 identity map 成立，所以 cosupport 条件同样满足。$\square$

**例 3.12.** 若 $X=\mathbb A^1$，则常值 sheaf $E_X[1]$ 是 perverse sheaf，而 $E_X$ 不是 perverse sheaf。原因是唯一 open stratum 维数为 $1$，perverse normalization 要求局部系统放在 cohomological degree $-1$。

这个一维计算解释了 shift，却没有处理奇点。对一个光滑 stratum 的局部系统，middle extension 在边界处同时排除支撑其上的 subobject 与 quotient，从而给出最小而对偶对称的延拓；intersection complex 由此出现。

**命题 3.13.** 令 $i:Z\hookrightarrow X$ 为复代数簇的闭嵌入。对任意 $\mathcal P\in\operatorname{Perv}(Z,E)$，有
$$
i_!\mathcal P\simeq i_\ast\mathcal P\in\operatorname{Perv}(X,E),
$$
且 $i_\ast$ 在 perverse hearts 上 fully faithful。特别地，若 $Z$ 光滑纯维数 $d$，则 $i_\ast E_Z[d]$ 是支撑在 $Z$ 上的 perverse sheaf。

**证明.** 闭嵌入是 proper，故约定 3.5 给出 $i_!\simeq i_\ast$。BBD perverse formalism 中闭嵌入的 $i_\ast$ 是 t-exact 且 derived-level fully faithful，这是外部输入 `BBD-1` 的一部分，因此其在 hearts 上也 fully faithful。最后，对光滑 $Z$，命题 3.11 给出 $E_Z[d]\in\operatorname{Perv}(Z,E)$，代入一般结论即可。$\square$

## 3.4 Intersection complexes

**定义 3.14.** 令 $S\subset X$ 为光滑 locally closed stratum，$j:S\hookrightarrow\overline S$，$\mathcal L$ 为 $S$ 上的 local system，$d=\dim S$。intersection complex 定义为 middle extension
$$
\operatorname{IC}(\overline S,\mathcal L)=j_{!*}(\mathcal L[d]).
$$
若 $\mathcal L=E_S$，简记为 $\operatorname{IC}_{\overline S}$。

**外部输入定理 3.15.** middle extension $j_{!*}$ 存在且保持 perversity；$\operatorname{IC}(\overline S,\mathcal L)$ 是 simple perverse sheaf，前提是 $\mathcal L$ 是 irreducible local system。  
来源：BBD。

**定义 3.16.** 对 Schubert variety $\overline X_w\subset G/B$，本书采用 normalization
$$
\operatorname{IC}_w=\operatorname{IC}(\overline X_w,E_{X_w}).
$$
由于 $X_w\simeq\mathbb A^{\ell(w)}$，其常值 sheaf shift 为 $E_{X_w}[\ell(w)]$。

**命题 3.17.** 若 $\overline S$ 光滑、连通、纯维数为 $d$，且 $S\subset\overline S$ 是非空 dense open stratum，则
$$
\operatorname{IC}(\overline S,E_S)\simeq E_{\overline S}[d].
$$

**证明.** 命题 3.11 给出 $E_{\overline S}[d]\in\operatorname{Perv}(\overline S,E)$。把定理 3.15 用于 identity stratum $\overline S\hookrightarrow\overline S$ 和不可约秩一 local system $E_{\overline S}$，可知该 perverse sheaf 是 simple。它限制到 $S$ 正是 $E_S[d]$。

若 $\mathcal Q\subset E_{\overline S}[d]$ 是支撑在边界上的 perverse subobject，则 simplicity 强迫 $\mathcal Q=0$ 或 $\mathcal Q=E_{\overline S}[d]$；后一种情形限制到 $S$ 后会得到 $0=E_S[d]$，矛盾，故 $\mathcal Q=0$。对支撑在边界上的 quotient 使用同一 simplicity 论证，也只能得到零对象。因此 $E_{\overline S}[d]$ 满足 middle extension 的无边界 subobject/quotient 刻画。由定理 3.15 的唯一性，
$$
j_{!*}E_S[d]\simeq E_{\overline S}[d].
$$
这就是所需同构。$\square$

## 3.5 Decomposition theorem 的边界

IC 层解决了单个奇异闭包上的延拓问题。若再沿 proper map 推前，结果通常不再是单个 IC 层；decomposition theorem 说明它仍可由 semisimple perverse pieces 组成，但其适用对象、系数与 hard Lefschetz 附加条件必须分别陈述。

**外部输入定理 3.18（本书使用的 Betti decomposition package）.** 令 $f:X\to Y$ 为复代数簇之间的 proper algebraic morphism，$E$ 为特征 $0$ 域，$X$ 为不可约复代数簇，$\operatorname{IC}_X$ 取定义 3.14 的 perverse normalization。则在 $D^b_c(Y,E)$ 中存在同构
$$
Rf_\ast\operatorname{IC}_X
\simeq
\bigoplus_{i\in\mathbb Z}
{}^pH^i(Rf_\ast\operatorname{IC}_X)[-i],
$$
每个 ${}^pH^i$ 是 semisimple perverse sheaf，因而对某个适配分层可写成有限直和
$$
{}^pH^i(Rf_\ast\operatorname{IC}_X)
\simeq
\bigoplus_{(S,\mathcal L)}
\operatorname{IC}(\overline S,\mathcal L)^{\oplus m_{i,S,\mathcal L}},
$$
其中 $\mathcal L$ 是有限秩不可约 local system。该 derived splitting 一般不 canonical；perverse cohomology objects 及其 semisimple isomorphism classes 才是 canonical 数据。

若进一步 $f$ projective，$\eta=c_1(\mathcal A)$ 来自 $f$-ample line bundle，则相对 hard Lefschetz 是另一项结论：对每个 $i\ge0$，
$$
\eta^i:{}^pH^{-i}(Rf_\ast\operatorname{IC}_X)
\xrightarrow{\ \sim\ }
{}^pH^{i}(Rf_\ast\operatorname{IC}_X).
$$
若 $X$ 光滑纯维数 $d$，才可把输入改写为 $\operatorname{IC}_X\simeq E_X[d]$。这一版本见 BBD 6.2.5，semismall 特化采用相应的 semismall decomposition theorem；本书不重证这些结论。

**警告 3.19（不能越过的边界）.** 定理 3.18 不允许作下列替换：

1. 不得把 $\operatorname{IC}_X$ 无条件替换成任意 semisimple perverse sheaf；常用推广要求输入属于 geometric origin、pure Hodge module 或相应有 weight 的类别。
2. 特征 $0$ 不能删去；modular coefficients 下 decomposition 和 semisimplicity 都可能失败。
3. Properness 足够陈述上述 decomposition，但相对 hard Lefschetz 的这一版本还需要 projectivity 和一个 $f$-ample class。
4. Betti semisimplicity 本身不提供 Frobenius weights 或 trace positivity；需要这些结论时必须切换到 mixed/Hodge 或 $\ell$-adic 模型并重新声明假设。

**检查表 3.20.** 使用 decomposition theorem 前必须说明：

1. $f$ 的定义域、值域和 algebraicity，以及它是 proper 还是 projective；
2. 输入是 $\operatorname{IC}_X$、光滑源上的 $E_X[\dim X]$，还是另一个已证明属于允许类别的对象；
3. 使用 Betti、$\ell$-adic 还是 mixed Hodge module 版本及其系数条件；
4. 所需输出是 derived splitting、perverse semisimplicity，还是 relative hard Lefschetz；
5. splitting 是否被错误地当作 canonical；
6. 是否另行使用 purity、weights 或 Frobenius trace 推出正性。

现在可以把光滑 stratum 上的局部系统延拓为 IC 层，并在 proper pushforward 后按 perverse cohomology 分解；同时，quotient stack 模型保证群作用没有被压成一个含混的“等变条件”。第四章把这些操作放到 $B\backslash G/B$ 的卷积图中，Schubert IC 层因而从奇点不变量变成 Hecke 代数的范畴化基向量。

## 练习

**练习 3.1.** 令 $X=\mathbb A^1$，分层为 $\{0\}\sqcup\mathbb G_m$。写出 perverse sheaf support/cosupport 条件在两个 stratum 上的形式。

**练习 3.2.** 对 $SL_2/B\simeq\mathbb P^1$ 的 Schubert 分层，写出 $\operatorname{IC}_e$ 和 $\operatorname{IC}_s$。

**练习 3.3.** 证明若 $f:X\to Y$ 是同构，则 $f^\ast,f_\ast,f_!,f^!$ 在 constructible derived categories 上均为等价，并说明这些等价与 Verdier duality 相容。
