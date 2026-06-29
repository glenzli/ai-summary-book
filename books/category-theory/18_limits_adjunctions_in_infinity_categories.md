# 第十八章：$\infty$-范畴中的等价、极限和伴随

## 本章目标

本章在 quasi-category 口径下说明 $\infty$-范畴中的等价边、同伦范畴、极限和伴随。重点是把普通范畴论中的定义替换为同伦不变版本。

## 依赖前置知识

需要第十七章的单纯集、inner horn 和 quasi-category。

## 18.1 同伦范畴

**定义 18.1.** 设 $C$ 为 quasi-category。其同伦范畴 $hC$ 的对象是 $C_0$ 的元素。态射 $x\to y$ 是从 $x$ 到 $y$ 的 1-单纯形按 2-单纯形生成的同伦关系取商。

**外部输入命题 18.2.** 上述关系给出范畴 $hC$，复合由 $\Lambda_1^2$ 的填充定义，并且不同填充给出同一同伦类。

证明需要 quasi-category 的 horn calculus，来源见 Lurie HTT 与 Riehl-Verity。

**定义 18.3.** 1-单纯形 $f:x\to y$ 称为等价，若它在 $hC$ 中成为同构。

## 18.2 映射空间

**定义 18.4.** 设 $C$ 为 quasi-category，$x,y\in C_0$。右映射空间 $\operatorname{Map}^R_C(x,y)$ 是如下单纯集：其 $n$-单纯形是映射

$$
\sigma:\Delta^{n+1}\to C
$$

满足 $\sigma(0)=x$，并且 $\sigma$ 在由顶点 $1,\dots,n+1$ 张成的面上等于 $y$ 的常值 $n$-单纯形。换言之，后 $n+1$ 个顶点及其间所有高维面都退化到 $y$。

左映射空间 $\operatorname{Map}^L_C(x,y)$ 对偶定义：其 $n$-单纯形是映射 $\sigma:\Delta^{n+1}\to C$，满足 $\sigma(n+1)=y$，并且 $\sigma$ 在由顶点 $0,\dots,n$ 张成的面上等于 $x$ 的常值 $n$-单纯形。

不同标准模型之间存在自然弱等价。本书在不需要区分模型时记作

$$
\operatorname{Map}_C(x,y).
$$

**外部输入定理 18.5.** 若 $C$ 是 quasi-category，则 $\operatorname{Map}_C(x,y)$ 是 Kan 复形，且

$$
\pi_0\operatorname{Map}_C(x,y)\cong hC(x,y).
$$

本书后续使用该定理表达“态射空间”而非仅 Hom 集。

**命题 18.6.** 若 $C=N(\mathcal A)$ 是普通范畴的 nerve，则 $\operatorname{Map}^R_C(x,y)$ 是离散单纯集 $\mathcal A(x,y)$。

**证明.** 一个 $n$-单纯形 $\Delta^{n+1}\to N(\mathcal A)$ 等价于一个函子 $[n+1]\to\mathcal A$。右映射空间的条件要求顶点 $0$ 送到 $x$，并要求由 $1,\dots,n+1$ 张成的子范畴整体送到对象 $y$ 及其恒等态射。因此该函子唯一由边 $0\to1$ 的像

$$
f:x\to y
$$

决定；函子性强制所有边 $0\to i$ 的像也是 $f$，所有 $i\to j$ 且 $1\le i\le j$ 的像为 $\operatorname{id}_y$。反过来，任意态射 $f:x\to y$ 都按此规则给出唯一的 $n$-单纯形。面映射和退化映射不改变 $f$，所以所得单纯集正是集合 $\mathcal A(x,y)$ 对应的离散单纯集。$\square$

## 18.3 join、slice 与锥

**定义 18.7.** 单纯集 $X,Y$ 的 join 记为 $X\star Y$。它把 $X$ 的顶点置于 $Y$ 的顶点之前，并满足

$$
\Delta^m\star\Delta^n\cong\Delta^{m+n+1}.
$$

特别地，$\Delta^0\star K$ 是 $K$ 上的右锥。

**定义 18.8.** 设 $C$ 是 quasi-category，$p:K\to C$ 是图形。slice quasi-category $C_{/p}$ 由泛性质刻画：

$$
\mathbf{sSet}(T,C_{/p})
\cong
\{\,T\star K\to C\mid (T\star K)|_K=p\,\}.
$$

对偶地，$C_{p/}$ 用 $K\star T$ 定义。

**命题 18.9.** 若 $C=N(\mathcal C)$ 且 $p:K\to N(\mathcal C)$ 来自普通图形，则 $C_{/p}$ 的 $0$-单纯形是普通锥，$1$-单纯形是锥之间的态射。

**证明.** $0$-单纯形是映射 $\Delta^0\to C_{/p}$，等价于扩张 $\Delta^0\star K\to N(\mathcal C)$。这给出一个新顶点和从该顶点到图形各顶点的相容边，即普通锥。$1$-单纯形对应 $\Delta^1\star K\to N(\mathcal C)$，即两个锥顶和二者之间与锥边相容的态射。由于 nerve 的单纯形就是普通范畴中的可复合态射串，该描述与普通锥范畴一致。$\square$

## 18.4 极限与余极限

**定义 18.10.** 设 $K$ 为单纯集，$p:K\to C$ 为图形。$p$ 的极限是终对象

$$
\lim p\in C_{/p}
$$

在适当定义的 cone $\infty$-范畴中。余极限对偶定义为 cocone $\infty$-范畴中的始对象。

**命题 18.11.** 若 $C=N(\mathcal C)$ 是普通范畴的 nerve，则 $C$ 中的极限与 $\mathcal C$ 中的普通极限一致。

**证明.** 对普通范畴 $\mathcal C$，附录 B 证明 $N(\mathcal C)$ 的内角填充唯一。因此由 $p:K\to N(\mathcal C)$ 构造的 cone quasi-category 的 $0$-单纯形正是普通锥，其 $1$-单纯形正是锥之间的态射；唯一填充保证其同伦范畴就是普通锥范畴的 nerve 所对应的同伦范畴。于是 $C_{/p}$ 中的终对象等价于普通锥范畴中的终对象。按第三章定义，这正是 $\mathcal C$ 中的普通极限。余极限由对偶论证。$\square$

**命题 18.12.** 对象 $t\in C$ 是终对象，当且仅当对所有 $x\in C$，映射空间 $\operatorname{Map}_C(x,t)$ 可缩。

**证明.** 取 $K=\varnothing$。此时 $C_{/K}\simeq C$，极限就是终对象。终对象的定义要求从任意 $x$ 到 $t$ 的锥空间为可缩空间；这正是 $\operatorname{Map}_C(x,t)$ 可缩。$\square$

## 18.5 普通极限的显式恢复

**例子 18.A.** 令 $\mathcal C$ 是有 pullback 的普通范畴，并给定 span

$$
A\xrightarrow{f}C\xleftarrow{g}B.
$$

把它视为图形 $p:K\to N(\mathcal C)$。一个锥对象就是对象 $X$ 及态射

$$
u:X\to A,\qquad v:X\to B
$$

满足

$$
fu=gv.
$$

锥之间的态射 $X\to X'$ 是同时与 $u,v$ 相容的态射。因此 $N(\mathcal C)_{/p}$ 的同伦范畴就是 ordinary pullback 锥范畴；终对象是普通 pullback $A\times_CB$。

**命题 18.B.** 若 $t$ 是 ordinary category $\mathcal C$ 的终对象，则 $t$ 也是 $N(\mathcal C)$ 的终对象。反过来，若 $t$ 是 $N(\mathcal C)$ 的终对象，则 $t$ 是 $\mathcal C$ 的终对象。

**证明.** 在 $N(\mathcal C)$ 中，终对象判别为对任意 $x$，$\operatorname{Map}_{N(\mathcal C)}(x,t)$ 可缩。普通 nerve 的映射空间与离散集合 $\mathcal C(x,t)$ 等价；离散空间可缩当且仅当它是单点。因此该条件等价于对每个 $x$，集合 $\mathcal C(x,t)$ 是单点，这正是 ordinary terminal object 的定义。$\square$

## 18.6 伴随

**定义 18.13.** $\infty$-范畴之间的函子 $F:C\to D$ 与 $G:D\to C$ 构成伴随 $F\dashv G$，若存在单位和余单位以及相干三角同伦。等价地，可由 Lurie 的 adjunction data 或 Cartesian fibration 语言定义。

**定义 18.C.** 从 $C$ 到 $D$ 的 correspondence 在本书中用函子

$$
H:C^{\operatorname{op}}\times D\to\mathcal S
$$

表示。它可看作把一对对象 $(x,y)$ 送到“从 $x$ 到 $y$ 的广义态射空间”。称 $H$ 左可表示，若存在函子 $F:C\to D$ 和自然等价

$$
H(x,y)\simeq\operatorname{Map}_D(Fx,y).
$$

称 $H$ 右可表示，若存在函子 $G:D\to C$ 和自然等价

$$
H(x,y)\simeq\operatorname{Map}_C(x,Gy).
$$

**命题 18.D.** 若 correspondence $H:C^{op}\times D\to\mathcal S$ 同时由 $F:C\to D$ 左表示、由 $G:D\to C$ 右表示，则 $F\dashv G$，其映射空间刻画为

$$
\operatorname{Map}_D(Fx,y)\simeq\operatorname{Map}_C(x,Gy).
$$

**证明.** 左可表示性给出对 $x,y$ 自然的等价

$$
H(x,y)\simeq\operatorname{Map}_D(Fx,y).
$$

右可表示性给出对 $x,y$ 自然的等价

$$
H(x,y)\simeq\operatorname{Map}_C(x,Gy).
$$

将第一个等价取逆再与第二个复合，得到

$$
\operatorname{Map}_D(Fx,y)\simeq H(x,y)\simeq\operatorname{Map}_C(x,Gy).
$$

这些等价在 $x$ 和 $y$ 中自然，因此按外部输入定理 18.14 的刻画给出伴随 $F\dashv G$。$\square$

**例子 18.E.** 若 $C=N(\mathcal C)$、$D=N(\mathcal D)$ 来自普通范畴，则 correspondence $H$ 为集合值双函子

$$
\mathcal C^{op}\times\mathcal D\to\mathbf{Set}
$$

的离散化。左可表示和右可表示分别说

$$
H(x,y)\cong\mathcal D(Fx,y),\qquad
H(x,y)\cong\mathcal C(x,Gy).
$$

于是命题 18.D 正好恢复第四章的 Hom 自然同构定义。

**定义 18.F（adjunction data 的低维模型）.** 设 $p:M\to\Delta^1$ 是同时为 Cartesian fibration 和 coCartesian fibration 的内纤维。记纤维

$$
M_0=C,\qquad M_1=D.
$$

沿唯一非退化边 $0\to1$ 的 coCartesian 传输给出函子

$$
F:C\to D,
$$

沿同一边的 Cartesian 传输给出函子

$$
G:D\to C.
$$

若由 $p$ 定义的 correspondence 同时由 $F$ 左表示、由 $G$ 右表示，则称这个双纤维对象给出一组 adjunction data。

**构造 18.G.** 从上述 adjunction data 可抽取单位和余单位。对 $x\in C$，取覆盖 $0\to1$ 的 coCartesian 边

$$
x\to F x.
$$

再取以 $F x$ 为终点的 Cartesian 边

$$
G F x\to F x.
$$

Cartesian 泛性质使第一条边唯一分解出一条纤维 $C$ 中的边

$$
\eta_x:x\to G F x.
$$

这给出单位 $\eta:\operatorname{id}_C\to G F$。对 $y\in D$，对偶地比较以 $G y$ 为起点的 coCartesian 边

$$
G y\to F G y
$$

与覆盖 $0\to1$、终点为 $y$ 的 Cartesian 边

$$
G y\to y
$$

得到余单位

$$
\varepsilon_y:F G y\to y.
$$

高维单纯形中的相干性给出三角恒等式的同伦版本。

**定义 18.I（walking adjunction）.** 普通 2-范畴中的 walking adjunction 是由两个对象 $+$、$-$，两个 1-态射

$$
f:+\to-,\qquad g:-\to+
$$

以及 2-态射

$$
\eta:\operatorname{id}_{+}\Rightarrow g f,\qquad
\varepsilon:f g\Rightarrow\operatorname{id}_{-}
$$

生成，并满足三角恒等式

$$
(\varepsilon f)\circ(f\eta)=\operatorname{id}_f,\qquad
(g\varepsilon)\circ(\eta g)=\operatorname{id}_g.
$$

其 Duskin nerve 或相应 scaled nerve 给出 $\infty$-范畴中 adjunction data 的一个模型。直观上，一个 $\infty$-伴随就是把 walking adjunction 送入 $\mathcal{Cat}_\infty$ 的相干图形。

**定义 18.K（scaled nerve 口径）.** scaled simplicial set 是单纯集 $X$ 连同一族被标记为 thin 的 $2$-单纯形，且所有退化 $2$-单纯形均 thin。严格 $2$-范畴 $\mathcal B$ 的 scaled nerve 记为

$$
N^{sc}(\mathcal B).
$$

其 $0$-单纯形为对象，$1$-单纯形为 $1$-态射，$2$-单纯形记录 $2$-态射；thin $2$-单纯形记录可逆或指定为相干等式的 $2$-态射。严格完整定义需要 Duskin nerve 与 scaled model structure，本书在正文只使用这个低维解释。

**构造 18.L.** 令 $\operatorname{Adj}$ 为定义 18.I 的 walking adjunction $2$-范畴。一个 scaled nerve 映射

$$
N^{sc}(\operatorname{Adj})\to \mathcal{Cat}_\infty
$$

给出两个 $\infty$-范畴 $C,D$，两个函子 $F:C\to D$、$G:D\to C$，单位与余单位

$$
\eta:\operatorname{id}_C\to GF,\qquad
\varepsilon:FG\to\operatorname{id}_D,
$$

以及三角恒等式的指定高阶相干同伦。换言之，scaled nerve 把“伴随”看成一个由生成元和关系描述的高阶相干图形。

**命题 18.M.** 普通范畴之间的伴随 $F:\mathcal C\rightleftarrows\mathcal D:G$ 诱导构造 18.L 中的 scaled nerve 映射。

**证明.** 普通伴随给出单位 $\eta$、余单位 $\varepsilon$，并满足严格三角恒等式。于是存在严格 $2$-函子

$$
\operatorname{Adj}\to\mathbf{Cat}
$$

把 $+$ 送到 $\mathcal C$，把 $-$ 送到 $\mathcal D$，把生成 $1$-态射 $f,g$ 送到 $F,G$，把生成 $2$-态射送到 $\eta,\varepsilon$。三角恒等式保证该赋值尊重 $\operatorname{Adj}$ 的关系。再把普通范畴嵌入 $\infty$-范畴，得到所需的 scaled nerve 映射。$\square$

**命题 18.J.** 若 $F\dashv G$ 是由 adjunction data 给出的伴随，则在同伦范畴 $hC$ 和 $hD$ 中，单位 $\eta$ 与余单位 $\varepsilon$ 满足普通三角恒等式。

**证明.** adjunction data 包含填充高维单纯形的数据，这些单纯形正是两条复合

$$
F\xrightarrow{F\eta}FGF\xrightarrow{\varepsilon F}F,
\qquad
G\xrightarrow{\eta G}GFG\xrightarrow{G\varepsilon}G
$$

与恒等变换之间的相干同伦。传到同伦范畴后，相干同伦成为自然变换的相等，因此得到普通三角恒等式。$\square$

**外部输入定理 18.H.** HTT 的 adjunction data、correspondence 的左右可表示性、以及映射空间自然等价三种伴随定义彼此等价。该等价需要 Cartesian/coCartesian fibration 的 horn calculus 和高阶相干三角恒等式。

**外部输入定理 18.14.** 在 quasi-category 理论中，伴随可由同伦范畴上的伴随加上映射空间等价刻画：对 $x\in C,y\in D$，有自然等价

$$
\operatorname{Map}_D(Fx,y)\simeq\operatorname{Map}_C(x,Gy).
$$

该等价需满足高阶自然性。完整证明见 HTT 和 Riehl-Verity。

**定理 18.15.** $\infty$-范畴中的左伴随保持所有存在的余极限；右伴随保持所有存在的极限。

**证明.** 使用外部输入定理 18.14 的映射空间刻画，以及 $\infty$-范畴中余极限的映射空间判别。若 $L=\operatorname{colim}p$，则对任意 $y\in D$，

$$
\operatorname{Map}_D(F L,y)
\simeq
\operatorname{Map}_C(L,Gy)
\simeq
\lim_{k\in K^{\operatorname{op}}}\operatorname{Map}_C(p(k),Gy)
\simeq
\lim_{k\in K^{\operatorname{op}}}\operatorname{Map}_D(Fp(k),y).
$$

这正是 $F L$ 表示 $F p$ 的余极限的映射空间条件。$\square$

## 18.7 Ordinary nerve 的低维判别

**命题 18.16.** 设 $\mathcal C$ 为普通范畴。$N(\mathcal C)$ 中的边 $f:x\to y$ 是 $\infty$-范畴意义下的等价，当且仅当 $f$ 是 $\mathcal C$ 中的同构。

**证明.** 按定义，$f$ 是等价当且仅当它在 $hN(\mathcal C)$ 中成为同构。由练习 18.2，$hN(\mathcal C)\cong\mathcal C$，且该同构把边 $f$ 的同伦类送回 $\mathcal C$ 中的态射 $f$。因此 $f$ 在 $hN(\mathcal C)$ 中同构，当且仅当 $f$ 在 $\mathcal C$ 中同构。$\square$

**命题 18.17.** 若 $C$ 是 Kan 复形，则 $hC$ 是群胚。

**证明.** 第十七章命题 17.18 已说明 Kan 复形作为 quasi-category 时每条边在同伦范畴中可逆。因此 $hC$ 的每个态射都有逆，故 $hC$ 是群胚。$\square$

**例子 18.18（同伦范畴的遗忘性）.** 若一个 quasi-category 的两个平行边之间存在不同的高阶同伦信息，同伦范畴只记录这些边的连通分支。换言之，$hC(x,y)=\pi_0\operatorname{Map}_C(x,y)$，而映射空间的高阶同伦群和相干复合数据被遗忘。因此“$hC\simeq hD$”通常不能推出 $C\simeq D$。

## 18.8 本章小结

$\infty$-范畴把 Hom 集替换为映射空间，把唯一性替换为可缩空间中的唯一性。Join 和 slice 给出锥范畴的 quasi-categorical 定义；ordinary nerve 中的等价边正是同构。普通极限、伴随和保持性定理仍成立，但证明必须在映射空间层面进行，而不是只看同伦范畴。

## 练习

**练习 18.1.** 解释为什么只知道 $hC$ 不足以恢复 $\operatorname{Map}_C(x,y)$。

**练习 18.2.** 对普通范畴 $\mathcal C$，计算 $hN(\mathcal C)$。

**练习 18.3.** 写出终对象在 $\infty$-范畴中的映射空间刻画。

**练习 18.4.** 对定理 18.15 的右伴随版本给出完整证明。

**练习 18.5.** 查阅 HTT 中 adjunction between $\infty$-categories 的定义，并比较本章定义 18.13。

**练习 18.6.** 用 join 的公式验证 $\Delta^0\star\Delta^1\cong\Delta^2$。

**练习 18.7.** 用 slice 的泛性质描述 $C_{/x}$ 的对象。

**练习 18.8.** 证明命题 18.12 的始对象对偶版本。

**练习 18.9.** 对 ordinary span $A\to C\leftarrow B$，写出 $N(\mathcal C)_{/p}$ 中一个 $1$-单纯形对应的普通锥态射。

**练习 18.10.** 证明 ordinary initial object 与 $N(\mathcal C)$ 中的始对象一致。

**练习 18.11.** 写出 $\operatorname{Map}^R_C(x,y)_0$ 和 $\operatorname{Map}^R_C(x,y)_1$ 的数据。

**练习 18.12.** 证明 $\operatorname{Map}^L_{N(\mathcal A)}(x,y)$ 也是离散单纯集 $\mathcal A(x,y)$。

**练习 18.13.** 在普通范畴情形下，把定义 18.C 展开为集合值双函子，并说明左可表示性意味着什么。

**练习 18.14.** 用命题 18.D 重新证明第四章 Hom 自然同构版本的伴随定义。

**练习 18.15.** 在定义 18.F 中，说明 coCartesian 传输为什么方向为 $C\to D$，而 Cartesian 传输为什么方向为 $D\to C$。

**练习 18.16.** 按构造 18.G，在普通范畴的 Grothendieck fibration 类比中写出单位 $\eta_x:x\to GFx$ 的来源。

**练习 18.17.** 写出 walking adjunction 中两条三角恒等式分别作用在哪个 1-态射上。

**练习 18.18.** 解释为什么命题 18.J 只推出同伦范畴中的严格三角恒等式，而不是说高阶相干数据消失。

**练习 18.19.** 说明 scaled simplicial set 与 marked simplicial set 标记的数据维度有何不同。

**练习 18.20.** 对普通伴随 $F\dashv G$，写出严格 $2$-函子 $\operatorname{Adj}\to\mathbf{Cat}$ 在两个对象和两个生成 $1$-态射上的取值。

**练习 18.21.** 解释为什么三角恒等式是构造 18.L 中必须包含的 $2$-维关系。

**练习 18.22.** 比较 definition by correspondence 与 definition by walking adjunction：二者分别强调伴随的哪一面？

**练习 18.23.** 证明命题 18.16 的反向：普通同构在 nerve 中给出等价边。

**练习 18.24.** 说明为什么 $hC$ 只能恢复每个映射空间的 $\pi_0$。

**练习 18.25.** 若 $C$ 是 Kan 复形，解释为什么它作为 $\infty$-范畴没有非可逆 $1$-态射。
