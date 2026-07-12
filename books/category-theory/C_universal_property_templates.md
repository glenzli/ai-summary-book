# 附录 C：常用泛性质证明模板

## 本章目标

本附录给出正文中反复使用的证明模板，重点区分底层对象与带结构候选、协变与反变自然性、任意同构与保持泛结构的唯一同构。模板只列证明责任，不能替代在具体章节中验证类型、存在性和交换等式。

## 依赖前置知识

需要第二章的 Yoneda 引理、第三章的锥范畴、第四章的伴随，以及第六章的 Kan 延拓定义。大小符号采用附录 A。

## C.1 结构化唯一同构模板

**模板 C.1.** 设一个泛性质被写成候选范畴 $\mathcal K$ 中的始对象或终对象，且有忘却函子

$$
U:\mathcal K\to\mathcal C.
$$

要证明两个终候选 $A,B\in\mathcal K$ 唯一同构：

1. 由 $B$ 的终性构造 $\mathcal K$ 中唯一态射 $u:A\to B$。
2. 由 $A$ 的终性构造 $\mathcal K$ 中唯一态射 $v:B\to A$。
3. $vu$ 与 $\operatorname{id}_A$ 都是 $A\to A$ 的 $\mathcal K$-态射，故终性给出 $vu=\operatorname{id}_A$。
4. 同样以 $B$ 为终对象，得到 $uv=\operatorname{id}_B$。
5. 若 $u'$ 是另一个 $\mathcal K$ 中的同构 $A\to B$，它首先是同一 Hom 集中的态射，故 $u'=u$。

始对象情形把所有箭头反转。忘却后得到

$$
U(u):U(A)\xrightarrow{\cong}U(B)
$$

在 $\mathcal C$ 中是同构，但唯一性只对能提升为 $\mathcal K$-态射的底层同构成立。除非另有证明，不得声称 $U(A)$ 与 $U(B)$ 之间只有一个任意同构。

积的候选范畴以三元组 $(P,p_A,p_B)$ 为对象，以同时保持两个投影的态射为态射。即使底层对象 $P$ 有许多自同构，积结构的自同构仍只有恒等态射。这是“唯一到唯一同构”的标准含义。

## C.2 表示性与方差模板

**模板 C.2.** 设 $\mathcal C$ 局部 $\mathcal U$-小。要证明对象 $R$ 表示协变函子

$$
F:\mathcal C\to\mathbf{Set}_{\mathcal U},
$$

必须完成：

1. 构造分量
   $$
   \theta_X:\mathcal C(R,X)\to F(X).
   $$
2. 对每个 $X$ 证明 $\theta_X$ 为双射，分别给出逆映射或存在性与唯一性。
3. 对每个 $f:X\to Y$ 和 $g:R\to X$ 验证
   $$
   F(f)(\theta_X(g))=\theta_Y(fg).
   $$
4. 令 $u=\theta_R(\operatorname{id}_R)$，记录结构化唯一性条件：另一表示
   $(R',u')$ 与 $(R,u)$ 之间唯一的相容同构 $\phi:R\to R'$ 满足
   $F(\phi)(u)=u'$。

对反变函子

$$
P:\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U},
$$

表示映射的类型改为

$$
\theta_X:\mathcal C(X,R)\to P(X).
$$

若 $f:X\to Y$、$g:Y\to R$，自然性必须写成

$$
P(f)(\theta_Y(g))=\theta_X(gf).
$$

不能把协变公式中的箭头原样抄到反变情形。协变泛元素是元素范畴的始对象；反变泛元素是元素范畴的终对象。

## C.3 极限模板

**模板 C.3.** 设 $\mathcal J$ 为 $\mathcal U$-小范畴，
$D:\mathcal J\to\mathcal C$，且 $\mathcal C$ 局部 $\mathcal U$-小。要证明
$(L,\pi_j)$ 是 $D$ 的极限：

1. 类型检查每个 $\pi_j:L\to D(j)$。
2. 对每个 $\alpha:j\to k$ 验证锥等式
   $$
   D(\alpha)\pi_j=\pi_k.
   $$
3. 给定任意锥 $(X,\lambda_j)$，构造候选 $u:X\to L$。
4. 对每个 $j$ 验证 $\pi_j u=\lambda_j$。
5. 若 $u':X\to L$ 也满足这些等式，指出哪一个泛性质或哪一族联合检测态射，从而推出 $u'=u$。

第 5 步不能只写“由唯一性”，除非已经说明使用的是哪个对象的哪条唯一性。完成后得到自然同构

令 $\operatorname{Cone}(X,D)$ 表示所有顶点固定为 $X$ 的锥组成的
$\mathcal U$-小集合；对 $f:X'\to X$ 预复合每条锥腿，使
$X\mapsto\operatorname{Cone}(X,D)$ 成为反变函子。于是上述泛性质给出

$$
\mathcal C(-,L)\xrightarrow{\cong}
\operatorname{Cone}(-,D)
$$

作为 $\mathcal C^{\operatorname{op}}\to\mathbf{Set}_{\mathcal U}$ 的同构；其分量把 $u$ 送到 $(\pi_j u)_j$。两个极限之间的唯一同构必须满足

$$
\pi'_j u=\pi_j\qquad(\forall j).
$$

余极限模板把箭头反转，并把反变 Hom 函子
$\mathcal C(-,L)$ 换成协变 Hom 函子 $\mathcal C(Q,-)$。

## C.4 伴随模板

**模板 C.4.** 要证明 $F:\mathcal C\to\mathcal D$ 左伴随于
$G:\mathcal D\to\mathcal C$，先给出双射

$$
\Phi_{X,Y}:\mathcal D(FX,Y)\xrightarrow{\cong}\mathcal C(X,GY).
$$

证明责任为：

1. 给出 $\Phi$ 与 $\Phi^{-1}$ 的公式并证明互逆。
2. 对 $u:X'\to X$ 验证第一变量的反变自然性
   $$
   \Phi_{X',Y}(fF(u))=\Phi_{X,Y}(f)u.
   $$
3. 对 $v:Y\to Y'$ 验证第二变量的协变自然性
   $$
   \Phi_{X,Y'}(vf)=G(v)\Phi_{X,Y}(f).
   $$

若改用单位和余单位，必须给出自然变换

$$
\eta:\operatorname{id}_{\mathcal C}\Rightarrow GF,\qquad
\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}
$$

并分别验证自然性和两个三角恒等式

$$
\varepsilon_{FX}F(\eta_X)=\operatorname{id}_{FX},
\qquad
G(\varepsilon_Y)\eta_{GY}=\operatorname{id}_{GY}.
$$

两个左伴随 $F,F'$ 的唯一相容同构
$\theta:F\Rightarrow F'$ 由

$$
G(\theta_X)\eta_X=\eta'_X
$$

刻画；等价地，它满足
$\varepsilon'_Y\theta_{GY}=\varepsilon_Y$。若没有写出这条相容式，“伴随唯一”仍是不完整陈述。

## C.5 Kan 延拓模板

**模板 C.5.** 对
$K:\mathcal C\to\mathcal D$ 与 $F:\mathcal C\to\mathcal E$，左 Kan 延拓候选
$(L,\eta)$ 的类型是

$$
L:\mathcal D\to\mathcal E,\qquad
\eta:F\Rightarrow LK.
$$

其泛性质必须明确写成映射

$$
\operatorname{Nat}(L,H)\longrightarrow\operatorname{Nat}(F,HK),
\qquad
\theta\longmapsto(\theta K)\eta
$$

对每个 $H$ 都是双射且对 $H$ 自然。两个左 Kan 延拓之间的唯一相容同构
$\alpha:L\Rightarrow L'$ 满足

$$
(\alpha K)\eta=\eta'.
$$

点态证明还需：

1. 声明 $K/d$ 的大小以及投影 $\pi_d:K/d\to\mathcal C$。
2. 假设特定图形 $F\pi_d$ 的余极限存在，而不是无说明地写一个可能不存在的余极限。
3. 由余极限结构映射构造 $L$ 在态射上的作用，并用结构映射联合检测恒等和复合。
4. 验证 $F\Rightarrow LK$ 的自然性。
5. 构造 Kan 双射的逆，并验证两个复合、对 $d$ 的自然性和对 $H$ 的自然性。

右 Kan 延拓把结构映射改为
$\varepsilon:RK\Rightarrow F$，泛性质映射为

$$
\operatorname{Nat}(H,R)\longrightarrow\operatorname{Nat}(HK,F),
\qquad
\theta\longmapsto\varepsilon(\theta K),
$$

点态索引改为 $d/K$ 上的 $F\rho_d$ 极限。

## C.6 普通与高阶唯一性的边界

普通范畴中，始对象或终对象之间有唯一的结构保持同构。在
$\infty$-范畴中，相应陈述升级为：若普遍对象存在，则普遍对象及其等价组成的
$\infty$-群胚是可缩的。存在性必须另证；一旦存在，等价性和高阶相干由可缩性记录。不能把“可缩选择空间”降格成
某个 Hom 集中严格只有一个元素，也不能把普通范畴的唯一同构措辞未经修改地搬入高阶章节。

## C.7 外部输入定理模板

**模板 C.6.** 使用外部输入定理时，正文必须说明：

1. 可引用的定理名称或准确主题；
2. 本书使用的完整假设和结论类型；
3. 它在当前证明中承担存在性、比较、相干性还是计算作用；
4. 来源在 [SOURCES.md](SOURCES.md)、[CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md) 或附录 D 中的位置；
5. 本书未重证的边界，以及后文哪些结论只把它当作输入。

证明路线只能解释来源定理如何适用，不能替代来源定理本身的证明状态。

## C.8 本章小结

泛性质证明的共同结构是：先建立候选范畴或表示函子的正确类型，再证明存在性、唯一性和自然性。对象唯一性总是相对于保留的结构；方差决定自然性方块方向；大小假设决定 Hom、Nat、极限和 Kan 点态公式是否存在于所声明的 universe。

## 练习

**练习 C.1.** 用模板 C.1 证明二元积唯一，并明确写出“保持投影”的相容条件。

**练习 C.2.** 用模板 C.2 证明自由群表示函子
$G\mapsto\mathbf{Set}_{\mathcal U}(S,UG)$。

**练习 C.3.** 用模板 C.4 证明
$-\times A\dashv(-)^A$ 于 $\mathbf{Set}_{\mathcal U}$，并验证双射对两个变量自然。
