# 范畴论答案手册

本文件给出 `books/category-theory/` 当前版本全部练习的参考答案。答案以“检查定义、写出泛性质、标明自然性变量”为原则；对需要大型外部理论的题目，给出标准结论和应核对的来源。

## 序章

**答案 0.1.** 若不指定 universe，“小范畴”的对象集合可在任意大集合中变化；所有小范畴的对象类不再是同一层级中的集合。固定 $\mathcal U\in\mathcal V$ 后，$\mathcal U$-小范畴组成 $\mathcal V$-层级中的范畴 $\mathbf{Cat}_{\mathcal U}$。

**答案 0.2.** 例：集合 $A,B$ 的积 $A\times B$。输入为二元组 $(A,B)$，输出为集合 $P$ 和投影 $P\to A,P\to B$。候选自然双射为
$$
\mathbf{Set}(X,P)\cong \mathbf{Set}(X,A)\times\mathbf{Set}(X,B).
$$

**答案 0.3.** 同构要求存在严格可逆函子，等价只要求完全忠实且本质满。例：一个含两个同构对象和一条唯一同构的群胚等价于终范畴，但对象数不同，所以不严格同构。

**答案 0.4.** 终对象的定义只保证对每个 $X$，Hom 集 $\mathcal C(X,1)$ 是单点集。若 $1$ 与 $1'$ 都终，则存在唯一态射 $1\to1'$ 与 $1'\to1$，且复合必为恒等，所以二者唯一同构。除非范畴的数据本身把 $1$ 与 $1'$ 定义为同一对象，否则不能说严格相等。

**答案 0.5.** 范畴等价的拟逆构造需要为目标范畴中每个对象选择一个同构于它的像对象。不同选择给出可能不同的拟逆函子，但完全忠实性和本质满性保证这些拟逆之间唯一到自然同构相容，因此等价结论不依赖具体选择。

## 第一章

**答案 1.1.** 若采用含幺环和保持单位的环同态，恒等函数保持加法、乘法和单位；两个保持单位的环同态复合仍保持这些结构。函数复合的结合律和单位律给出范畴公理。

**答案 1.2.** 单对象范畴中态射复合就是幺半群乘法。态射 $m$ 同构当且仅当存在 $n$ 使 $nm=e=mn$，即 $m$ 是幺半群中的可逆元素。

**答案 1.3.** 薄范畴中 $x\cong y$ 意味着 $x\le y$ 且 $y\le x$；偏序的反对称性给出 $x=y$。反过来 $x=y$ 时恒等态射给出同构。

**答案 1.4.** 对 $f:X\to Y$ 定义 $\mathcal C(A,f)(u)=fu$。恒等性来自 $\operatorname{id}_X u=u$；复合性来自 $(gf)u=g(fu)$。

**答案 1.5.** 恒等自然变换为 $(\operatorname{id}_F)_X=\operatorname{id}_{F X}$。纵向复合结合律逐点化为 $\mathcal D$ 中态射复合结合律。

**答案 1.6.** 包含函子 $\mathbf{Ab}\hookrightarrow\mathbf{Grp}$ 完全忠实但非本质满，因为非阿贝尔群不与阿贝尔群同构。

**答案 1.7.** 对任意 $X,Y$，Hom 映射
$$
\mathcal C(X,Y)\to\mathcal E(GFX,GFY)
$$
是两个双射的复合，故为双射。

**答案 1.8.** 令 $\mathcal C$ 为有两个同构对象 $a,b$ 的连通群胚，且任意 Hom 集单点；令 $\mathcal D=*$。唯一函子 $\mathcal C\to\mathcal D$ 完全忠实且本质满，是等价；但对象数不同，不可能是严格同构。

**答案 1.9.** 若 $\mathcal S,\mathcal T$ 都是 $\mathcal C$ 的骨架，则包含函子 $\mathcal S\hookrightarrow\mathcal C$ 与 $\mathcal T\hookrightarrow\mathcal C$ 都是等价。对每个 $S\in\mathcal S$，存在唯一到同构的 $T\in\mathcal T$ 与其同构；骨架条件使该 $T$ 唯一。选择这些同构并用全子范畴的 Hom 集识别得到等价 $\mathcal S\simeq\mathcal T$。

**答案 1.10.** 若 $F$ 完全忠实且 $F(f)$ 有逆 $u$，由完全性取 $g$ 使 $F(g)=u$。则
$$
F(gf)=uF(f)=\operatorname{id},\qquad F(fg)=F(f)u=\operatorname{id}.
$$
由忠实性得 $gf=\operatorname{id}$、$fg=\operatorname{id}$，故 $f$ 是同构。

**答案 1.11.** 唯一函子 $BG\to *$ 对非平凡群 $G$ 本质满，但不忠实，因为 Hom 映射 $G\to\{*\}$ 不是单射。

**答案 1.12.** 唯一函子 $BG\to *$ 本质满且完全，因为 $G\to\{*\}$ 为满射；当 $G$ 非平凡时不忠实，因为该 Hom 映射不是单射。因此它不是范畴等价。

## 第二章

**答案 2.1.** 在 $\mathcal C^{\operatorname{op}}$ 中，始对象变为终对象。由终对象唯一性，始对象在唯一同构意义下唯一。

**答案 2.2.** 在 $\mathbf{Set}_{\mathcal U}$ 中，$A\times B$ 带投影 $p_A,p_B$，对任意 $X$ 有自然双射
$$
\mathbf{Set}(X,A\times B)\cong\mathbf{Set}(X,A)\times\mathbf{Set}(X,B).
$$
它表示函子 $X\mapsto\mathbf{Set}(X,A)\times\mathbf{Set}(X,B)$。

**答案 2.3.** 单对象范畴 $\mathcal C_M$ 上的预层是函子 $\mathcal C_M^{op}\to\mathbf{Set}$，等价于带右 $M$-作用的集合。

**答案 2.4.** 对 $a\in F(A)$ 定义 $\alpha^a_X(f:A\to X)=F(f)(a)$。自然性由 $F(gf)=F(g)F(f)$。任意自然变换由 $\alpha_A(\operatorname{id}_A)$ 唯一决定。

**答案 2.5.** 若 $y(f):yA\to yB$ 是同构，则其逆为某个 $y(g)$，因为 $y$ 完全忠实。于是 $y(gf)=\operatorname{id}_{yA}$ 且 $y(fg)=\operatorname{id}_{yB}$；忠实性给出 $gf=\operatorname{id}_A$、$fg=\operatorname{id}_B$。

**答案 2.6.** 若 $\mathcal C(A,-)\cong F\cong\mathcal C(B,-)$，则特别有自然同构 $\mathcal C(A,-)\cong\mathcal C(B,-)$。协变 Yoneda 给出唯一同构 $B\to A$，等价地 $A\cong B$；方向取决于采用 $\mathcal C(A,-)$ 的协变 Yoneda 约定。

**答案 2.7.** 恒等函子由单点集 $1$ 表示，因为
$$
\mathbf{Set}(1,X)\cong X
$$
自然成立。泛元素是 $1$ 中唯一元素在 $\operatorname{id}_{\mathbf{Set}}(1)=1$ 中的元素。

**答案 2.8.** 若常值双点函子由 $A$ 表示，则
$$
\mathbf{Set}(A,\varnothing)\cong 2.
$$
这迫使 $A=\varnothing$ 不成立时左边为空；若 $A=\varnothing$，左边为单点集，也不等于双点集。故不可表。

**答案 2.9.** 自然变换 $\alpha:P\Rightarrow Q$ 在每个 $X$ 上的函数可分别限制到两个不交分支，得到自然变换
$\alpha^A:yA\Rightarrow Q$ 与 $\alpha^B:yB\Rightarrow Q$。反过来，一对这样的自然变换在两个分支上逐段定义唯一的
$\alpha_X:P(X)\to Q(X)$，两个分支的自然性给出 $\alpha$ 的自然性。因此
$$
\operatorname{Nat}(P,Q)
\cong\operatorname{Nat}(yA,Q)\times\operatorname{Nat}(yB,Q)
\cong Q(A)\times Q(B).
$$

**答案 2.10.** 对偶函子类型为
$$
(-)^*:(\mathbf{Vect}^{\mathrm{fd}}_k)^{\operatorname{op}}
\to\mathbf{Vect}^{\mathrm{fd}}_k,
$$
而 $\operatorname{id}$ 与 $(-)^{**}$ 都是
$\mathbf{Vect}^{\mathrm{fd}}_k\to\mathbf{Vect}^{\mathrm{fd}}_k$。因此
$\operatorname{id}\Rightarrow(-)^*$ 的源范畴不匹配，不能写成自然变换。评价映射
$\iota_V(v)(\lambda)=\lambda(v)$ 满足
$f^{**}\iota_V=\iota_Wf$，故给出
$\operatorname{id}\Rightarrow(-)^{**}$；在有限维情形它逐分量可逆。

## 第三章

**答案 3.1.** 图形为 $A\xrightarrow{f}C\xleftarrow{g}B$。拉回 $P$ 带 $p_A:P\to A,p_B:P\to B$，满足 $fp_A=gp_B$；任意 $X$ 上相容二元组唯一分解经 $P$。

**答案 3.2.** 集合 $P=\{(a,b)\mid f(a)=g(b)\}$ 带投影。相容映射 $x\mapsto a_x,b_x$ 唯一给出 $x\mapsto(a_x,b_x)$。

**答案 3.3.** 余等化子为 $q:B\to Q$，满足 $qf=qg$；任意 $h:B\to X$ 若 $hf=hg$，存在唯一 $\bar h:Q\to X$ 使 $\bar hq=h$。

**答案 3.4.** 对小图形 $D:\mathcal J\to\mathcal C$，令
$$
P=\prod_jD(j),\qquad Q=\prod_{\alpha:j\to k}D(k).
$$
两箭头 $P\rightrightarrows Q$ 由 $D(\alpha)p_j$ 与 $p_k$ 给出。其等化子即极限。

**答案 3.5.** 极限逐点由第三章命题 3.12。余极限逐点同理，因自然变换到逐点余极限的自然性逐对象验证，且目标范畴 $\mathbf{Set}$ 余极限逐点存在。

**答案 3.6.** 若 $Q=\operatorname{colim}D$，结构映射为 $\iota_j:Dj\to Q$，则
$$
\mathcal C(Q,X)\to\lim_{j\in\mathcal J^{op}}\mathcal C(Dj,X)
$$
把 $h:Q\to X$ 送到族 $h\iota_j$。相容性来自余锥条件。反向地，相容族 $h_j:Dj\to X$ 是到 $X$ 的余锥，由余极限泛性质唯一给出 $h:Q\to X$。对 $X$ 的自然性由后复合保持。

**答案 3.7.** 若 $L\dashv U$，单位 $\eta_j:j\to ULj$ 给出 $j/U$ 中对象 $(Lj,\eta_j)$。任意 $(i,\alpha:j\to Ui)$ 对应唯一 $\bar\alpha:Lj\to i$，满足 $U\bar\alpha\,\eta_j=\alpha$，故 $(Lj,\eta_j)$ 为始对象。

**答案 3.8.** 对偏序包含 $A\hookrightarrow P$，$p/U$ 是由所有 $a\in A$ 且 $p\le a$ 组成的偏序。因此包含共尾当且仅当每个集合
$$
A_{\ge p}=\{a\in A\mid p\le a\}
$$
非空且其可比性无向图连通。若这些 $A_{\ge p}$ 还都是滤过的，则连通性自动成立。

**答案 3.9.** 设 $t$ 为终对象。$D(t)$ 带余锥 $D(j)\xrightarrow{D(!)}D(t)$。任意余锥 $\lambda_j:Dj\to X$ 中，$\lambda_t:D(t)\to X$ 唯一决定全体分量，因为 $\lambda_j=\lambda_tD(j\to t)$。故 $D(t)$ 为余极限。

**答案 3.10.** 满射：任意代表 $(p,x)$，取 $a\in A$ 且 $p\le a$，则 $(p,x)$ 与 $(a,D(p\le a)x)$ 等价。单射：若 $a,b\in A$ 的元素在 $\operatorname{colim}_P D$ 中相等，滤过性给出 $p\ge a,b$ 使二者在 $D(p)$ 中相等；再取 $c\in A$ 且 $p\le c$，则二者已在 $\operatorname{colim}_A D$ 中相等。

**答案 3.11.** 令 $R=\{(f(a),g(a))\mid a\in A\}$。包含 $R$ 的等价关系的交仍是包含 $R$ 的等价关系，因此存在最小等价关系 $\sim$。商映射 $q:B\to B/{\sim}$ 满足 $qf=qg$。若 $r:B\to Q$ 满足 $rf=rg$，则关系核
$$
b\equiv b'\quad\Longleftrightarrow\quad r(b)=r(b')
$$
是包含 $R$ 的等价关系，故 $\sim$ 包含于 $\equiv$，于是 $r$ 唯一分解经 $q$。

**答案 3.12.** 给环值图形 $D:\mathcal J\to\mathbf{Ring}$。先在 Set 中取底层极限 $L$，即相容族集合。逐坐标定义加法、乘法、$0$ 和 $1$：
$$
(x_j)+(y_j)=(x_j+y_j),\qquad (x_j)(y_j)=(x_jy_j),\qquad 0=(0_j),\qquad 1=(1_j).
$$
因为结构映射是保持单位的环同态，相容族对这些运算封闭。投影是环同态。任意环到图形的相容锥在底层集合上唯一分解，经逐坐标运算检查该分解保持加法、乘法和单位，故忘却函子创造小极限。

**答案 3.13.** 非空集合范畴有终对象 $1$，有限积由通常笛卡尔积给出，空积为 $1$，有限个非空集合的积仍非空。但例子 3.22 中两函数 $\{*\}\rightrightarrows\{0,1\}$ 的等化子在 Set 中为空集，不属于该范畴，因此没有所有等化子。有限极限若全存在则等化子存在，所以它没有所有有限极限。

## 第四章

**答案 4.1.** 给集合映射 $S\to U(A)$，由自由阿贝尔群泛性质唯一延拓为群同态 $\mathbb Z[S]\to A$。自然双射为
$$
\mathbf{Ab}(\mathbb Z[S],A)\cong\mathbf{Set}(S,U A).
$$

**答案 4.2.** 右伴随为 $(-)^A$。双射
$$
\mathbf{Set}(A\times X,Y)\cong\mathbf{Set}(X,Y^A)
$$
由 currying 给出，$f(a,x)$ 对应 $x\mapsto(a\mapsto f(a,x))$。

**答案 4.3.** 若 $G$ 右伴随，且 $1_{\mathcal D}$ 为终对象，则
$$
\mathcal C(X,G1)\cong\mathcal D(FX,1)
$$
为单点集，所以 $G1$ 终。

**答案 4.4.** 若 $F$ 完全忠实，则 $\mathcal C(X,Y)\to\mathcal D(FX,FY)$ 是双射，单位是 $\operatorname{id}_{FX}$ 的转置，因而为同构。反过来若 $\eta$ 是同构，则任意 $f:FX\to FY$ 唯一转置为 $\eta_Y^{-1}G(f)\eta_X:X\to Y$，故 $F$ 完全忠实。

**答案 4.5.** 对图形 $D$ 的极限 $L$ 和任意 $X$：
$$
\mathcal C(X,G L)\cong\mathcal D(FX,L)
\cong\lim_j\mathcal D(FX,Dj)
\cong\lim_j\mathcal C(X,G Dj).
$$
由表示性，$G L$ 是 $GD$ 的极限。

**答案 4.6.** 若 $G$ 完全忠实，则在伴随双射下
$$
\mathcal D(FGY,Y)\to\mathcal C(GY,GY)
$$
中 $\varepsilon_Y$ 对应 $\operatorname{id}_{GY}$，且由全忠实性推出 $\varepsilon_Y$ 为同构。反过来若 $\varepsilon$ 同构，则任意 $h:GX\to GY$ 的逆像为
$$
\varepsilon_YF(h)\varepsilon_X^{-1}:X\to Y,
$$
从而 $G$ 在 Hom 集上为双射。

**答案 4.7.** 若环境余极限 $C$ 已在 $\mathcal A$ 中，则对任意 $A\in\mathcal A$，
$$
\mathcal A(C,A)\cong\mathcal C(IC,IA)
\cong\lim_j\mathcal C(IDj,IA)
\cong\lim_j\mathcal A(Dj,A).
$$
所以 $C$ 满足 $\mathcal A$ 中余极限的表示性。

**答案 4.8.** 由伴随和全忠实性，
$$
\mathcal A(LIA,B)\cong\mathcal C(IA,IB)\cong\mathcal A(A,B)
$$
对 $B$ 自然。Yoneda 引理给出 $LIA\cong A$，这就是余单位为同构。

**答案 4.9.** $G_{\operatorname{ab}}=G/[G,G]$。任意群同态 $G\to A$ 到阿贝尔群 $A$ 都杀死交换子子群 $[G,G]$，故唯一分解为群同态 $G_{\operatorname{ab}}\to A$。这给出
$$
\mathbf{Ab}(G_{\operatorname{ab}},A)\cong\mathbf{Grp}(G,IA),
$$
所以 $\mathbf{Ab}$ 是 $\mathbf{Grp}$ 的反射子范畴。

**答案 4.10.** 余反射子范畴指包含 $I:\mathcal A\hookrightarrow\mathcal C$ 有右伴随 $R$。若 $C=\lim ID$ 在环境范畴中存在，则
$$
\mathcal A(A,RC)\cong\mathcal C(IA,C)
\cong\lim_j\mathcal C(IA,IDj)
\cong\lim_j\mathcal A(A,Dj),
$$
故 $RC$ 是 $\mathcal A$ 中的极限。

**答案 4.11.** 在偏序范畴中 Hom 集至多单点。伴随双射
$$
Q(fp,q)\cong P(p,gq)
$$
等价于左侧非空当且仅当右侧非空，即 $f(p)\le q$ 当且仅当 $p\le g(q)$。

**答案 4.12.** 若 $\mathcal C$ 有终对象 $1$，唯一函子 $!:\mathcal C\to *$ 的右伴随把 $*$ 送到 $1$，因为
$$
*(!X,*)\cong\{*\}\cong\mathcal C(X,1).
$$
若 $\mathcal C$ 有始对象 $0$，同理把 $*$ 送到 $0$ 给出左伴随，因为
$$
\mathcal C(0,X)\cong\{*\}\cong *(*,!X).
$$

**答案 4.13.** 笛卡尔闭范畴中指数对象 $Y^A$ 按定义满足自然同构
$$
\mathcal C(A\times X,Y)\cong\mathcal C(X,Y^A).
$$
这正是函子 $A\times -$ 左伴随于 $(-)^A$ 的 Hom 双射。

**答案 4.14.** 忘却函子 $U:\mathbf{Grp}\to\mathbf{Set}$ 是右伴随，但不保持余积。两个平凡群在 $\mathbf{Grp}$ 中的余积仍为平凡群，其底层集合为单点；而两个单点集在 $\mathbf{Set}$ 中的余积有两个元素。因此 $U$ 不保持该余积。

## 第五章

**答案 5.1.** 投影函子 $\int_{\mathcal C}P\to\mathcal C$ 送 $(C,x)$ 到 $C$，送态射 $f:(C,x)\to(D,y)$ 到 $f:C\to D$。

**答案 5.2.** $yA(C)=\mathcal C(C,A)$，故元素范畴对象为 $(C,u:C\to A)$，态射为使三角形交换的 $C\to D$；这正是 slice 范畴 $\mathcal C/A$。

**答案 5.3.** 若 $(C,x,f)$ 与 $(D,y,g)$ 在 $P(A)$ 中有同一像 $a$，则 $f:(A,a)\to(C,x)$ 与 $g:(A,a)\to(D,y)$ 在元素范畴中把二者同时连到 $(A,a,\operatorname{id})$，故在余极限商中相等。

**答案 5.4.** 若 $f,g:X\rightrightarrows Y$ 不同，取 $x\in X$ 使 $f(x)\ne g(x)$。单点集 $1\to X$ 选出 $x$，于是 $f\circ x\ne g\circ x$。

**答案 5.5.** 群同态 $\mathbb Z\to G$ 等价于选择 $G$ 的一个元素。若 $f,g:G\rightrightarrows H$ 不同，取 $x$ 使 $f(x)\ne g(x)$，对应 $\mathbb Z\to G$ 检测二者不同。

**答案 5.6.** 单对象族 $\{G\}$ 生成意味着任意不同 $f,g:X\rightrightarrows Y$ 可被某个 $u:G\to X$ 区分，即 $\mathcal C(G,f)\ne\mathcal C(G,g)$。这正是 Hom 函子 $\mathcal C(G,-)$ 在 Hom 集上映射为单射，也就是忠实。

**答案 5.7.** $\mathbf{Ab}(\mathbb Z,A)\cong A$，所以 $\mathbb Z$ 检测不同群同态，因而是生成元。若 $e:B\to C$ 是满同态且 $f:\mathbb Z\to C$，则 $f(1)=c$ 可提升为某个 $b\in B$，由 $1\mapsto b$ 唯一确定 $\tilde f:\mathbb Z\to B$，且 $e\tilde f=f$。故 $\mathbb Z$ 投射。

**答案 5.8.** 当 $\mathcal C=*$ 时，预层就是集合 $S$，元素范畴是离散集合 $S$，每个元素对应一个可表预层即单点集。密度公式变成
$$
S\cong\coprod_{s\in S}1.
$$

## 第六章

**答案 6.1.** 若 $u:(c,\alpha)\to(c',\alpha')$、$v:(c',\alpha')\to(c'',\alpha'')$，则 $\alpha''K(vu)=\alpha'K(u)=\alpha$。恒等态射满足 $\alpha K(\operatorname{id})=\alpha$。

**答案 6.2.** $\operatorname{Lan}_iF(c)$ 由所有 $a\in\mathcal A$ 及箭头 $a\to c$ 上的 $F(a)$ 的余极限给出，是把 $F$ 从子范畴按所有进入 $c$ 的方式自由延拓。

**答案 6.3.** 对 $K=\operatorname{id}$，逗号范畴 $\operatorname{id}/c$ 有终对象 $(c,\operatorname{id})$，故左 Kan 延拓点态为 $F(c)$；右 Kan 延拓同理。

**答案 6.4.** 对 $\beta:F\to HK$ 构造 $\bar\beta_d:Ld\to Hd$。若 $v:d\to d'$，两条 $Ld\to Hd'$ 在每个 $F(c)\to Ld$ 上都为 $H(v\alpha)\beta_c$，故相等。

**答案 6.5.** 预层密度可写为
$$
P\cong \operatorname{Lan}_{y_P} y
$$
沿元素范畴投影到 $\mathcal C$ 后由 Yoneda 嵌入作左 Kan 延拓；点态公式就是 co-Yoneda 公式。

**答案 6.6.** 对恒等 $v=\operatorname{id}_d$，$R(v)$ 与 $\operatorname{id}_{R(d)}$ 在所有投影下都等于同一投影。对 $d\xrightarrow{v}d'\xrightarrow{w}d''$，$R(w)R(v)$ 与 $R(wv)$ 在任意投影 $p_{(c,\beta:d''\to Kc)}$ 下都等于 $p_{(c,\beta wv)}$，故相等。

**答案 6.7.** 若 $(c',\beta:Kc\to Kc')$ 是 $Kc/K$ 的对象，完全忠实性给出唯一 $u:c\to c'$ 使 $K(u)=\beta$。这正是从 $(c,\operatorname{id})$ 到 $(c',\beta)$ 的唯一态射，因此 $(c,\operatorname{id})$ 为始对象。

**答案 6.8.** 若 $H:\mathcal E\to\mathcal E'$ 保持所有 $d/K$ 形状极限，则
$$
H((\operatorname{Ran}_K F)(d))
\cong H\left(\lim_{d/K}F\right)
\cong \lim_{d/K}HF
\cong(\operatorname{Ran}_K HF)(d),
$$
且这些同构对 $d$ 自然。

**答案 6.9.** 当 $K:\mathcal C\to *$ 时，唯一对象 $*/K$ 与 $K/*$ 都同构于 $\mathcal C$。故
$$
\operatorname{Lan}_KF(*)\cong\operatorname{colim}_{\mathcal C}F,\qquad
\operatorname{Ran}_KF(*)\cong\lim_{\mathcal C}F.
$$

**答案 6.10.** 对任意 $X$，限制给出
$$
\operatorname{Cocone}(F\pi,X)\to\operatorname{Cocone}(F\pi V,X).
$$
因 $V$ 共尾，第三章定理 3.16 的构造把右侧余锥唯一扩张为 $K/d$ 上余锥，且扩张和限制互逆。因此两个余锥函子自然同构，表示对象即同构。

**答案 6.11.** 若 $K\dashv R$，余单位 $\varepsilon_d:KRd\to d$ 给出 $K/d$ 中对象。任意 $(c,\alpha:Kc\to d)$ 的伴随转置为唯一 $\bar\alpha:c\to Rd$，满足 $\varepsilon_dK\bar\alpha=\alpha$，这正是 $(c,\alpha)\to(Rd,\varepsilon_d)$ 的唯一态射。

**答案 6.12.** 对 $i:\{0,1\}\hookrightarrow[1]$，$0/i$ 有两个对象 $(0,\operatorname{id}:0\to0)$ 与 $(1,0\to1)$，且由于 $\{0,1\}$ 为离散范畴，没有二者之间的非恒等态射。因此
$$
(\operatorname{Ran}_iF)(0)\cong A\times B.
$$
$1/i$ 只有 $(1,\operatorname{id})$，因为不存在 $1\to0$，所以
$$
(\operatorname{Ran}_iF)(1)\cong B.
$$
右 Kan 延拓给出的箭头 $A\times B\to B$ 是第二投影。

**答案 6.13.** 取 $K:\varnothing\to *$ 和空图形 $F:\varnothing\to\mathcal E$。右 Kan 延拓在 $*$ 处应为 $\lim_\varnothing F$，即终对象。若 $\mathcal E$ 没有终对象，则该右 Kan 延拓不存在。

## 第七章

**答案 7.1.** 单位把字 $w$ 看成长为一的“字的字”，乘法拼接；左右单位说明拼接单层括号不改变字，结合律说明去括号顺序不影响最终字。

**答案 7.2.** 单子 $T(S)=U\mathbb Z[S]$。$T$-代数 $T(A)\to A$ 等价于给集合 $A$ 一个阿贝尔群结构，使自由阿贝尔群上的线性组合在 $A$ 中求值；代数同态正是群同态。

**答案 7.3.** $\eta_X(x)=\{x\}$。乘法 $\mu_X:\mathcal P\mathcal P(X)\to\mathcal P(X)$ 为并集：$\mathcal A\mapsto\bigcup_{A\in\mathcal A}A$。

**答案 7.4.** 设 $f:X\to TY,g:Y\to TZ,h:Z\to TW$。两种复合分别化为
$$
\mu_W\mu_{TW}T^2h\,Tg\,f
$$
和
$$
\mu_WT\mu_WT^2h\,Tg\,f,
$$
由 $\mu\circ\mu T=\mu\circ T\mu$ 相等。

**答案 7.5.** 余单子为 $(G,\epsilon,\delta)$，其中 $\epsilon:G\to\operatorname{id}$、$\delta:G\to G^2$，满足对偶单位律和结合律。余代数为 $a:A\to G A$，满足 $\epsilon_Aa=\operatorname{id}$ 与 $\delta_Aa=G(a)a$。

**答案 7.6.** $J(\operatorname{id}_X)=\eta_X$，即 Kleisli 恒等。若 $X\xrightarrow{f}Y\xrightarrow{g}Z$，则
$$
Jg\circ_K Jf
=\mu_ZT(\eta_Zg)\eta_Yf
=\mu_ZT\eta_Z\,Tg\,\eta_Yf
=\eta_Zgf
=J(gf),
$$
其中用 $\eta$ 自然性和单位律。

**答案 7.7.** 余单位在对象 $Y$ 处是 Kleisli 态射 $TY\to Y$，即 $\mathcal C$ 中的 $\operatorname{id}_{TY}:TY\to TY$。在 $Y=JX=X$ 时，右伴随 $G_T$ 把它送为
$$
T^2X\xrightarrow{T\operatorname{id}}T^2X\xrightarrow{\mu_X}TX,
$$
即单子乘法 $\mu_X$。

**答案 7.8.** 给 $k:X\to A$，延拓必须满足 $h\eta_X=k$。候选为 $h=aTk$。它是代数同态且满足 $h\eta_X=k$。若 $h$ 是任意代数同态，则
$$
h=h\mu_XT\eta_X=aTh\,T\eta_X=aT(h\eta_X)=aTk,
$$
故唯一。

**答案 7.9.** 列表单子的自由代数 $F^TX$ 是 $X$ 上有限字集合，代数结构为拼接。Eilenberg-Moore 代数是幺半群。Kleisli 态射 $X\to Y$ 是函数 $X\to T(Y)$，即给每个 $x$ 指定一个 $Y$ 中有限字。

**答案 7.10.** 函子把 $f:X\to TY$ 送到 $\mu_YTf:TX\to TY$。对 $g:Y\to TZ$，
$$
\mu_ZT(\mu_ZTg f)
=\mu_ZT\mu_ZT^2gTf
=\mu_Z\mu_{TZ}T^2gTf
=\mu_ZTg\,\mu_YTf,
$$
其中用结合律和 $\mu$ 的自然性。右端正是两个代数同态的复合。

**答案 7.11.** 恒等单子的 Kleisli Hom 为
$$
\mathcal C_{\operatorname{id}}(X,Y)=\mathcal C(X,Y),
$$
复合和恒等都为原来的复合和恒等。Eilenberg-Moore 代数为 $(A,a:A\to A)$，单位律强制 $a=\operatorname{id}_A$，代数同态条件自动化为普通态射条件。

**答案 7.12.** 对反射 $L:\mathcal C\rightleftarrows\mathcal A:I$，诱导单子为 $T=IL$。单位为反射单位 $\eta_X:X\to ILX$。乘法为
$$
T^2=ILIL\xrightarrow{I\varepsilon L}IL=T,
$$
其中 $\varepsilon:LI\to\operatorname{id}_{\mathcal A}$ 是余单位。

**答案 7.13.** 阿贝尔化反射的单子把群 $G$ 送到
$$
G_{\operatorname{ab}}=G/[G,G],
$$
再视为群。单位为商映射 $G\to G/[G,G]$。由于阿贝尔群再次阿贝尔化不变，乘法 $(G_{\operatorname{ab}})_{\operatorname{ab}}\to G_{\operatorname{ab}}$ 是同构。

**答案 7.14.** 任意单子 $T$ 都由 Kleisli 伴随 $J\dashv G_T$ 诱导，也由 Eilenberg-Moore 伴随 $F^T\dashv U^T$ 诱导。二者中间范畴通常不同：前者的对象与 $\mathcal C$ 相同，态射为 $X\to TY$；后者对象为所有 $T$-代数。因此“由伴随产生单子”不确定唯一伴随。

## 第八章

**答案 8.1.** $(\mathbf{Set},\sqcup,\varnothing)$ 是幺半范畴，结合和单位由集合余积同构给出。它不是笛卡尔幺半结构，因为单位应为终对象而非始对象。

**答案 8.2.** 单对象范畴的幺半结构等价于该 Hom 幺半群上再给一个与复合相容的幺半运算；可视为严格幺半范畴的一对象情形，即两个幺半结构满足交换律型相容。

**答案 8.3.** 若 $A$ 有乘法 $m$ 和单位 $u$，则 $FA$ 的乘法为
$$
FA\otimes FA\xrightarrow{\phi}F(A\otimes A)\xrightarrow{F m}FA
$$
单位为 $\mathbb1\to F\mathbb1\xrightarrow{F u}FA$。相干图由 $F$ 的幺半相干性和 $A$ 的代数公理给出。

**答案 8.4.** 余代数对象为 $C$ 加态射 $\Delta:C\to C\otimes C$ 与 $\epsilon:C\to\mathbb1$，满足对偶的余结合律和余单位律。

**答案 8.5.** 辫子满足两个六边形，表达 $\beta_{X,Y\otimes Z}$ 与 $\beta_{X,Y},\beta_{X,Z}$ 的相容，以及 $\beta_{X\otimes Y,Z}$ 与 $\beta_{X,Z},\beta_{Y,Z}$ 的相容。对称还要求 $\beta_{Y,X}\beta_{X,Y}=\operatorname{id}$。

**答案 8.6.** 函子复合给出严格幺半范畴。余代数对象是自函子 $G$ 加自然变换
$$
\delta:G\Rightarrow G^2,\qquad \epsilon:G\Rightarrow\operatorname{id}_{\mathcal C}
$$
满足余结合律和余单位律。这正是余单子的定义。

**答案 8.7.** 给定 $(A,m: A\times A\to A,u:1\to A)$，令 $e=u(*)$，二元运算为 $a\cdot b=m(a,b)$。代数对象结合律和单位律逐元素展开为
$$
(a\cdot b)\cdot c=a\cdot(b\cdot c),\qquad e\cdot a=a=a\cdot e.
$$
反向构造直接给出乘法和单位映射。保持代数结构的态射正是幺半群同态。

**答案 8.8.** 单位映射 $\varnothing\to A$ 唯一。左单位律要求 $m:A\sqcup A\to A$ 在右副本上为恒等，右单位律要求它在左副本上为恒等，因此 $m$ 只能是折叠映射 $\nabla=[\operatorname{id},\operatorname{id}]$。该乘法与余积交换同构复合后仍是 $\nabla$，故它是交换代数对象。

**答案 8.9.** 结合律左边为
$$
(FA\otimes FA)\otimes FA\to F(A\otimes A)\otimes FA\to F((A\otimes A)\otimes A)\to F(A\otimes A)\to FA.
$$
右边为
$$
(FA\otimes FA)\otimes FA\to FA\otimes(FA\otimes FA)\to FA\otimes F(A\otimes A)\to F(A\otimes(A\otimes A))\to F(A\otimes A)\to FA.
$$
松幺半函子的结合相干图把前半段识别，$A$ 的结合律把后半段识别，因此两边相等。

**答案 8.10.** 设 $K_2$ 为常值二元集函子，$S(X)=X\times X$。若复合幺半结构有辫子，则有自然同构 $K_2S\cong SK_2$。但 $K_2S$ 是常值二元集函子，而 $SK_2$ 是常值四元集函子。二元集与四元集不同构，矛盾。

## 第九章

**答案 9.1.** 评价映射 $\operatorname{ev}:Z^X\times X\to Z$ 为 $(f,x)\mapsto f(x)$。给 $g:Y\times X\to Z$，其 curry 化为 $\bar g(y)(x)=g(y,x)$，直接代入得 $\operatorname{ev}(\bar g(y),x)=g(y,x)$。

**答案 9.2.** 闭幺半范畴给出伴随
$$
-\otimes X\dashv [X,-].
$$
任意左伴随保持所有存在的余极限。初对象是空图形的余极限，所以
$$
0\otimes X\simeq (-\otimes X)(0)
\simeq \operatorname{colim}_{\varnothing}(-\otimes X)
$$
仍是初对象。等价地，对任意 $Y$，
$$
\mathcal C(0\otimes X,Y)\cong\mathcal C(0,[X,Y])
$$
为单点集，因此 $0\otimes X$ 满足初对象的泛性质。

**答案 9.3.** 若 $X$ 有限维，线性映射空间 $\operatorname{Hom}_k(X,Z)$ 自然同构于 $X^*\otimes Z$；同构依赖有限维性。

**答案 9.4.** 由 Day 公式：
$$
(ya\star yb)(c)=\int^{u,v}\mathcal C(u,a)\times\mathcal C(v,b)\times\mathcal C(c,u\otimes v)
\cong\mathcal C(c,a\otimes b)
$$
两次 co-Yoneda 给出同构。

**答案 9.5.** coend 公式为
$$
\coprod_{f:c\to c'}H(c',c)\rightrightarrows\coprod_cH(c,c)\to\int^cH(c,c).
$$

**答案 9.6.** 对态射 $h:Z\to Z'$，比较
$$
[\mathbb1,Z]\to[\mathbb1,Z']\qquad\text{与}\qquad Z\to Z'
$$
在任意 $Y$ 上表示的 Hom 映射。二者都对应复合
$$
\mathcal C(Y\otimes\mathbb1,Z)\xrightarrow{h\circ-}\mathcal C(Y\otimes\mathbb1,Z'),
$$
故命题 9.11 的同构对 $Z$ 自然。

**答案 9.7.** 在笛卡尔闭范畴中 $[X,Z]=Z^X$ 且 $\otimes=\times$。命题 9.12 给出
$$
Z^{X\times Y}\cong (Z^Y)^X.
$$
这正是指数律；其 Hom 集证明为
$$
\mathcal C(T,Z^{X\times Y})\cong\mathcal C(T\times X\times Y,Z)
\cong\mathcal C(T,(Z^Y)^X).
$$

**答案 9.8.** 若 $-\sqcup X$ 有右伴随，则它是左伴随，因此保持初对象。但初对象 $\varnothing$ 被送到 $X$。当 $X\ne\varnothing$ 时，$X$ 不是初对象，矛盾。

**答案 9.9.** 在对象 $c$ 处，
$$
(P\star y\mathbb1)(c)=\int^{a,b}P(a)\times\mathcal C(b,\mathbb1)\times\mathcal C(c,a\otimes b).
$$
先对 $b$ 用 co-Yoneda，得
$$
\int^a P(a)\times\mathcal C(c,a\otimes\mathbb1).
$$
由右单位约束化为 $\int^aP(a)\times\mathcal C(c,a)$，再用 co-Yoneda 得 $P(c)$。

**答案 9.10.** 在偏序范畴中，有限积是交 $\wedge$。笛卡尔闭要求对每个 $x$，函子 $(-)\wedge x$ 有右伴随 $(-)^x$，即
$$
y\wedge x\le z\Longleftrightarrow y\le z^x.
$$
这正是偏序意义下右伴随的定义。反过来，若这些右伴随存在，则指数对象由 $z^x$ 给出，故范畴笛卡尔闭。

## 第十章

**答案 10.1.** $\mathbf{Ab}$-富范畴的复合是阿贝尔群同态
$$
\mathcal A(B,C)\otimes\mathcal A(A,B)\to\mathcal A(A,C),
$$
等价于双线性映射 $\mathcal A(B,C)\times\mathcal A(A,B)\to\mathcal A(A,C)$。

**答案 10.2.** 令 $\mathbf{2}$ 为幺半偏序 $0<1$ 且张量为 $\wedge$，单位为 $1$。给预序集 $(P,\le)$ 构造 $\mathbf{2}$-富范畴：对象为 $P$，并置
$$
\mathcal P(x,y)=
\begin{cases}
1,&x\le y,\\
0,&x\nleq y.
\end{cases}
$$
单位公理要求 $1\le\mathcal P(x,x)$，等价于 $x\le x$。复合公理要求
$$
\mathcal P(y,z)\wedge\mathcal P(x,y)\le\mathcal P(x,z),
$$
这正是从 $x\le y$ 和 $y\le z$ 推出 $x\le z$。反向地，任一 $\mathbf{2}$-富范畴定义关系 $x\le y$ 当且仅当 $\mathcal P(x,y)=1$，单位和复合公理给出自反性与传递性；若再按双向可达关系取商，则得到偏序集。

**答案 10.3.** $\mathbf{Cat}$-富范畴的数据为：对象；对每对对象 $A,B$ 给 Hom 范畴 $\mathcal K(A,B)$；复合函子
$$
\mathcal K(B,C)\times\mathcal K(A,B)\to\mathcal K(A,C);
$$
以及单位函子 $1\to\mathcal K(A,A)$。Hom 范畴的对象就是严格 2-范畴中的 $1$-态射，Hom 范畴的态射就是 $2$-态射。富范畴的结合律和单位律是严格相等，因此给出严格 2-范畴；反过来，严格 2-范畴的 Hom 范畴和水平复合正给出一个 $\mathbf{Cat}$-富范畴。

**答案 10.4.** 取 $\mathcal V=\mathbf{Set}$、权重常值单点集，则
$$
\mathcal A(A,\{1,D\})\cong\operatorname{Nat}(1,\mathcal A(A,D-))
$$
即从 $A$ 到 $D$ 的锥集合，故恢复普通极限。

**答案 10.5.** 若 $\mathcal V$ 闭且有所需 end，富自然变换对象通常写作
$$
\operatorname{Nat}_{\mathcal V}(F,G)=\int_A [F A,G A].
$$
这个 end 是等化子，强制对每个 $u:A\to B$ 的自然性方块相容：从 $FA$ 到 $GB$ 的两条路径分别经 $Fu$ 后再取 $G B$，或先到 $GA$ 再经 $Gu$。因此它不是逐点内部 Hom 的简单积，而是把所有对象处的内部 Hom 按自然性条件截出的子对象。

**答案 10.6.** 若 $\mathcal A$ 是预加性范畴，$F:\mathcal A^{\operatorname{op}}\to\mathbf{Ab}$ 是加性函子，则
$$
\operatorname{Nat}_{\mathbf{Ab}}(\mathcal A(-,A),F)\cong F(A)
$$
是阿贝尔群同构。自然变换由 $\operatorname{id}_A$ 的像决定，任意 $x\in F(A)$ 给出 $\alpha_B(f:B\to A)=F(f)(x)$。

**答案 10.7.** 证明需要把态射 $X\to[\mathcal A(B,A),F(B)]$ 与评价型态射 $X\otimes\mathcal A(B,A)\to F(B)$ 互相转换；这正是内部 Hom 与张量的伴随，即闭结构。

**答案 10.8.** 若富范畴只有一个对象 $*$，唯一 Hom 对象 $M=\mathcal A(*,*)$ 带复合 $M\otimes M\to M$ 和单位 $\mathbb1\to M$。富范畴公理正是结合律和单位律。反向由任意代数对象 $M$ 构造一对象富范畴。

**答案 10.9.** 对 $A,B\in\mathcal A$，定理 10.10 应用于富预层 $\mathcal A(-,B)$ 给出
$$
\operatorname{Fun}_{\mathcal V}(\mathcal A^{\operatorname{op}},\mathcal V)(\mathcal A(-,A),\mathcal A(-,B))
\cong \mathcal A(A,B).
$$
这说明 Yoneda 嵌入在 Hom 对象上给出同构，故富满忠实。

**答案 10.10.** 在 $([0,\infty],\ge,+,0)$ 中，复合态射
$$
d(y,z)+d(x,y)\to d(x,z)
$$
存在当且仅当 $d(y,z)+d(x,y)\ge d(x,z)$，即三角不等式。单位态射 $0\to d(x,x)$ 存在当且仅当 $0\ge d(x,x)$，即 $d(x,x)=0$。富函子的 Hom 态射存在当且仅当 $d_Y(Fx,Fx')\le d_X(x,x')$。

**答案 10.11.** 加权极限定义给出
$$
\mathcal A(B,\{W,D\})
\cong
\operatorname{Fun}_{\mathcal V}(\mathcal J,\mathcal V)(W,\mathcal A(B,A)).
$$
当 $\mathcal J$ 为一对象单位富范畴时，右边等于 $[W,\mathcal A(B,A)]$。这正是余张量 $A^W$ 的泛性质，所以 $\{W,D\}\cong A^W$。

**答案 10.12.** 富函子范畴的 Hom 对象通常是
$$
\int_A \mathcal B(FA,GA)
$$
或闭情形下的相应内部 Hom end。end 的等化子公式需要对所有对象取积。若 $\mathcal V$ 没有所需无限积，该 end 可能没有对象承载，因此富函子范畴 Hom 对象可能不存在。

## 第十一章

**答案 11.1.** end 的等化子中，源积按态射 $f:C\to C'$ 的一份取 $H(C',C)$。第一箭头使用 $H$ 在第一变量的反变性：
$$
H(C',C)\xrightarrow{H(f,C)}H(C,C),
$$
并落入目标积的 $C$ 分量。第二箭头使用 $H$ 在第二变量的协变性：
$$
H(C',C)\xrightarrow{H(C',f)}H(C',C'),
$$
并落入 $C'$ 分量。等化子条件正要求 end 的一族分量 $\omega_C:E\to H(C,C)$ 对每个 $f$ 满足
$$
H(f,C)\omega_{C'}=H(C',f)\omega_C,
$$
即 dinaturality。

**答案 11.2.** 若 $\mathcal C$ 离散，除恒等态射外没有态射。end 等化子中的所有条件都来自恒等态射，而恒等态射给出的两条映射相同，所以没有额外限制：
$$
\int_C H(C,C)\cong\prod_C H(C,C).
$$
coend 对偶地由余等化子给出；恒等态射产生的两条箭头相同，不再施加非平凡商关系，因此
$$
\int^C H(C,C)\cong\coprod_C H(C,C).
$$

**答案 11.3.** 设 $\alpha:F\Rightarrow G$ 与 $\beta:G\Rightarrow H$。end 描述给出族 $\alpha_C:FC\to GC$、$\beta_C:GC\to HC$，并分别满足
$$
Gf\,\alpha_C=\alpha_D\,Ff,\qquad
Hf\,\beta_C=\beta_D\,Gf
$$
对每个 $f:C\to D$ 成立。逐点复合定义为 $(\beta\alpha)_C=\beta_C\alpha_C$。检查自然性：
$$
Hf(\beta_C\alpha_C)=(Hf\beta_C)\alpha_C
=(\beta_DGf)\alpha_C
=\beta_D(Gf\alpha_C)
=\beta_D(\alpha_DFf)
=(\beta_D\alpha_D)Ff.
$$
故逐点复合仍满足 end 的等化子条件。

**答案 11.4.** co-Yoneda 中 $[C,x,f]$ 映到 $P(f)(x)$。逆把 $a$ 送到 $[A,a,\operatorname{id}]$。关系 $(C,P(u)y,f)\sim(D,y,uf)$ 下二者同映到 $P(f)P(u)y=P(uf)y$，故良定义。

**答案 11.5.** 取 $P$，co-Yoneda 给
$$
P\cong\int^C P(C)\times yC.
$$
右边是按元素 $x\in P(C)$ 对可表 $yC$ 作的余商，即元素范畴上的可表预层余极限。

**答案 11.6.** 给一个集合 $E$ 上的余 dinatural 族 $q_C:H(C,C)\to E$，等价于给出从余积 $\coprod_C H(C,C)$ 到 $E$ 的映射，且对每个 $f:C\to C'$ 与 $x\in H(C',C)$ 有
$$
q_C(H(f,C)x)=q_{C'}(H(C',f)x).
$$
这正是该映射在命题 11.8 的等价关系上常值。故它唯一分解过商集合，商集合满足 coend 的始性。

**答案 11.7.** 因为 $\mathcal C$ 离散，end 为
$$
\prod_{C\in\mathcal C}\{0,1\},
$$
coend 为
$$
\coprod_{C\in\mathcal C}\{0,1\}.
$$
当对象集无限时，二者都是无限集合，因此不属于 $\mathbf{FinSet}$。

**答案 11.8.** 由命题 11.6，
$$
\int_C\mathbf{Set}(\mathcal C(C,A),P(C))
\cong
\operatorname{Nat}(\mathcal C(-,A),P).
$$
反变 Yoneda 引理把右边自然同构于 $P(A)$。

**答案 11.9.** 映射 $\Phi$ 把代表 $(C,f:C\to A,x\in F(C))$ 送到 $F(f)(x)$。逆映射把 $a\in F(A)$ 送到代表
$$
[A,\operatorname{id}_A,a].
$$
由 coend 关系，任意代表 $[C,f,x]$ 等于 $[A,\operatorname{id}_A,F(f)(x)]$，故两者互逆。

**答案 11.10.** 一对象范畴中每个态射是 $m\in M$。余等化子源的一份 $H(*,*)$ 可记为 $S$。第一箭头由第一变量作用给出 $x\mapsto x\cdot m$，第二箭头由第二变量作用给出 $x\mapsto m\cdot x$，所以 coend 商关系为 $x\cdot m\sim m\cdot x$。

**答案 11.11.** 离散有限情形中没有非恒等态射关系，故
$$
\int^{c\in C}\int^{d\in D}H(c,d,c,d)
\cong
\coprod_{c\in C}\coprod_{d\in D}H(c,d,c,d).
$$
有限余积可按索引重排，得到
$$
\coprod_{(c,d)\in C\times D}H(c,d,c,d)
\cong
\int^{(c,d)}H(c,d,c,d).
$$

## 第十二章

**答案 12.1.** 要证有限集合 $A$ 为 $\omega$-紧，即证明
$$
\mathbf{Set}(A,\operatorname{colim}_jX_j)\to
\operatorname{colim}_j\mathbf{Set}(A,X_j)
$$
为双射。给定 $f:A\to\operatorname{colim}X_j$，由于 $A$ 有限，所有 $f(a)$ 可分别由有限多个阶段 $X_{j_a}$ 中元素表示；滤过性给出共同上界 $k$，于是 $f$ 由某个 $A\to X_k$ 表示，得满射。若两个 $A\to X_i,X_j$ 在余极限中相同，则对每个 $a\in A$，它们在某个后续阶段相等；有限多个相等条件再由滤过性合并到一个共同后续阶段，得单射。

**答案 12.2.** 对无限集合 $A$，恒等映射 $A\to\operatorname{colim}_{B\subset A,\ B finite}B=A$ 不经过任一有限阶段，故 $A$ 不是 $\omega$-紧。

**答案 12.3.** 对预层 $P:\mathcal C^{op}\to\mathbf{Set}$，元素范畴 $\int P$ 的对象为 $(C,x\in P(C))$。有典范图形
$$
\int P\to\widehat{\mathcal C},\qquad (C,x)\mapsto yC.
$$
其余极限到 $P$ 的态射由每个元素 $x\in P(C)$ 对应的 Yoneda 态射 $yC\to P$ 给出。对任意预层 $Q$，
$$
\widehat{\mathcal C}(P,Q)\cong
\lim_{(C,x)\in\int P}\widehat{\mathcal C}(yC,Q)
\cong
\lim_{(C,x)\in\int P}Q(C),
$$
这正是从所有元素 $x$ 兼容地给出 $Q$ 中像的集合。因此该余极限满足表示 $P$ 的泛性质。

**答案 12.4.** 局部有限可表现范畴是余完备范畴 $\mathcal C$，并且存在一小族有限可表现对象，使每个对象都是这些对象的滤过余极限。这里“有限可表现”指 $\mathcal C(A,-)$ 保持滤过余极限。群范畴 $\mathbf{Grp}$ 的有限表现群由有限生成元和有限关系给出，例如
$$
\langle x\mid x^n=1\rangle.
$$
任意群可写成其有限生成子群和有限关系近似的滤过余极限，因此 $\mathbf{Grp}$ 是局部有限可表现范畴的典型例子。

**答案 12.5.** 生成族 $\mathcal G$ 的作用是检测态射：若 $f,g:X\rightrightarrows Y$ 在所有 $\mathcal C(G,-)$ 上相同，则 $f=g$。强生成还可检测同构。紧生成则包含两个额外条件：$\mathcal G$ 中对象紧，即 $\mathcal C(G,-)$ 保持相应滤过余极限；并且每个对象可由这些紧对象通过滤过余极限生成。因此“生成”是检测性质，“紧生成”同时控制构造方式和 Hom 与滤过余极限的交换。

**答案 12.6.** 由 Yoneda，
$$
\widehat{\mathcal C}(yC,P)\cong P(C).
$$
滤过余极限逐点计算，因此
$$
\widehat{\mathcal C}(yC,\operatorname{colim}_jP_j)
\cong(\operatorname{colim}_jP_j)(C)
\cong\operatorname{colim}_jP_j(C)
\cong\operatorname{colim}_j\widehat{\mathcal C}(yC,P_j).
$$

**答案 12.7.** 预层密度定理给出
$$
P\cong\operatorname{colim}_{(C,x)\in\int P}yC.
$$
因此任意预层都由可表预层经小余极限生成；若两个态射在所有可表预层上相同，则由该余极限表达式推出它们处处相同。

**答案 12.8.** 强生成子不仅检测平行态射是否相等，还检测态射是否为同构：$f$ 是同构当且仅当所有 $\mathcal C(G,f)$ 都是双射。这把大范畴中的同构问题化为小生成对象上的集合映射问题。

**答案 12.9.** 外部输入定理 12.16 的假设不是单纯“保持小余极限”，还要求可达性。证明右伴随时，对每个 $d\in\mathcal D$ 需要表示函子
$$
c\mapsto\mathcal D(Fc,d).
$$
可达性保证该函子由一小部分紧对象上的数据控制，从而满足解集条件并可由某个对象表示。若 $F$ 只保持小余极限但不可达，这个 Hom 函子可能需要真正大的数据来控制，表示对象候选不受集合大小约束；因此伴随函子定理不能推出右伴随存在。

**答案 12.10.** 有限个有限子集 $T_1,\dots,T_n\subset S$ 被有限并集 $T_1\cup\cdots\cup T_n$ 支配。空有限族由空集支配。由于偏序范畴中平行箭头至多一条，等化平行箭头条件自动成立。因此 $\operatorname{Fin}(S)$ 滤过。

**答案 12.11.** 写 $X\cong\operatorname{colim}_iA_i$。因 $F,G$ 保持该滤过余极限，
$$
FX\cong\operatorname{colim}_iFA_i,\qquad GX\cong\operatorname{colim}_iGA_i.
$$
自然性给出图形间自然变换 $(FA_i)\to(GA_i)$，其余极限态射正是 $\eta_X$。若每个 $\eta_{A_i}$ 为同构，则该图形同构的余极限仍为同构。

**答案 12.12.** 可数无限集合 $A$ 在 $\mathbf{Set}$ 中不是 $\omega$-紧，因为它不是有限集合；但 $|A|<\omega_1$，由例子 12.5 它是 $\omega_1$-紧对象。

## 第十三章

**答案 13.1.** 零态射 $X\to Y$ 是复合 $X\to0\to Y$。由于 $0$ 终，$X\to0$ 唯一；由于 $0$ 始，$0\to Y$ 唯一，故复合唯一。

**答案 13.2.** 在 $\mathbf{Ab}$ 中，核为子群
$$
\ker f=\{a\in A\mid f(a)=0\}
$$
连同包含映射 $\ker f\hookrightarrow A$。若 $u:X\to A$ 满足 $fu=0$，则 $u$ 的像落在该子群中，并唯一分解经 $\ker f$，所以满足核的泛性质。余核为商群
$$
\operatorname{coker}f=B/\operatorname{im}f
$$
及商映射 $B\to B/\operatorname{im}f$。任意 $v:B\to Y$ 若 $vf=0$，则 $v$ 在 $\operatorname{im}f$ 上为零，因而唯一分解过该商群。

**答案 13.3.** biproduct 带 $i_A,i_B,p_A,p_B$，满足 $p_Ai_A=1$、$p_Bi_B=1$、$i_Ap_A+i_Bp_B=1$。这些等式给出积和余积泛性质。

**答案 13.4.** 有限生成自由 $R$-模的直和仍有限生成自由，零模也是有限生成自由，Hom 集有阿贝尔群结构，故该范畴是加性的。它通常不是阿贝尔的原因是核和余核不封闭在有限生成自由模中。例如取 $R=k[\epsilon]/(\epsilon^2)$，考虑有限生成自由 $R$-模中的自同态
$$
R\xrightarrow{\epsilon}R.
$$
其核为理想 $(\epsilon)$。这是非零有限生成 $R$-模，但不是自由 $R$-模：在 $(\epsilon)$ 上乘以 $\epsilon$ 恒为零，而非零自由 $R$-模总含有不被 $\epsilon$ 零化的元素。故该态射在有限生成自由 $R$-模范畴中没有核。阿贝尔范畴要求所有态射有核、余核并满足 coimage-image 条件，因此这里通常失败。

**答案 13.5.** AB3 指存在任意小余积。AB4 指小余积正合，也就是说一族短正合列逐项取余积后仍短正合。AB5 指滤过余极限正合：滤过图形的短正合列取余极限后仍正合。Grothendieck 范畴通常定义为 AB5 的阿贝尔范畴且有生成元；生成元保证对象和态射可由一小部分 Hom 数据检测，AB5 保证 sheaf、模和同调构造中的滤过极限行为良好。

**答案 13.6.** 在 $\mathbf{Ab}$ 中，
$$
\operatorname{coim}(f)=A/\ker(f),\qquad
\operatorname{im}(f)=f(A)\subseteq B.
$$
第一同构定理给出 $A/\ker(f)\cong f(A)$，这正是 coimage 到 image 的典范同构。

**答案 13.7.** 若 $k:K\to A$ 是 $f:A\to B$ 的核且 $ku=kv$，则 $k(u-v)=0$。核态射本身作为等化子是单态射；等价地，等化子的泛性质直接给 $u=v$。

**答案 13.8.** 短正合列由核和余核刻画。正合函子保持核、余核以及有限 biproduct 中的零对象，因此把 $0\to A\to B\to C\to0$ 送到仍在中间项正合、左端单且右端满的短正合列。

**答案 13.9.** 若 $f,g:M\rightrightarrows N$ 不同，取 $m\in M$ 使 $f(m)\ne g(m)$。态射 $R\to M$ 由 $1\mapsto m$ 决定，于是 $f$ 与 $g$ 预复合该态射后不同。因此 $R$ 检测平行态射。

**答案 13.10.** 设 $\mathcal A$ 为 Grothendieck 范畴且 $G$ 为生成元。Gabriel-Popescu 定理把 $\mathcal A$ 表示为某个模范畴 $\operatorname{Mod}_R$ 的正合反射局部化，其中 $R=\operatorname{End}(G)$ 的合适对偶环。模范畴提供“自由生成”的代数模型；局部化函子把由生成元表示后应当消失或应当被识别的对象商掉。因局部化正合，$\mathcal A$ 的阿贝尔和滤过余极限结构可从模范畴中控制。

**答案 13.11.** 若 $e:E\to X$ 等化 $f,g:X\rightrightarrows Y$，且 $eu=ev$，则由等化子的泛性质，对态射 $eu$ 有唯一提升到 $E$；$u$ 和 $v$ 都是这样的提升，所以 $u=v$。余等化子为满态射是对偶论证。

**答案 13.12.** 群同态 $f:A\to B$ 单射当且仅当 $\ker f=0$。它满射当且仅当 $B/\operatorname{im}f=0$，而后者正是 $\operatorname{coker}f=0$。这验证了命题 13.18 在 $\mathbf{Ab}$ 中的具体形式。

**答案 13.13.** 正合函子保持核和余核，因此
$$
F(\operatorname{im}f)
=F(\ker(\operatorname{coker}f))
\cong
\ker(\operatorname{coker}(Ff))
=\operatorname{im}(Ff).
$$

**答案 13.14.** 若有限阿贝尔群范畴有可数余积 $\coprod_{n\in\mathbb N}\mathbb Z/2$，则包含到 $\mathbf{Ab}$ 后应满足同一余积泛性质，因而同构于可数直和 $\bigoplus_{\mathbb N}\mathbb Z/2$。该群无限，不属于有限阿贝尔群范畴，矛盾。

## 第十四章

**答案 14.1.** 覆盖 $\{U_i\subset U\}$ 生成的筛由所有开嵌入 $V\subset U$ 组成，其中 $V$ 的映射局部因子化经过某个 $U_i$；等价地 $V\subset\bigcup_iU_i$。

**答案 14.2.** 设 $X$ 连通，取常值预层 $F(U)=A$。若 $U=U_1\sqcup U_2$ 是两个非空开集的并，预层限制映射都是恒等。一个匹配族可在 $U_1$ 上取 $a_1$，在 $U_2$ 上取 $a_2$。由于交为空，相容条件没有强迫 $a_1=a_2$。但全局截面 $F(U)=A$ 只能限制为同一个元素，所以当 $a_1\ne a_2$ 时该匹配族不能由全局截面粘合。这说明常值预层通常不是 sheaf。

**答案 14.3.** subcanonical 意味每个 $yU$ 是 sheaf，因此 Yoneda 嵌入 $\mathcal C\to\widehat{\mathcal C}$ 的像落入 $\operatorname{Sh}(\mathcal C,J)$。

**答案 14.4.** 预层有限极限逐点计算。由命题 14.15，sheaf 的预层有限极限仍是 sheaf，因此 sheaf 范畴中的有限极限由预层有限极限创建。定理 14.9 的作用是保证 sheaf 子范畴还是左正合反射子范畴，从而 sheaf 化与有限极限相容。

**答案 14.5.** 定义 14.10 从外部给出 Grothendieck topos：它等价于某个小站点上的 sheaf 范畴。Giraud 定理则给出内在判别：一个范畴只要有有限极限、小余极限、余极限与拉回相容、等价关系有效并有小生成族等，就来自某个站点的 sheaf 范畴。两种描述等价，但逻辑方向不同：站点表示便于构造对象，Giraud 公理便于识别一个已给范畴是否为 topos。

**答案 14.6.** 对覆盖筛 $S$，预层 $F$ 的 separated 条件要求
$$
F(U)\to\operatorname{Nat}(S,F)
$$
为单射。因此若两个全局截面限制到覆盖上相同，它们本来就相同，这是粘合的唯一性。sheaf 条件要求同一映射为双射；除了单射，还要求每个匹配族都来自某个全局截面，即粘合存在性。因此 separated 是 sheaf 条件的唯一性部分。

**答案 14.7.** $F^+(U)$ 的元素可表示为一对 $(S,\alpha)$，其中 $S$ 是覆盖 $U$ 的筛，
$\alpha:S\to F$ 是自然变换。把 $S$ 看成由所有局部对象 $V\to U$ 组成的覆盖数据，
则 $\alpha$ 给每个 $V\to U$ 一个截面 $\alpha_V\in F(V)$，并要求这些截面对 $S$ 中态射自然相容，
这正是匹配族。两个表示 $(S,\alpha)$ 与 $(T,\beta)$ 等价，当存在共同覆盖细化
$R\subseteq S\cap T$，使 $\alpha|_R=\beta|_R$。因此 $F^+$ 是“匹配族按共同细化取商”。

**答案 14.8.** 对任意覆盖筛 $S$，sheaf 条件要求限制映射
$$
F(U)\to\operatorname{Nat}(S,F)
$$
为双射。Separated 条件只要求这同一映射为单射。双射蕴含单射，所以任意 sheaf 自动 separated。用截面语言说，若两个全局截面有相同的局部限制，则它们是同一个匹配族的两个粘合；sheaf 条件中的唯一性迫使二者相等。

**答案 14.9.** 几何态射 $f:\mathcal E\to\mathcal F$ 中，$f^*:\mathcal F\to\mathcal E$ 左伴随于 $f_*:\mathcal E\to\mathcal F$。要求 $f^*$ 左正合是为了让 inverse image 保持有限极限，从而保留逻辑连接词、交和终对象等几何结构。

**答案 14.10.** 平凡拓扑中每个对象只有最大筛作为覆盖。于是
$$
F^+(U)=\operatorname{Nat}(yU,F)\cong F(U)
$$
由 Yoneda 引理得到，所以 plus 构造不改变预层。

**答案 14.11.** 若 $F,G$ 是 sheaf，则预层积 $(F\times G)(U)=F(U)\times G(U)$。对覆盖筛 $S$，
$$
\operatorname{Nat}(S,F\times G)\cong\operatorname{Nat}(S,F)\times\operatorname{Nat}(S,G)
\cong F(U)\times G(U).
$$
故 $F\times G$ 是 sheaf，并给出 sheaf 范畴中的二元积。

**答案 14.12.** 若 $F$ 是 sheaf，则 $F$ 已在反射子范畴中。由命题 14.16，对任意 sheaf $G$，
$$
\operatorname{Sh}(aF,G)\cong\widehat{\mathcal C}(F,iG)\cong\operatorname{Sh}(F,G).
$$
由 Yoneda 引理，$aF\cong F$，单位 $F\to iaF$ 是同构。

**答案 14.13.** 若 $f^*,g^*$ 都保持有限极限，则复合 $f^*g^*$ 也保持有限极限，因为有限极限先被 $g^*$ 送为有限极限，再被 $f^*$ 保持。复合 inverse image 仍是左伴随，右伴随为对应 direct image 的反向复合。

## 第十五章

**答案 15.1.** 若 $\alpha:F\Rightarrow G$、$\beta:H\Rightarrow K$，横向复合在对象 $A$ 上为
$$
(\beta*\alpha)_A=\beta_{G A}\circ H(\alpha_A)=K(\alpha_A)\circ\beta_{F A}.
$$

**答案 15.2.** 设 $\alpha:F\Rightarrow G$、$\alpha':G\Rightarrow H$、$\beta:K\Rightarrow L$、$\beta':L\Rightarrow M$。交换律比较
$$
(\beta'\beta)*(\alpha'\alpha)
$$
与
$$
(\beta'*\alpha')(\beta*\alpha).
$$
在对象 $A$ 上展开，左边是沿大矩形先纵向复合再横向复合的外边，右边是先把矩形分成两个小方块再纵向复合。自然性给出中间小方块交换，普通态射复合的结合律允许去括号，因此两条复合路径相等。

**答案 15.3.** 环 $R,S,T$ 之间的水平态射可取双模。若 $M$ 是 $(R,S)$-双模、$N$ 是 $(S,T)$-双模，则复合为相对张量积
$$
M\otimes_SN.
$$
给三重双模 $M,N,P$ 时，$(M\otimes_SN)\otimes_TP$ 与 $M\otimes_S(N\otimes_TP)$ 由相对张量积的泛性质给出典范同构，但它们不是字面相同的对象。因此结合律只能由指定的结合约束同构表达，连同单位双模的左右单位约束，构成双范畴而非严格 2-范畴。

**答案 15.4.** 严格 2-函子 $F$ 满足
$$
F(gf)=F(g)F(f),\qquad F(1_x)=1_{Fx}
$$
为严格等式，并严格保持 $2$-态射的水平、纵向复合。伪函子则给出指定的可逆 $2$-胞腔
$$
F(g)F(f)\Rightarrow F(gf),\qquad 1_{Fx}\Rightarrow F(1_x),
$$
用这些胞腔替代严格等式。为了使三重和含单位的复合不依赖括号移动路径，还必须满足五边形和三角形相干公理。

**答案 15.5.** Bicategory coherence theorem 的作用是控制弱结合和弱单位带来的括号选择。它说明由结合约束和单位约束构成的典范比较图都交换，并且任意双范畴在适当双等价意义下可由更严格的模型表示。它推广了 Mac Lane 幺半相干性：幺半范畴是一对象双范畴，因此幺半范畴中的括号相干只是双范畴相干的一对象特例。

**答案 15.6.** 设 $\mathcal K$ 只有一个对象 $*$。Hom 范畴 $\mathcal K(*,*)$ 的对象是 $1$-态射，态射是 $2$-态射。水平复合给出函子
$$
\mathcal K(*,*)\times\mathcal K(*,*)\to\mathcal K(*,*),
$$
单位 $1$-态射给出幺半单位。严格 $2$-范畴公理给出张量严格结合和严格单位律，因此得到严格幺半范畴。

**答案 15.7.** 普通范畴等价给出函子 $F:\mathcal C\to\mathcal D$、拟逆 $G:\mathcal D\to\mathcal C$ 和自然同构 $GF\cong\operatorname{id}_{\mathcal C}$、$FG\cong\operatorname{id}_{\mathcal D}$。在 $\mathbf{Cat}$ 中自然同构就是可逆 $2$-态射，所以 $F$ 是 $2$-范畴意义下的等价。

**答案 15.8.** 若 $M$ 是 $(S,R)$-双模，$N$ 是 $(T,S)$-双模，$P$ 是 $(U,T)$-双模，则结合同构为
$$
(P\otimes_TN)\otimes_SM\cong P\otimes_T(N\otimes_SM),
$$
具体方向取决于双模复合的书写约定。它来自张量积的泛性质和平衡关系。

**答案 15.9.** 只给出 $F(g)F(f)\cong F(gf)$ 还不足以保证三重复合的两种比较一致。四个可复合 $1$-态射会产生五边形中的两条典范比较路径；相干五边形要求它们相等，从而使伪函子的复合不依赖括号移动路径。

## 第十六章

**答案 16.1.** 模型范畴公理要求弱等价类满足 $2$-out-of-$3$：对可复合态射
$$
X\xrightarrow{f}Y\xrightarrow{g}Z,
$$
若 $f,g,gf$ 中任意两个属于弱等价类，则第三个也属于弱等价类。这个条件保证把弱等价形式反演时复合行为稳定；例如若 $f$ 与 $gf$ 已可逆，则 $g=(gf)f^{-1}$ 在局部化中也可逆。

**答案 16.2.** 平凡纤维化是交集
$$
\operatorname{Fib}\cap W
$$
中的态射，即同时为纤维化和弱等价。平凡余纤维化是
$$
\operatorname{Cof}\cap W
$$
中的态射，即同时为余纤维化和弱等价。这里“平凡”不表示态射为同构，而表示它在同伦意义下为等价；提升和分解公理正以这两个交集为核心。

**答案 16.3.** 链复形的 quasi-isomorphism 是在所有同调对象上诱导同构的链映射。导出范畴的目标是只保留由同调能检测的同伦信息，因此必须把 quasi-isomorphism 反演。反演后，两个 quasi-isomorphic 的复形代表同一个导出对象；短正合列、解析分辨率和投射/内射分辨率给出的不同模型也因此变为等价。

**答案 16.4.** 模型范畴或 $\infty$-范畴的映射对象通常是空间或单纯集；同伦范畴的 Hom 集只取这些映射空间的连通分支：
$$
\operatorname{Ho}(\mathcal M)(X,Y)\simeq \pi_0\operatorname{Map}(QX,RY).
$$
因此它记录“态射到同伦为止”，但不记录路径之间的路径、更高同伦群，也不记录复合的高阶相干数据。两个不同高阶同伦理论可能有等价的同伦范畴，但映射空间不同。

**答案 16.5.** Kan fibration 是单纯集映射 $p:X\to Y$ 的提升性质：对所有 $0\le i\le n$ 的 horn 包含 $\Lambda_i^n\hookrightarrow\Delta^n$，任意相容方块都有填充。Quasi-category 是单纯集 $C$ 的内 horn 填充条件：只要求 $0<i<n$ 的 horn $\Lambda_i^n\to C$ 可填充。外 horn 对应可逆性或解方程条件；quasi-category 不要求外 horn，所以允许非可逆 $1$-态射，而 Kan 复形要求所有 horn，因而建模 $\infty$-群胚。

**答案 16.6.** 若弱等价就是同构，则局部化没有新增需要反演的态射；所有已被要求可逆的态射在原范畴中已经可逆。因此局部化泛性质由恒等函子 $\mathcal C\to\mathcal C$ 满足。

**答案 16.7.** 在离散模型结构中，平凡纤维化和平凡余纤维化都是同构。若方块中 $i$ 是同构，对角线可取上边复合 $i^{-1}$；若 $p$ 是同构，对角线可取 $p^{-1}$ 复合右边。交换性由原方块交换和同构逆的唯一性给出。

**答案 16.8.** 若 $X$ cofibrant，则 $\varnothing\to X$ 是余纤维化。左 Quillen 函子保持初对象和余纤维化，所以 $F\varnothing\to FX$ 是余纤维化，而 $F\varnothing\cong\varnothing$，故 $FX$ cofibrant。

**答案 16.9.** 相对范畴只指定哪些态射是弱等价；它没有余纤维化、纤维化、提升和分解数据。因此不能从中选择“初对象到 cofibrant 对象的平凡纤维化”这种 cofibrant replacement 结构。

**答案 16.10.** 同伦范畴等价只比较反演弱等价后的 $1$-范畴。Quillen 等价还要求模型结构相容，并诱导底层同伦理论或 $\infty$-范畴等价；它保留映射空间和导出函子层面的信息。

## 第十七章

**答案 17.1.** 保序映射 $[1]\to[2]$ 有六个：$(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)$。非退化边对应 $(0,1),(0,2),(1,2)$，其余为退化边。

**答案 17.2.** $\delta^i$ 是漏掉 $i$ 的严格递增函数，故保序；$\sigma^i$ 只把 $i,i+1$ 合并，其余保持顺序，故保序。

**答案 17.3.** Yoneda 给
$$
\mathbf{sSet}(\Delta^n,X)=\operatorname{Nat}(\Delta(-,[n]),X)\cong X([n])=X_n
$$
对 $X$ 和 $[n]$ 自然。

**答案 17.4.** $\Lambda_1^2$ 有三个顶点 $0,1,2$，非退化边为 $0\to1$ 和 $1\to2$，缺少面 $0\to2$ 所在边作为复合边，因此表示两条可复合边。

**答案 17.5.** $N(\mathcal C)_3$ 是函子 $[3]\to\mathcal C$，即对象 $X_0,\dots,X_3$ 和相容态射 $X_i\to X_j$。长边 $X_0\to X_2$、$X_1\to X_3$、$X_0\to X_3$ 由相邻边复合决定。

**答案 17.6.** 内 horn 编码复合存在性；外 horn 编码可逆性或解方程性质。Kan 复形要求所有 horn 填充，对应 $\infty$-群胚；quasi-category 只要求内 horn，因此允许非可逆态射。

**答案 17.7.** 若 $N(\mathcal C)$ 是 Kan 复形，则由命题 17.18，其同伦范畴中每条边都是同构。但 $hN(\mathcal C)\cong\mathcal C$，所以 $\mathcal C$ 中每个态射都可逆。若 $\mathcal C$ 有非可逆态射，矛盾。

**答案 17.8.** Kan-Quillen 模型结构的 fibrant objects 是 Kan 复形，用来建模 spaces；Joyal 模型结构的 fibrant objects 是 quasi-categories，用来建模 $\infty$-categories。

**答案 17.9.** $\infty$-群胚应只有可逆态射及高阶可逆同伦。命题 17.18 说明 Kan 复形作为 quasi-category 时，所有 $1$-态射在同伦范畴中可逆，符合 $\infty$-群胚直觉。

**答案 17.10.** 对任意 $m$，
$$
N([n])_m=\operatorname{Fun}([m],[n])=\Delta([m],[n])=(\Delta^n)_m.
$$
对 $\Delta$ 中态射的作用都是预复合，所以这些逐级双射组成单纯集同构。

**答案 17.11.** $\Lambda_2^3$ 缺少第 $2$ 个面，即漏掉顶点 $2$ 的面 $(0,1,3)$。给定 $X_0\to X_1\to X_2\to X_3$ 后，所有长边由复合确定：$X_0\to X_2$、$X_1\to X_3$、$X_0\to X_3$。缺失面 $(0,1,3)$ 的三角关系要求 $X_0\to X_3$ 等于 $(X_1\to X_3)(X_0\to X_1)$，这由结合律唯一成立。

**答案 17.12.** $0$-单纯形给出对象函数，$1$-单纯形给出态射函数。源、目标由 $d_1,d_0$ 保持确定；恒等由退化 $s_0$ 确定；复合由 $2$-单纯形的长边确定。因此 nerve 映射一旦在 $0$、$1$ 维给定并与这些结构相容，高维单纯形都由可复合态射串唯一决定。

**答案 17.13.** 外角填充表达求逆或解方程。若 $\mathcal C$ 是群胚，缺失的外边可用已有边的逆和复合构造；所有二维相容关系由范畴复合和逆律满足。高维 horn 的填充由相邻边串及其复合唯一决定。

**答案 17.14.** $\Delta^0=N(*)$，其中 $*$ 是群胚，故由命题 17.22 它是 Kan 复形。$\Delta^1=N([1])$，而 $[1]$ 中 $0\to1$ 不可逆，不是群胚，故 $\Delta^1$ 不是 Kan 复形。

## 第十八章

**答案 18.1.** $hC$ 只记录 $\pi_0\operatorname{Map}_C(x,y)$，不记录映射空间的高阶同伦群、路径之间的路径或相干复合。

**答案 18.2.** $hN(\mathcal C)\cong\mathcal C$。普通范畴 nerve 的复合严格唯一，1-单纯形的同伦关系不再额外识别不同态射。

**答案 18.3.** 对象 $t$ 终，当且仅当对所有 $x$，映射空间 $\operatorname{Map}_C(x,t)$ 可缩。

**答案 18.4.** 若 $R=\lim p$ 且 $G$ 是右伴随，则对任意 $x$：
$$
\operatorname{Map}_C(x,GR)\simeq\operatorname{Map}_D(Fx,R)
\simeq\lim_k\operatorname{Map}_D(Fx,p(k))
\simeq\lim_k\operatorname{Map}_C(x,Gp(k)).
$$
故 $GR$ 是 $Gp$ 的极限。

**答案 18.5.** HTT 中伴随可由 correspondence 或 adjunction data 定义。本章定义 18.13 是压缩表述；精确定义要求单位、余单位和全部高阶相干三角数据。

**答案 18.6.** 由 $\Delta^m\star\Delta^n\cong\Delta^{m+n+1}$，取 $m=0,n=1$ 得 $\Delta^0\star\Delta^1\cong\Delta^2$。顶点顺序为新锥顶在 $\Delta^1$ 两个顶点之前。

**答案 18.7.** 由 slice 泛性质，$C_{/x}$ 的对象是映射 $\Delta^0\star\Delta^0\to C$，其在右端顶点限制为 $x$。因 $\Delta^0\star\Delta^0\cong\Delta^1$，这就是所有指向 $x$ 的边 $y\to x$。

**答案 18.8.** 对象 $s$ 始，当且仅当对所有 $x$，映射空间 $\operatorname{Map}_C(s,x)$ 可缩。证明与终对象情形对偶，使用 cocone/slice 的对偶定义。

**答案 18.9.** 两个锥为 $(X,u:X\to A,v:X\to B)$ 与 $(X',u':X'\to A,v':X'\to B)$。一个 $1$-单纯形对应态射 $h:X\to X'$，满足 $u'h=u$ 且 $v'h=v$；这些等式自动与 $A\to C\leftarrow B$ 的相容条件兼容。

**答案 18.10.** 始对象对偶地满足对所有 $x$，$\operatorname{Map}_{N(\mathcal C)}(s,x)$ 可缩。该映射空间等价于离散集合 $\mathcal C(s,x)$，故可缩当且仅当 $\mathcal C(s,x)$ 为单点，即 $s$ 是 ordinary initial object。

**答案 18.11.** $0$-单纯形是映射 $\Delta^1\to C$，其第 $0$ 个顶点为 $x$、第 $1$ 个顶点为 $y$，即一条边 $x\to y$。$1$-单纯形是映射 $\Delta^2\to C$，其第 $0$ 个顶点为 $x$，而由顶点 $1,2$ 张成的边退化为 $y$ 的恒等边；它可理解为两条边 $x\to y$ 之间的一条右同伦。

**答案 18.12.** 左映射空间的 $n$-单纯形是函子 $[n+1]\to\mathcal A$，其中由 $0,\dots,n$ 张成的子范畴常值为 $x$，末顶点为 $y$。该函子唯一由边 $n\to n+1$ 的像 $f:x\to y$ 决定；反过来每个 $f$ 给出唯一这样的单纯形。面和退化不改变 $f$，所以得到离散单纯集 $\mathcal A(x,y)$。

**答案 18.13.** 普通范畴情形下，correspondence 是双函子
$$
H:\mathcal C^{op}\times\mathcal D\to\mathbf{Set}.
$$
左可表示意味着存在 $F:\mathcal C\to\mathcal D$，使得对所有 $x,y$ 有自然同构 $H(x,y)\cong\mathcal D(Fx,y)$。

**答案 18.14.** 若同一 $H$ 又右可表示，即 $H(x,y)\cong\mathcal C(x,Gy)$，则复合两个自然同构得到
$$
\mathcal D(Fx,y)\cong H(x,y)\cong\mathcal C(x,Gy),
$$
这正是第四章 Hom 自然同构定义的伴随 $F\dashv G$。

**答案 18.15.** coCartesian 边把源纤维中的对象沿基底箭头 $0\to1$ 推到目标纤维，所以给出 $C=M_0\to M_1=D$。Cartesian 边以目标纤维对象为终点，并把它沿 $0\to1$ 拉回到源纤维，所以给出 $D=M_1\to M_0=C$。

**答案 18.16.** 对 $x$，coCartesian 边 $x\to Fx$ 和以 $Fx$ 为终点的 Cartesian 边 $GFx\to Fx$ 覆盖同一基底箭头。Cartesian 泛性质要求 $x\to Fx$ 唯一分解为纤维内箭头 $x\to GFx$ 后接 $GFx\to Fx$；这个纤维内箭头就是单位 $\eta_x$。

**答案 18.17.** 第一条三角恒等式 $(\varepsilon f)\circ(f\eta)=\operatorname{id}_f$ 作用在左伴随 $f$ 上；第二条 $(g\varepsilon)\circ(\eta g)=\operatorname{id}_g$ 作用在右伴随 $g$ 上。

**答案 18.18.** 传到同伦范畴时，同伦类被当作相等，因此三角恒等式变成严格等式。但原 $\infty$-范畴中仍保留这些同伦本身以及同伦之间的更高相干；这些数据没有消失，只是 $hC$ 不记录它们。

**答案 18.19.** marked simplicial set 标记 $1$-单纯形，常用于记录等价边或 Cartesian edges。scaled simplicial set 标记 $2$-单纯形，常用于记录哪些 $2$-态射应视为 thin 或相干等式。

**答案 18.20.** 严格 $2$-函子把 $+$ 送到 $\mathcal C$，把 $-$ 送到 $\mathcal D$；把 $f:+\to-$ 送到 $F:\mathcal C\to\mathcal D$，把 $g:-\to+$ 送到 $G:\mathcal D\to\mathcal C$。

**答案 18.21.** 只有 $F,G,\eta,\varepsilon$ 还不足以表达伴随；还必须要求复合
$$
F\xrightarrow{F\eta}FGF\xrightarrow{\varepsilon F}F
$$
和
$$
G\xrightarrow{\eta G}GFG\xrightarrow{G\varepsilon}G
$$
等于恒等。高阶口径中这些等式提升为指定相干 $2$-维数据。

**答案 18.22.** correspondence 定义强调 Hom 或 mapping space 的表示性等价；walking adjunction 定义强调由单位、余单位和三角相干生成的代数型结构。二者等价，但突出的是伴随的不同面向。

**答案 18.23.** 若 $f:x\to y$ 是普通范畴 $\mathcal C$ 中的同构，取逆 $g:y\to x$。在 $hN(\mathcal C)\cong\mathcal C$ 中，$[g][f]=[\operatorname{id}_x]$ 且 $[f][g]=[\operatorname{id}_y]$。因此 $f$ 在 $hN(\mathcal C)$ 中是同构，按定义是 $N(\mathcal C)$ 中的等价边。

**答案 18.24.** 同伦范畴的 Hom 集定义为边按 $2$-单纯形生成的同伦关系取商。映射空间 $\operatorname{Map}_C(x,y)$ 的连通分支正是这些同伦类，因此
$$
hC(x,y)\cong\pi_0\operatorname{Map}_C(x,y).
$$
更高同伦群和高阶路径不进入 Hom 集。

**答案 18.25.** 若 $C$ 是 Kan 复形，则任意边都有外 horn 填充给出的左右逆，故在 $hC$ 中可逆。于是作为 $\infty$-范畴时，它的所有 $1$-态射都是等价，没有非可逆 $1$-态射。

## 第十九章

**答案 19.1.** 普通 fibration 中，覆盖 $\alpha:b\to p(e)$ 的箭头 $\tilde\alpha:e'\to e$ Cartesian，若任意 $g:x\to e$ 及分解 $p(g)=\alpha\beta$ 唯一提升为 $x\to e'$。

**答案 19.2.** coCartesian lift 沿边 $\alpha:s\to t$ 把纤维 $X_s$ 中对象向前推到 $X_t$，因此给出协变传输。

**答案 19.3.** 对伪函子 $F:B^{op}\to\mathbf{Cat}$，Grothendieck construction 对象为 $(b,x\in F(b))$；态射 $(b,x)\to(c,y)$ 为 $\alpha:b\to c$ 和 $x\to F(\alpha)(y)$。

**答案 19.4.** 第六章公式用普通逗号范畴 $K/d$ 上的余极限；定义 19.9 用 $\infty$-categorical slice $C\times_DD_{/d}$ 上的同伦余极限。

**答案 19.5.** HTT 的 horn 定义用 marked/inner fibration lifting 表达 Cartesian 边的泛性质；映射空间定义是其等价的同伦不变表述。

**答案 19.6.** $C^\natural$ 标记所有等价边；$C^\sharp$ 标记所有边；$C^\flat$ 只标记退化边。通常有包含关系 $C^\flat\subseteq C^\natural\subseteq C^\sharp$。

**答案 19.7.** Cartesian fibration 不只需要总空间和基空间，还需要区分哪些边实现拉回传输。marked simplicial sets 允许把这些 Cartesian edges 作为结构的一部分记录下来。

**答案 19.8.** 普通 Cartesian arrow 要求任意箭头按基底分解唯一提升；映射空间同伦拉回条件把“唯一”替换为“提升空间可缩”，因此是同伦化的唯一分解性质。

**答案 19.9.** 对 $(\alpha,\varphi):(b,x)\to(c,y)$，右乘 $(\operatorname{id}_b,\operatorname{id}_x)$ 给
$$
(\alpha\operatorname{id}_b,\ F(\operatorname{id}_b)(\varphi)\operatorname{id}_x)=(\alpha,\varphi).
$$
左乘 $(\operatorname{id}_c,\operatorname{id}_y)$ 同理给 $(\alpha,\operatorname{id}_{F(\alpha)y}\varphi)=(\alpha,\varphi)$。

**答案 19.10.** 对象为 $(0,a)$ 与 $(1,b)$，其中 $a\in\mathcal A,b\in\mathcal B$。纤维分别是 $\mathcal A$ 与 $\mathcal B$。跨纤维态射只能从 $(0,a)$ 到 $(1,b)$，由 $\alpha:0\to1$ 和态射 $a\to u(b)$ 组成；没有从 $1$ 到 $0$ 的跨纤维态射。

**答案 19.11.** 设有跨纤维态射 $(0,a)\to(1,b)$，它由 $\mathcal A$ 中态射 $\phi:a\to u(b)$ 给出。若它覆盖 $0\to1$，则经 Cartesian lift
$$
(0,u b)\to(1,b)
$$
分解时，唯一候选为纤维 $0$ 中的态射 $(0,a)\to(0,u b)$，即 $\phi:a\to u(b)$。复合公式给回原态射，唯一性由 $\phi$ 被原跨纤维态射唯一决定。

**答案 19.12.** Cartesian lift 的泛性质只给出可缩的选择空间，而不是指定一个严格唯一对象。两个选择都是同一泛性质的解，因此由唯一到同伦唯一的原则给出等价。

**答案 19.13.** 设 $e:x\to y$ 覆盖 $\alpha$、$f:y\to z$ 覆盖 $\beta$ 且二者 Cartesian。任意箭头 $w\to z$ 若基底分解经过 $\beta\alpha$，先由 $f$ 的 Cartesian 性唯一分解到 $y$，再由 $e$ 的 Cartesian 性唯一分解到 $x$。两次唯一性合成给出 $fe$ 的 Cartesian 性。

**答案 19.14.** 对 $x\in\mathcal C_2$，等式 $u_{02}=u_{01}u_{12}$ 表示直接把 $x$ 从纤维 $2$ 限制到纤维 $0$，等于先限制到纤维 $1$ 再限制到纤维 $0$：
$$
u_{02}(x)=u_{01}(u_{12}(x)).
$$

**答案 19.15.** 当 $S=[1]$ 且限制函子为 $u:\mathcal B\to\mathcal A$ 时，Cartesian section 由 $a\in\mathcal A$、$b\in\mathcal B$ 和等价 $a\simeq u(b)$ 组成；在普通严格模型中就是选择 $b$ 并令 $a=u(b)$。

**答案 19.16.** Descent data 要求在各局部对象上选择数据，并在交叠和高重交叠上给出相容等价。把这些局部范畴组织为覆盖单纯形上的 Cartesian fibration 后，Cartesian sections 正是这种同伦相干的相容选择；定理 19.H 把它识别为相应图形的极限。

**答案 19.17.** 给定 $F(b)$ 中态射 $\varphi:x\to y$，令 Grothendieck construction 中的态射为
$$
(\operatorname{id}_b,\varphi):(b,x)\to(b,y).
$$
它覆盖 $B$ 中的 $\operatorname{id}_b$。由于 $F(\operatorname{id}_b)=\operatorname{id}_{F(b)}$，这正是定义 19.A 的态射数据。

**答案 19.18.** 对象映射为 $(b,a)\mapsto(b,a)\in B\times\mathcal A$。态射 $(\alpha:b\to c,\varphi:a\to a')$ 在常值函子的 Grothendieck construction 中对应 $(\alpha,\varphi):(b,a)\to(c,a')$，因为限制函子为恒等。复合公式化为 $(\beta,\psi)(\alpha,\varphi)=(\beta\alpha,\psi\varphi)$，与乘积范畴复合一致。

**答案 19.19.** 当 $S=\Delta^0$ 时，straightening 对应常值为总纤维 $X$ 的图形。Cartesian section 只是选择 $X$ 中一个对象；section 之间的态射就是 $X$ 中的态射。因此 $\operatorname{Sect}^{Cart}_{\Delta^0}(X)\simeq X$，与定理 19.H 的点状极限一致。

## 第二十章

**答案 20.1.** 若 $0$ 是零对象，$X\to0$ 与 $0\to Y$ 唯一，故复合 $X\to0\to Y$ 给出唯一零态射。

**答案 20.2.** 链映射 $f:A\to B$ 的映射锥为 $\operatorname{Cone}(f)^n=B^n\oplus A^{n+1}$，微分 $d(b,a)=(d_Bb+f(a),-d_Aa)$。在导出 $\infty$-范畴中它表示余纤维。

**答案 20.3.** 三角范畴只记录同伦范畴和 distinguished triangles，不记录映射谱或高阶相干，因此不能唯一恢复稳定 $\infty$-范畴。

**答案 20.4.** sequential spectrum 为 pointed spaces 序列 $E_n$ 和结构映射 $\Sigma E_n\to E_{n+1}$；$\Omega$-谱要求伴随映射 $E_n\to\Omega E_{n+1}$ 为弱等价。

**答案 20.5.** heart 为阿贝尔范畴需要稳定性提供纤维/余纤维和正合三角，t-结构提供截断和正负正交，从而定义核、余核并证明阿贝尔公理。

**答案 20.6.** $\Sigma X$ 由推出方块 $X\to0\leftarrow0$ 定义。稳定性使该推出方块同时为拉回，因此 $X$ 表示 $0\to\Sigma X$ 的纤维，即 $\Omega\Sigma X\simeq X$。

**答案 20.7.** $\Sigma X=\operatorname{cofib}(X\to0)$。正合函子保持零对象和余纤维，故
$$
F(\Sigma X)\simeq F\operatorname{cofib}(X\to0)\simeq\operatorname{cofib}(FX\to0)=\Sigma F(X).
$$

**答案 20.8.** 结构映射为 $\sigma_n:\Sigma E_n\to E_{n+1}$。由 $\Sigma\dashv\Omega$，它对应伴随映射
$$
\tilde\sigma_n:E_n\to\Omega E_{n+1}.
$$
$\Omega$-谱条件要求每个 $\tilde\sigma_n$ 都是等价。

**答案 20.9.** 球面满足 $\Sigma S^n\simeq S^{n+1}$，结构映射取该标准识别。它与悬挂坐标选择相容，并把 sequential spectrum 的第 $n$ 项自然推进到第 $n+1$ 项。

**答案 20.10.** 三角范畴只记录 $\pi_0$ 层面的 Hom 群和 distinguished triangles；映射谱还记录所有 $\pi_n$，即所有悬挂度数上的态射群及其高阶同伦相干。因此不同稳定 $\infty$-范畴可能有相同三角同伦范畴但不同映射谱。

**答案 20.11.** 正合函子与悬挂相容，所以给出
$$
hC(\Sigma^nX,Y)\to hD(F\Sigma^nX,FY)\cong hD(\Sigma^nFX,FY).
$$
这就是映射谱同伦群上的诱导映射。

**答案 20.12.** $\mathbf{Sp}$ 是幺半 $\infty$-范畴，乘法为 smash product。一个 $E_1$-代数正是带同伦相干结合乘法和单位的对象，因此 ring spectrum 应定义为 $\mathbf{Sp}$ 中的 $E_1$-代数。

**答案 20.13.** 按定义 $H^0(Y)=\tau_{\le0}\tau_{\ge0}Y$。先截到 $C_{\ge0}$，再截到 $C_{\le0}$，所得对象同时在 $C_{\ge0}$ 和 $C_{\le0}$ 中，因此属于 heart。

**答案 20.14.** 余核对象 $Q$ 应满足：对任意 heart 对象 $T$，从 $Q$ 到 $T$ 的映射等价于从 $B$ 到 $T$ 且复合 $A\to B\to T$ 为零的映射。$\operatorname{cofib}(f)$ 表示在稳定范畴中杀掉 $A$ 后的对象；取 $H^0$ 把它返回 heart，因此给出 heart 中余核。

**答案 20.15.** 两步滤过为 $0=F_{-1}X\to F_0X\to F_1X=X$。因此
$$
\operatorname{gr}_0X=\operatorname{cofib}(0\to F_0X)\simeq F_0X,
$$
$$
\operatorname{gr}_1X=\operatorname{cofib}(F_0X\to X).
$$

**答案 20.16.** 在 exact couple 中 $d=jk$。于是
$$
d^2=jkjk=j(kj)k.
$$
正合性给出 $\operatorname{im}(j)=\ker(k)$，因此 $kj=0$，故 $d^2=0$。

**答案 20.17.** 若滤过有限，例如 $F_pX=0$ 对 $p<a$ 且 $F_pX=X$ 对 $p>b$，则 $\operatorname{gr}_pX=0$ 除有限多个 $p$ 外皆为零。固定总次数 $n=p+q$ 时，只有这些有限个 $p$ 可能贡献 $E_1^{p,q}$。

**答案 20.18.** $E_\infty$ 页描述的是目标 $H^*(X)$ 上某个滤过的 associated graded，即各层商 $F_p/F_{p-1}$。从层商恢复对象还需要扩张数据；不同扩张可能有相同 associated graded。

**答案 20.19.** 若 $A,B$ 有 biproduct，则
$$
f+g=
A\xrightarrow{\Delta}A\oplus A\xrightarrow{f\oplus g}B\oplus B\xrightarrow{\nabla}B.
$$
这里 $\Delta$ 是对角态射，$\nabla$ 是余对角态射。

**答案 20.20.** 若 $fu=fv$，则 $f(u-v)=0$。由核的泛性质，$u-v$ 唯一经 $\ker(f)$ 分解。若 $\ker(f)=0$，则 $u-v=0$，故 $u=v$。因此 $f$ 是 monomorphism。

**答案 20.21.** 短正合列给出三角 $A\to B\to C\to\Sigma A$，长正合列为
$$
\cdots\to H^n(A)\to H^n(B)\to H^n(C)\xrightarrow{\partial}
H^{n+1}(A)\to H^{n+1}(B)\to\cdots.
$$
若 $A,B,C$ 本身位于 heart，则只有 $n=0$ 附近非零，并恢复通常短正合列的 exactness。

**答案 20.22.** 三步滤过为 $0=F_{-1}X\to F_0X\to F_1X\to F_2X=X$。因此
$$
\operatorname{gr}_0X\simeq F_0X,\qquad
\operatorname{gr}_1X=\operatorname{cofib}(F_0X\to F_1X),
$$
$$
\operatorname{gr}_2X=\operatorname{cofib}(F_1X\to X).
$$

**答案 20.23.** 有限滤过使每个总次数只含有限多个 filtration degree。导出 exact couple 时，微分不可能从任意远处持续进入或离开固定项，因此没有无限链导致的 $\lim^1$ 或无限扩张障碍。

**答案 20.24.** 标准 t-结构中，heart 由只在次数 $0$ 有上同调的复形组成，等价于原阿贝尔范畴。通常上同调 $H^n(X)$ 被视为集中在次数 $0$ 的对象，因此属于 heart。

**答案 20.25.** Exhaustive filtration 意味滤过的所有层合起来恢复原对象，即 $\operatorname{colim}_pF_pX\to X$ 是等价。它排除“滤过只覆盖了对象一部分”的情况。

**答案 20.26.** Separated 要求无限向下交集为零，避免同一元素在所有滤过层中不可分辨。Complete 要求对象等于由滤过商得到的极限，保证可从所有有限阶段恢复整体。

**答案 20.27.** t-结构 left complete，若每个对象 $X$ 都满足
$$
X\simeq\lim_n\tau_{\le n}X.
$$
也就是说，对象由其 Postnikov tower 的向下截断极限恢复。

**答案 20.28.** 有限滤过只有有限多个 graded pieces，因此不存在无限下降链、无限上升链或 $\lim^1$ 型完成障碍。谱序列在每个总次数经过有限步即稳定。

**答案 20.29.** Postnikov tower 的相邻层通常由 cohomology object 控制：
$$
\operatorname{fib}(\tau_{\le n}X\to\tau_{\le n-1}X)
$$
等价于 $H^n(X)$ 的相应平移。因此 graded pieces 是 cohomology objects 的移位。

**答案 20.30.** 固定总次数后，只有有限多个 $p$ 上的 $E_1^{p,n-p}$ 非零。微分若要进入或离开这些项，只能连接有限集合中的位置；当页数足够大时已无可能的源或靶，因此不存在无限微分链。

**答案 20.31.** 在稳定 $\infty$-范畴中，态射 $f:X\to Y$ 产生纤维-余纤维序列
$$
\operatorname{fib}(f)\to X\to Y\to\operatorname{cofib}(f).
$$
稳定性把该序列延长为
$$
\Omega\operatorname{cofib}(f)\to X\to Y\to\operatorname{cofib}(f).
$$
前两项正是 $f$ 的纤维序列，因此
$$
\operatorname{fib}(f)\simeq\Omega\operatorname{cofib}(f).
$$

**答案 20.32.** 若 $F$ 保持零对象和有限余极限，则保持悬挂：
$$
F\Sigma X\simeq F\operatorname{cofib}(X\to0)\simeq\operatorname{cofib}(FX\to0)=\Sigma FX.
$$
由于在稳定范畴中 $\Omega$ 是 $\Sigma$ 的逆等价，对 $X=\Sigma\Omega X$ 应用上式得 $\Sigma F\Omega X\simeq FX$，再取 $\Omega$ 得 $F\Omega X\simeq\Omega FX$。

**答案 20.33.** 三角范畴上的 exact functor 只给出 Hom 集层面和 distinguished triangles 的相容性。映射谱还包含所有悬挂次数上的态射、路径之间的高阶同伦和复合相干；这些信息在同伦范畴中被压缩到 $\pi_0$，所以不能由 exact triangle functor 自动恢复。

## 第二十一章

**答案 21.1.** 集合值 sheaf 的下降是等化子条件；space 值 sheaf 的下降是同伦极限条件，包含高阶相容同伦。

**答案 21.2.** 平凡站点 $*$ 上的 $\infty$-sheaf 范畴就是 $\operatorname{Fun}(*,\mathcal S)\simeq\mathcal S$。

**答案 21.3.** Čech nerve 前三层为 $\coprod_iU_i$、$\coprod_{i,j}U_i\times_UU_j$、$\coprod_{i,j,k}U_i\times_UU_j\times_UU_k$。

**答案 21.4.** left exact localization 是保持有限极限的反射 $L:\mathcal P\to\mathcal X$。sheaf 化应保持有限极限，因为 sheaf topos 的有限极限应与局部粘合相容。

**答案 21.5.** 普通 Giraud 用集合值范畴的余极限普遍性、等价关系有效等；高阶 Giraud 把等价关系替换为群胚对象有效，把集合级条件提升为同伦相容条件。

**答案 21.6.** 离散 space 值 $\infty$-sheaf 对覆盖族的 Čech nerve 给出同伦极限。对离散对象取 $\pi_0$ 后，同伦极限条件化为普通等化子条件，因此得到 ordinary sheaf。

**答案 21.7.** space $X$ 为 $0$-截断意味着对任意基点，其高阶同伦群消失，且路径空间离散；于是 $X$ 等价于离散 space $\pi_0X$。离散 spaces 正是集合。

**答案 21.8.** 覆盖 $U_0\to U$ 的 Čech nerve 满足 $U_n=U_0\times_U\cdots\times_UU_0$。每个 matching map 由相应投影和重复交叉给出，并由覆盖的拉回稳定性仍为覆盖，因此 Čech nerve 是超覆盖。

**答案 21.9.** Čech descent 只检查由单个覆盖族反复取交叉得到的单纯对象。Hyperdescent 还允许每一维用新的覆盖去覆盖 matching object，因此检查高维粘合数据及其相容同伦，比 Čech descent 更强。

**答案 21.10.** 对 $f:U\to X$，Čech nerve 为
$$
U,\quad U\times_XU,\quad U\times_XU\times_XU,\quad\ldots.
$$
其几何实现把 $U$ 中映到同一 $X$ 点的元素按全部高阶相容关系粘合。若 $f$ 是 effective epimorphism，该粘合结果正是 $X$。

**答案 21.11.** 在 $\mathbf{Set}$ 中，$U\to X$ 的 Čech nerve 的几何实现是按关系“两个元素有相同像”取商。该商等于 $X$ 当且仅当 $U\to X$ 满射。

**答案 21.12.** Groupoid object 是内部对象、箭头、复合、单位和逆的同伦相干版本。它描述对象之间的等价关系及其高阶相容同伦，因此是内部同伦等价关系。

**答案 21.13.** 在 $\mathcal S$ 中，任意集合看成离散 space 是 $0$-截断对象。任意普通群胚的 nerve，或 $BG$，是 $1$-截断对象；它可能有非平凡 $\pi_1$，但高于 $1$ 的同伦群消失。

**答案 21.14.** Postnikov tower 把对象逐层截断为 $0$-型、$1$-型、$2$-型等近似。它把高阶 sheaf 的信息拆成可逐级理解的同伦层，并为 hypercompletion 和同伦 sheaf cohomology 提供基础。

**答案 21.15.** Hyperdescent 是对所有 hypercovers 的下降条件。Hypercompletion 是把 $\infty$-topos 局部化到满足这种更强下降或由 Postnikov tower 可恢复的对象。标准条件下，hypercomplete sheaves 正是满足 hyperdescent 的 sheaves。

**答案 21.16.** 几何态射 $f:\mathcal X\to\mathcal Y$ 给出
$$
f^*:\mathcal Y\rightleftarrows\mathcal X:f_*,
$$
其中 $f^*$ 是左伴随且保持有限极限，$f_*$ 是右伴随。

**答案 21.17.** 点 $x$ 的 stalk $F_x$ 把 sheaf $F$ 送到其在 $x$ 附近的局部截面余极限。这个函子从 $\operatorname{Sh}_\infty(X)$ 到 $\mathcal S$，保持有限极限，并作为几何态射 $\mathcal S\to\operatorname{Sh}_\infty(X)$ 的 inverse image。

**答案 21.18.** 条件形式相同：都有伴随 $f^*\dashv f_*$，并要求 inverse image $f^*$ 保持有限极限。差别在于 $\infty$-topos 中这些范畴、函子和极限都在 $\infty$-范畴意义下理解，保留高阶同伦相干。

**答案 21.19.** 若 $F,G$ 是 $\infty$-sheaves，则定义
$$
(F\times G)(U)=F(U)\times G(U).
$$
对覆盖 Čech nerve $U_\bullet$，因为 $F$ 与 $G$ 分别满足下降，
$$
F(U)\times G(U)\simeq
\left(\lim_\Delta F(U_\bullet)\right)\times
\left(\lim_\Delta G(U_\bullet)\right)
\simeq
\lim_\Delta(F(U_\bullet)\times G(U_\bullet)).
$$
故逐点积仍是 sheaf，并满足积的泛性质。

**答案 21.20.** 命题 21.P 要把
$$
\lim_\alpha\lim_\Delta F_\alpha(U_\bullet)
$$
改写为
$$
\lim_\Delta\lim_\alpha F_\alpha(U_\bullet).
$$
这正是极限的 Fubini 交换律；没有它，就不能把“每个 $F_\alpha$ 满足下降”传递给逐点极限。

**答案 21.21.** 若 $F$ objectwise 离散，则对应集合值预层满足
$$
F(U)\to\prod_iF(U_i)\rightrightarrows\prod_{i,j}F(U_i\times_UU_j)
$$
为等化子。这里第一箭头是限制到覆盖，两个平行箭头是在双交上从两个覆盖分量限制得到。

## 第二十二章

**答案 22.1.** 普通代数对象用乘法和单位加严格交换图定义；定义 22.9 中的 $\infty$-operad 代数是 operad 映射，所有结合律、单位律和交换律由高阶单纯形相干编码。

**答案 22.2.** $E_\infty$-代数的交换律不是普通等式 $ab=ba$。它首先给出乘法 $A\otimes A\to A$ 与交换后乘法
$$
A\otimes A\xrightarrow{\tau}A\otimes A\to A
$$
之间的同伦；其次，对三个、四个及更多输入，还给出这些同伦之间的更高同伦。$E_\infty$-operad 的相应运算空间可缩，表示所有交换和重排的选择在同伦意义下唯一且彼此相容。因此 $E_\infty$-代数是“同伦相干交换”的代数，而不是严格交换代数。

**答案 22.3.** $\mathbf{Fin}_*$ 对象为有限带基点集合 $\langle n\rangle=\{*,1,\dots,n\}$。inert morphism 是每个非基点的原像恰有一个非基点的映射。

**答案 22.4.** 环谱是谱的幺半 $\infty$-范畴中的 $E_1$-代数：乘法 $A\wedge A\to A$ 和单位 $\mathbb S\to A$ 满足同伦相干结合律。

**答案 22.5.** 张量积分别保持余极限有三处关键用途。第一，自由代数通常由树形或单纯形表达式的几何实现构造，表达式中需要把张量与余极限交换。
第二，证明 $\operatorname{Alg}_{\mathcal O}(C)$ 或模范畴 presentable 时，要把底层范畴的可达余极限提升到代数对象。
第三，相对张量积 $M\otimes_A N$ 由 bar 构造的几何实现给出；若张量不保持这些余极限，bar 构造不能稳定地表达平衡泛性质。

**答案 22.6.** 非基点中不被送到 $*$ 的集合为 $T=\{1,3\}$。令 $\rho:\langle3\rangle\to\langle2\rangle$ 把 $1\mapsto1,3\mapsto2,2\mapsto*,*\mapsto*$；这是 inert。令 $\alpha:\langle2\rangle\to\langle2\rangle$ 把 $1\mapsto1,2\mapsto1,*\mapsto*$；这是 active，且 $f=\alpha\rho$。

**答案 22.7.** inert 态射要求目标每个非基点有唯一原像，因此只是在源中选出若干输入并按槽位投影。active 态射不把非基点送到基点，因此保留全部输入并把它们合成为目标输出，正对应多输入运算。

**答案 22.8.** $\rho_2:\langle3\rangle\to\langle1\rangle$ 满足 $\rho_2(2)=1$，并把 $*,1,3$ 都送到 $*$。目标唯一非基点 $1$ 的原像恰为 $\{2\}$，所以 $\rho_2$ 是 inert。

**答案 22.9.** Segal 条件说由 inert 投影诱导的函子
$$
\mathcal O^\otimes_{\langle n\rangle}\to\prod_{i=1}^n\mathcal O^\otimes_{\langle1\rangle}
$$
是等价。因此给定 $\langle n\rangle$ 上对象等价于给定它沿每个输入槽的投影，也就是给定 $n$ 个颜色的列表。

**答案 22.10.** 若 $C$ 是普通集合或阿贝尔群上的幺半范畴，左 $A$-模作用就是映射 $A\otimes M\to M$。在普通环情形中，底层集合上写作 $A\times M\to M$，满足结合律 $(ab)m=a(bm)$ 和单位律 $1m=m$。

**答案 22.11.** 由定义：
$$
\operatorname{Bar}_0(M,A,N)=M\otimes N,
$$
$$
\operatorname{Bar}_1(M,A,N)=M\otimes A\otimes N,
$$
$$
\operatorname{Bar}_2(M,A,N)=M\otimes A\otimes A\otimes N.
$$
面映射分别使用右作用、乘法和左作用；退化映射插入单位。

**答案 22.12.** 普通平衡张量积把 $(ma)\otimes n$ 与 $m\otimes(an)$ 识别。Bar 构造用一整个单纯对象系统地加入这些识别及其高阶相干，因此其几何实现给出同伦意义下的平衡张量积。

**答案 22.13.** 若 $M$ 是 $(A,B)$-双模，$N$ 是 $(B,C)$-双模，则复合应消去中间代数 $B$ 的左右作用。相对张量积 $M\otimes_BN$ 正是把 $mb\otimes n$ 与 $m\otimes bn$ 以同伦相干方式识别，因此得到 $(A,C)$-双模。

**答案 22.14.** 普通代数中 $Z(A)=\{a\in A\mid ax=xa\ \forall x\in A\}$。它也可看作 $A$ 作为 $(A,A)$-双模的双模自同态。定义 22.I 正是把这个 endomorphism 描述提升到 $\infty$-范畴。

**答案 22.15.** 圆周可由区间端点粘合得到；沿圆周积分一个 $E_1$-代数时，局部乘法数据按循环顺序粘合，形成 cyclic bar construction。其几何实现就是 Hochschild homology $HH(A)$。

**答案 22.16.** 在 Morita $\infty$-范畴中，$M:{}_A\to{}_B$ 和 $N:{}_B\to{}_A$ 的复合分别是 $M\otimes_BN$ 与 $N\otimes_AM$。若它们等价于单位双模 $A$ 和 $B$，就正是说 $M$ 与 $N$ 互为逆 1-态射。

**答案 22.17.** Dualizable 只要求对象有对偶以及评价/余评价。Fully dualizable 还要求这些评价/余评价态射本身继续有左右 adjoints，并递归到所有高阶层级，因此是更强的有限性条件。

**答案 22.18.** Cobordism hypothesis 说 framed fully extended TFT 构成的 $\infty$-群胚等价于目标中 fully dualizable objects 的 $\infty$-群胚。等价由“取点的值”给出，因此一旦知道点的取值及其 fully dualizable 结构，整个场论由定理唯一延拓。

**答案 22.19.** 单位双模 ${}_AA_A$ 的底层对象是 $A$。左作用和右作用都是乘法 $A\otimes A\to A$，分别作用在左因子和右因子上；结合律保证左右作用相容，单位律保证模单位公理。

**答案 22.20.** 映射 $A\otimes_AM\to M$ 为 $a\otimes m\mapsto am$。其逆为 $m\mapsto1\otimes m$。平衡关系给出 $a\otimes m=1\otimes am$，所以两个复合都是恒等。

**答案 22.21.** 对右 $A$-模 $P$，
$$
(P\otimes_AM)\otimes_BN
\cong P\otimes_A(M\otimes_BN)
\cong P\otimes_AA
\cong P.
$$
对右 $B$-模 $Q$ 同理得到
$$
(Q\otimes_BN)\otimes_AM\cong Q.
$$

**答案 22.22.** 取标准基 $e_i$ 与对偶基 $e^j$。映射
$$
k^n\otimes(k^n)^*\to M_n(k)
$$
把 $e_i\otimes e^j$ 送到矩阵单位 $E_{ij}$，即只在第 $(i,j)$ 位为 $1$ 的矩阵。

**答案 22.23.** Proper 条件要求 Hom 对象在基环上 perfect 或有限型。它说明任意两个对象之间的态射复形没有无限维不可控部分，是 Morita 理论中保证评价/余评价存在并有伴随的有限性条件之一。

**答案 22.24.** 非退化配对给出 $A\cong A^*$，因此任意双线性型都唯一对应一个线性映射到对偶。要求
$$
\langle ab,c\rangle=\langle a\otimes b,\Delta(c)\rangle
$$
对所有 $a,b,c$ 成立，因配对非退化，$\Delta(c)$ 被唯一确定。

**答案 22.25.** 圆柱是圆周到自身的恒等 bordism。代数上它可分解为单位后接乘法或余乘法后接余单位；Frobenius 代数的单位律和余单位律保证所得线性映射为 $\operatorname{id}_A$。

**答案 22.26.** 普通二维 TFT 只给闭一维流形赋值，因此基本对象是圆周，其值为 Frobenius 代数。Fully extended TFT 还给点、区间和带角 bordism 赋值；cobordism hypothesis 说最高层数据由点上的 fully dualizable object 控制。

**答案 22.27.** 普通幺半范畴中代数对象 $A$ 有乘法 $\mu:A\otimes A\to A$ 和单位 $\eta:\mathbb 1\to A$。结合律图要求
$$
\mu(\mu\otimes1)=\mu(1\otimes\mu):A\otimes A\otimes A\to A,
$$
单位律要求
$$
\mu(\eta\otimes1)=\operatorname{id}_A=\mu(1\otimes\eta)
$$
在单位约束识别后成立。

**答案 22.28.** 同伦相干结合律本来是某些比较路径或高阶单纯形的可缩选择。若映射空间离散，则两个平行态射之间存在同伦当且仅当它们相等，且高阶同伦没有额外自由度。因此相干条件退化为普通交换图严格交换。

**答案 22.29.** 强幺半函子给出结构等价
$$
F(A)\otimes F(A)\simeq F(A\otimes A),\qquad \mathbb 1_D\simeq F(\mathbb 1_C).
$$
于是 $FA$ 的乘法为
$$
FA\otimes FA\simeq F(A\otimes A)\xrightarrow{F\mu}FA,
$$
单位为
$$
\mathbb 1_D\simeq F\mathbb 1_C\xrightarrow{F\eta}FA.
$$
幺半函子的相干性把 $A$ 的结合律和单位律传递给 $FA$。

## 第二十三章

**答案 23.1.** 由 $\infty$-Yoneda，
$$
\operatorname{Map}_{\mathcal P(C)}(j(x),j(y))\simeq j(y)(x)=\operatorname{Map}_C(x,y).
$$
该等价对 $x,y$ 自然，所以 $j$ 全忠实。

**答案 23.2.** 终对象 $1_{\mathcal P(C)}$ 应满足对任意 $F$，映射空间 $\operatorname{Map}(F,1)$ 可缩。逐点取 $1_{\mathcal S}$ 得到预层 $c\mapsto1_{\mathcal S}$，自然变换到它逐点唯一到可缩，因此它是终对象。

**答案 23.3.** 对滤过图形 $F_i$，用 Yoneda 和逐点余极限：
$$
\operatorname{Map}(j(c),\operatorname{colim}_iF_i)
\simeq
(\operatorname{colim}_iF_i)(c)
\simeq
\operatorname{colim}_iF_i(c)
\simeq
\operatorname{colim}_i\operatorname{Map}(j(c),F_i).
$$

**答案 23.4.** 第十二章用 Hom 集保持滤过余极限定义紧对象；定义 23.4 把 Hom 集替换为映射空间，并要求 $\operatorname{Map}_C(x,-)$ 保持滤过余极限。

**答案 23.5.** $\operatorname{Ind}_\kappa(C)$ 是 $\mathcal P(C)$ 中包含所有可表预层 $j(c)$ 且对 $\kappa$-滤过余极限封闭的最小全子 $\infty$-范畴。

**答案 23.6.** $\mathcal S\simeq\mathcal P(*)$，其中 $*$ 是终小 $\infty$-范畴。预层 $\infty$-范畴 presentable，因此 $\mathcal S$ presentable。

**答案 23.7.** 若 $C,D$ presentable 且 $F:C\to D$ 保持所有小余极限，则 presentable $\infty$-范畴伴随函子定理说明 $F$ 是左伴随，因此有右伴随。

**答案 23.8.** 若 $X$ 局部，则 $\eta_X:X\to LX$ 是等价。对 $LX$，局部化公理给出 $\eta_{LX}:LX\to L^2X$ 与 $L\eta_X$ 等价，而 $L\eta_X$ 是等价，所以 $LX$ 局部。

**答案 23.9.** $Z$ 是 $S$-局部对象，若对每个 $f:A\to B$ 属于 $S$，
$$
\operatorname{Map}(B,Z)\to\operatorname{Map}(A,Z)
$$
是等价。这说明 $Z$ 无法区分 $f$ 的源和靶，即 $f$ 在映入 $Z$ 时被视为等价。

**答案 23.10.** Sheaf 化把任意预层 $F$ 送到 sheaf $aF$，并且对任意 sheaf $G$ 有
$$
\operatorname{Map}(aF,G)\simeq\operatorname{Map}(F,G).
$$
局部对象是 sheaves，局部等价由覆盖下降检测，因此它是 Bousfield localization 的一个几何例子。

**答案 23.11.** $\infty$-topos 可定义为预层 $\infty$-范畴的 left exact accessible localization。因此 left exact localization 是从自由预层世界构造高阶 sheaf 世界的机制。

**答案 23.12.** 稳定 $\infty$-范畴中推出方块等价于拉回方块，纤维可由余纤维和环路表达，环路又是悬挂的逆。因此保持有限余极限和零对象就控制有限极限。

**答案 23.13.** $\operatorname{Pr}^L$ 的对象是 presentable $\infty$-categories，态射是保持小余极限的函子，即左伴随。$\operatorname{Pr}^R$ 对象相同，态射是右伴随。

**答案 23.14.** 等价 $(\operatorname{Pr}^L)^{op}\simeq\operatorname{Pr}^R$ 表示给左伴随 $F:C\to D$ 取其右伴随 $G:D\to C$，可反向得到 $\operatorname{Pr}^R$ 中的态射，并且这个过程在高阶相干意义下可逆。

**答案 23.15.** 张量积 $C\otimes D$ 由泛性质刻画：
$$
\operatorname{Fun}^L(C\otimes D,E)
\simeq
\operatorname{Fun}^{L,L}(C\times D,E),
$$
右侧是分别保持余极限的双变量函子。

**答案 23.16.** 若张量积分别保持余极限，则自由代数、模对象和 bar 几何实现能在 presentable 环境中构造，并且相对张量积与余极限相容。这是第二十二章模 $\infty$-范畴和相对张量积存在性的基本假设。

**答案 23.17.** $L$-等价的定义是 $Lf$ 为等价。若 $f:X\to Y$ 本身是等价，则存在逆 $g:Y\to X$ 及同伦 $gf\simeq\operatorname{id}_X$、$fg\simeq\operatorname{id}_Y$。函子 $L$ 保持复合、单位和等价，于是 $Lg$ 是 $Lf$ 的逆，且
$$
Lg\,Lf\simeq L(gf)\simeq\operatorname{id}_{LX},\qquad
Lf\,Lg\simeq L(fg)\simeq\operatorname{id}_{LY}.
$$
因此 $Lf$ 是等价，故 $f$ 是 $L$-等价。

**答案 23.18.** 在由 $S$ 生成的局部化中，局部对象正是 $S$-局部对象。命题 23.27 说明 $f:X\to Y$ 被 $L$ 送成等价，当且仅当对每个 $S$-局部对象 $Z$，
$$
\operatorname{Map}(Y,Z)\to\operatorname{Map}(X,Z)
$$
是等价。因此局部等价可由所有局部对象共同检测。

**答案 23.19.** 若 $A,B$ 局部，则对任意 $X$，
$$
\operatorname{Map}(X,A\times B)\simeq\operatorname{Map}(X,A)\times\operatorname{Map}(X,B).
$$
把 $X$ 换成 $LX$，并用 $A,B$ 局部得到
$$
\operatorname{Map}(X,A\times B)\simeq\operatorname{Map}(LX,A\times B),
$$
故 $A\times B$ 局部。

## 第二十四章

**答案 24.1.** 对 $F:\mathcal C\to\mathcal D$，
$$
F_*(c,d)=\mathcal D(Fc,d),\qquad
F^*(d,c)=\mathcal D(d,Fc).
$$

**答案 24.2.** 若 $P:\mathcal C\nrightarrow\mathcal D$、$Q:\mathcal D\nrightarrow\mathcal E$，则
$$
(Q\circ P)(c,e)=\int^{d\in\mathcal D}P(c,d)\times Q(d,e).
$$

**答案 24.3.** 在 $(c,d)$ 处，
$$
(\operatorname{id}_{\mathcal D}\circ P)(c,d)
=
\int^{d'}P(c,d')\times\mathcal D(d',d)
\cong P(c,d)
$$
由 co-Yoneda 得到。

**答案 24.4.** 余单位
$$
\int^{c}\mathcal D(d,Fc)\times\mathcal D(Fc,d')\to\mathcal D(d,d')
$$
把一对态射 $d\to Fc$ 与 $Fc\to d'$ 送到它们在 $\mathcal D$ 中的复合。

**答案 24.5.** 幂等 $e:X\to X$ 分裂，若存在 $r:X\to Y$ 与 $s:Y\to X$，使 $sr=e$ 且 $rs=\operatorname{id}_Y$。

**答案 24.6.** 若 $u:(X,e)\to(Y,f)$、$v:(Y,f)\to(Z,g)$，则
$$
vu=gvu=vfu=vue,
$$
所以复合满足 Karoubi 态射条件。

**答案 24.7.** 若 $\mathcal C$ 已幂等完备，则每个 $(X,e)$ 中的 $e$ 在 $\mathcal C$ 中分裂为某个对象 $Y$。于是 $(X,e)$ 同构于嵌入像中的 $(Y,\operatorname{id}_Y)$，故 $\mathcal C\to\operatorname{Kar}(\mathcal C)$ 本质满且完全忠实。

**答案 24.8.** 普通余极限是常值单点权重 $1:\mathcal J^{op}\to\mathbf{Set}$ 的加权余极限：
$$
\operatorname{colim}_{j}D(j)\cong\int^{j}1\times D(j).
$$

**答案 24.9.** 普通 profunctor 取集合值，只记录广义态射集合；$\infty$-correspondence 取 space 值，记录广义态射空间及其高阶同伦。

**答案 24.10.** Profunctor 复合用 coend 平衡中间范畴变量；Morita 理论中双模复合用相对张量积 $M\otimes_BN$ 平衡中间代数 $B$ 的作用。后者是前者在线性、导出或谱值环境中的高阶代数版本。

**答案 24.11.** 若 $P(c,d)\cong\mathcal D(Fc,d)$ 且 $P(c,d)\cong\mathcal C(c,Gd)$，合并得到自然同构
$$
\mathcal D(Fc,d)\cong\mathcal C(c,Gd).
$$
这正是伴随 $F\dashv G$ 的 Hom-集定义。

**答案 24.12.** 离散范畴中没有非恒等态射，coend 的平衡关系没有额外识别。因此
$$
(Q\circ P)(c,e)=\coprod_d P(c,d)\times Q(d,e).
$$
若每个集合至多一个元素，该集合非空当且仅当存在 $d$ 使 $cPd$ 且 $dQe$，正是关系复合。

**答案 24.13.** Profunctor 伴随的单位在 $(c,c')$ 处为
$$
\mathcal C(c,c')\to\mathcal D(Fc,Fc').
$$
单位为同构当且仅当所有这些 Hom 映射为双射，也就是 $F$ 完全忠实。

## 第二十五章

**答案 25.1.** $\mathcal V$-profunctor $M:\mathcal A\nrightarrow\mathcal B$ 是富函子
$$
M:\mathcal A^{op}\otimes\mathcal B\to\mathcal V.
$$

**答案 25.2.** 当 $\mathcal V=\mathbf{Ab}$ 时，$\mathcal A,\mathcal B$ 是预加性范畴，profunctor 给每对对象一个阿贝尔群 $M(a,b)$，并且左右态射作用是双线性的。

**答案 25.3.** 若 $M:\mathcal A\nrightarrow\mathcal B$、$N:\mathcal B\nrightarrow\mathcal C$，则
$$
(N\circ M)(a,c)=\int^{b\in\mathcal B}M(a,b)\otimes N(b,c).
$$

**答案 25.4.** 右单位为
$$
\int^{b'}M(a,b')\otimes\mathcal B(b',b)\cong M(a,b),
$$
这是富 co-Yoneda 公式。

**答案 25.5.** 在 equipment 中，companion 是垂直态射 $f:A\to B$ 对应的同向水平态射 $f_*:A\nrightarrow B$；conjoint 是反向水平态射 $f^*:B\nrightarrow A$，二者满足单位/余单位二重胞腔和三角恒等式。

**答案 25.6.** 对富函子 $F:\mathcal A\to\mathcal B$，
$$
F_*(a,b)=\mathcal B(Fa,b),\qquad
F^*(b,a)=\mathcal B(b,Fa).
$$

**答案 25.7.** 在 $\mathbf{Prof}$ 中，水平态射是双变量函子。带有垂直函子的二重胞腔正是使相应四边形相容的自然变换 $M(a,b)\to N(fa,gb)$。

**答案 25.8.** Beck-Chevalley 条件说“先沿一个方向推/求和再拉回”与“先拉回再推/求和”得到同构。它是 base change 合理性的抽象表达。

**答案 25.9.** 若 $X'\cong Y'\times_YX$，则对 $E\to X$，
$$
Y'\times_YE\cong X'\times_XE.
$$
两边分别对应先推后拉和先拉后推，因此给出 Beck-Chevalley 同构。

**答案 25.10.** 对伪函子 $\mathcal F:B^{op}\to\mathbf{Cat}$，Grothendieck construction 的对象是 $(b,x)$，其中 $x\in\mathcal F(b)$；态射 $(b,x)\to(c,y)$ 是 $\alpha:b\to c$ 与 $x\to\alpha^*y$。

**答案 25.11.** 给定 fibration $p:E\to B$ 和 $\alpha:b\to c$，对 $y\in E_c$ 取 Cartesian lift $\alpha^*y\to y$。该选择对对象和态射给出重索引函子 $\alpha^*:E_c\to E_b$。

**答案 25.12.** Indexed category 是普通伪函子 $B^{op}\to\mathbf{Cat}$；Cartesian fibration 是 $\infty$-范畴版本，编码 $S^{op}\to\mathcal{Cat}_\infty$。二者由 straightening/Grothendieck construction 相连。

**答案 25.13.** Ordinary bicategory 只有对象、1-态射和 2-态射。Equipment 额外区分垂直态射与水平态射，并要求垂直态射有 companion/conjoint，从而能表达 base change 方块。

**答案 25.14.** Morita 理论既有代数之间的结构保持映射，也有双模作为广义态射；复合由相对张量积给出。Equipment 或 $(\infty,2)$-equipment 能同时记录这些方向、二重胞腔和 Beck-Chevalley 型相干。

**答案 25.15.** 若 $L\dashv R$、$L'\dashv R'$，左侧自然变换 $\alpha:LA\to BL'$ 的右 mate 为复合
$$
A R'\xrightarrow{\eta}R L A R'\xrightarrow{R\alpha R'}R B L'R'\xrightarrow{R B\varepsilon'}R B.
$$
这里 $\eta$ 是 $L\dashv R$ 的单位，$\varepsilon'$ 是 $L'\dashv R'$ 的余单位。

**答案 25.16.** 在 $(a,b)$ 处，$F_*$ 给出 $\mathcal B(Fa,b)$。目标 profunctor 经 co-Yoneda 化简为 $\mathcal B'(Gu(a),v(b))$。若 $vF=Gu$，分量就是
$$
\mathcal B(Fa,b)\to\mathcal B'(vF(a),v(b))=\mathcal B'(Gu(a),v(b)),
$$
即把态射送到其在 $v$ 下的像。

**答案 25.17.** Beck-Chevalley 比较表达沿方块两条路径得到的重索引/推前相同。若两个相邻方块的比较都是同构，则外矩形的两条路径可分解为这两个比较的复合，因此外矩形比较也是同构。故 exact squares 应对粘合封闭。

## 第二十六章

**答案 26.1.** 对有滤过余极限的 $\infty$-范畴 $C$，对象 $K$ compact，若 $\operatorname{Map}_C(K,-)$ 保持滤过余极限。稳定情形可等价地用映射谱函子表述。

**答案 26.2.** 对谱 $X$，若所有 $\pi_nX\cong\pi_0\operatorname{Map}_{\mathbf{Sp}}(\Sigma^n\mathbb S,X)$ 为零，则 $X\simeq0$。因此 $\mathbb S$ 及其悬挂检测零对象并生成 $\mathbf{Sp}$。

**答案 26.3.** 在 $D(R)$ 中，
$$
\operatorname{Map}_{D(R)}(R,\Sigma^nX)
$$
计算 $X$ 的第 $n$ 个同调或上同调对象。若这些全为零，则复形为零对象，因此 $R$ 生成 $D(R)$。

**答案 26.4.** 稳定范畴 $C$ 的 localizing subcategory $L\subseteq C$ 是一个稳定全子范畴，并且对 $C$ 中存在的所有小余积封闭。稳定全子范畴意味着它含零对象，并对有限极限、有限余极限、纤维、余纤维和悬挂/环路封闭。因此若 $X\to Y\to Z$ 是余纤维序列且 $X,Y\in L$，则 $Z\in L$；再加上任意小族 $\{X_i\}\subseteq L$ 的余积 $\coprod_iX_i$ 仍在 $L$，就得到“局部化核”应有的闭包性质。

**答案 26.5.** Verdier quotient $q:C\to C/L$ 把 $L$ 中对象送为零，并且任意把 $L$ 送为零的正合函子 $C\to D$ 唯一因子化经 $C/L$。

**答案 26.6.** 三角范畴中的 Verdier quotient 只控制同伦范畴和三角；稳定 $\infty$-范畴商保留映射空间/映射谱，并在传到同伦范畴后给出三角 Verdier quotient。

**答案 26.7.** 一个形式是：若 $C$ compactly generated stable，且 $H:C^{op}\to\mathcal S$ cohomological 并把小余积送到小积，则 $H$ 由某个对象表示。

**答案 26.8.** 保持小余积的正合函子在稳定 presentable 范畴中保持所有小余极限，因为小余极限由小余积和有限余极限生成。由 presentable 伴随函子定理，它有右伴随。

**答案 26.9.** $X$ 是 $L$-acyclic，若 $LX\simeq0$。$Y$ 是 $L$-local，若单位 $Y\to LY$ 是等价。

**答案 26.10.** 若每个 $X_i$ acyclic，则
$$
L(\coprod_iX_i)\simeq\coprod_iLX_i\simeq0,
$$
因为 $L$ 保持小余积。

**答案 26.11.** 令 $A_X=\operatorname{fib}(X\to LX)$。稳定范畴中纤维序列也是余纤维序列，因此有 $A_X\to X\to LX$。应用 $L$ 后 $LX\to L^2X$ 为等价，所以 $LA_X=0$。

**答案 26.12.** Smashing localization 是形如 $LX\simeq E\otimes X$ 的 Bousfield localization。谱中写作 $LX\simeq E\wedge X$。

**答案 26.13.** 若 $L\simeq E\otimes-$，则对每个图形 $X_i$ 有
$$
L(\operatorname{colim}_iX_i)\simeq
E\otimes\operatorname{colim}_iX_i.
$$
在闭幺半 presentable 稳定范畴中，$E\otimes-$ 是左伴随，其右伴随为内部 Hom $[E,-]$，所以保持所有小余极限。于是
$$
E\otimes\operatorname{colim}_iX_i
\simeq\operatorname{colim}_i(E\otimes X_i)
\simeq\operatorname{colim}_iLX_i.
$$
因此 smashing localization 保持所有小余极限，特别保持小余积。

**答案 26.14.** Verdier quotient 后 compact objects 的像未必已经幂等完备；某些 retract 只在商中出现。为了得到全部 compact objects，需要对小商 $C^\omega/L^\omega$ 作 Karoubi 完备化。

**答案 26.15.** 若 $\operatorname{cofib}(f)$ 是 $L$-acyclic，则对余纤维序列
$$
X\to Y\to\operatorname{cofib}(f)
$$
应用 $L$ 得
$$
LX\to LY\to0.
$$
稳定范畴中余纤维为零等价于第一箭头为等价，因此 $Lf$ 是等价。

**答案 26.16.** 序列 $A_X\to X\to LX$ 中第三项是局部化函子 $L$ 的值。局部化幂等性给出 $LX\to L^2X$ 为等价，因此 $LX$ 按定义是局部对象。

**答案 26.17.** 有理化 $L(X)=H\mathbb Q\wedge X$。所以 $X$ 为 $L$-acyclic 当且仅当
$$
H\mathbb Q\wedge X\simeq0.
$$
态射 $f$ 为有理等价当且仅当 $H\mathbb Q\wedge\operatorname{cofib}(f)\simeq0$。

## 第二十七章

**答案 27.1.** Small dg category 是 $\operatorname{Ch}(k)$-富范畴：有对象集、Hom 复形 $\mathcal A(x,y)$、链映射形式的复合
$$
\mathcal A(y,z)\otimes\mathcal A(x,y)\to\mathcal A(x,z)
$$
和单位 $k\to\mathcal A(x,x)$，满足结合律和单位律。

**答案 27.2.** Hom 复形复合是链映射，所以 cycle 的复合仍为 cycle，boundary 与任意 cycle 复合后仍为 boundary。故复合下降到 $H^0$。链级结合律和单位律取 $H^0$ 后给出普通范畴公理。

**答案 27.3.** dg 函子是对象函数加 Hom 复形链映射 $\mathcal A(x,y)\to\mathcal B(Fx,Fy)$，相容于单位和复合。Quasi-equivalence 要求这些 Hom 链映射都是 quasi-isomorphism，且 $H^0(F)$ 本质满。

**答案 27.4.** Hom quasi-isomorphism 在 $H^0$ 上给出 Hom 集同构，因此 $H^0(F)$ 完全忠实。本质满是 quasi-equivalence 的第二条件，所以 $H^0(F)$ 是范畴等价。

**答案 27.5.** 右 dg 模是 dg 函子 $M:\mathcal A^{op}\to\operatorname{Ch}(k)$。可表模为
$$
h_a=\mathcal A(-,a).
$$

**答案 27.6.** 富自然变换 $h_a\to M$ 由单位 $k\to\mathcal A(a,a)$ 上的像决定，即由 $M(a)$ 的元素决定；反向由右模作用
$$
M(a)\otimes\mathcal A(x,a)\to M(x)
$$
给出。两构造互逆并与微分相容。

**答案 27.7.** 对滤过图形 $M_i$，
$$
\operatorname{Map}(h_a,\operatorname{colim}_iM_i)
\simeq
(\operatorname{colim}_iM_i)(a)
\simeq
\operatorname{colim}_iM_i(a)
\simeq
\operatorname{colim}_i\operatorname{Map}(h_a,M_i).
$$
所以 $h_a$ compact。

**答案 27.8.** Perfect modules 是 $D(\mathcal A)$ 中由可表模 $h_a$ 生成的最小稳定、幂等完备全子范畴。等价地，它们是有限锥、悬挂、有限直和和 retract 后得到的紧型对象。

**答案 27.9.** Pretriangulated dg category 是可表模在 $D(\mathcal A)$ 中对有限稳定操作封闭的 dg category；于是 $H^0(\mathcal A)$ 从 dg 模导出范畴继承三角结构。

**答案 27.10.** 三角范畴的 dg enhancement 是 $H^0(\mathcal A)\simeq T$。稳定 $\infty$-范畴的 dg enhancement 是 $N_{\operatorname{dg}}(\mathcal A)\simeq C$。后者保留映射空间或映射谱的高阶信息，比仅给出 $H^0$ 更强。

**答案 27.11.** dg 函子 $F:\mathcal A\to\mathcal B$ 是 Morita equivalence，若限制函子
$$
F^*:D(\mathcal B)\to D(\mathcal A)
$$
是稳定 $\infty$-范畴等价。

**答案 27.12.** Quasi-equivalence 在可表模之间诱导 quasi-isomorphism，并且 $H^0$ 本质满保证 $\mathcal B$ 的可表模由 $F$ 的像生成。导出模范畴由可表模紧生成，因此 $F^*$ 在紧生成子上是等价，进而在整个导出模范畴上是等价。

**答案 27.13.** $\mathcal A$-$\mathcal B$ dg bimodule 是 dg 函子
$$
M:\mathcal A^{op}\otimes\mathcal B\to\operatorname{Ch}(k).
$$
它也可看成从 $\mathcal A$ 到 $\mathcal B$ 的链复形值 profunctor。

**答案 27.14.** 恒等 bimodule 为 $\mathcal A(-,-)$。对右模 $M$，
$$
M\otimes_{\mathcal A}^{\mathbb L}\mathcal A(-,a)\simeq M(a),
$$
这是富 co-Yoneda 公式的导出版；左单位同理。

**答案 27.15.** 等价 $D(\mathcal A)^\omega\simeq\operatorname{Perf}(\mathcal A)$ 说明大导出模范畴的紧对象完全由可表模经有限稳定操作和 retract 生成。因此 Morita 理论可用小的 perfect 子范畴控制整个 $D(\mathcal A)$。

**答案 27.16.** 把普通代数 $A$ 看成单对象 dg category，则 enveloping 代数为 $A^{op}\otimes A$，恒等 bimodule 是 $A$。因此
$$
HH(A)\simeq A\otimes^{\mathbb L}_{A^{op}\otimes A}A.
$$

**答案 27.17.** 在 dg Yoneda 中取 $M=h_b$：
$$
\underline{\operatorname{Hom}}_{\operatorname{Mod}_{\mathcal A}}(h_a,h_b)\simeq h_b(a)=\mathcal A(a,b).
$$
该 quasi-isomorphism 对 $a,b$ 自然，并与复合相容，因此 Yoneda 嵌入在 Hom 复形上全忠实。

**答案 27.18.** 单对象 dg category 的唯一 Hom 复形 $A=\mathcal A(*,*)$ 带有复合乘法 $A\otimes A\to A$ 和单位 $k\to A$，满足 dg algebra 公理。反过来，dg algebra $A$ 定义一个单对象 dg category，Hom 复形为 $A$，复合为乘法。

**答案 27.19.** 若 Hom 复形集中在 $0$ 次，dg 函子在 Hom 上的链映射就是 $0$ 次 $k$-模同态，并且相容于复合和单位。因此它正是普通 $k$-线性范畴之间的 $k$-线性函子。

## 第二十八章

**答案 28.1.** 稳定系数系统是在有有限拉回的基 $\mathcal B$ 上的反变函子
$$
\mathcal D:\mathcal B^{op}\to\operatorname{CAlg}(\operatorname{Pr}^L_{\operatorname{st}}).
$$
它把 $X$ 送到稳定闭幺半 presentable $\infty$-范畴 $\mathcal D(X)$，把 $f:X\to Y$ 送到拉回 $f^*:\mathcal D(Y)\to\mathcal D(X)$。

**答案 28.2.** Presentable $\infty$-范畴伴随函子定理说，presentable $\infty$-范畴之间保持小余极限的函子是左伴随。因此保持小余极限的 $f^*$ 有右伴随 $f_*$。

**答案 28.3.** 六个操作是
$$
f^*,\quad f_*,\quad f_!,\quad f^!,\quad -\otimes_X-,\quad \underline{\operatorname{Hom}}_X(-,-).
$$

**答案 28.4.** 强对称幺半函子定义中包含结构等价
$$
f^*\mathbb 1_Y\simeq\mathbb 1_X,\qquad
f^*(A\otimes_YB)\simeq f^*A\otimes_Xf^*B,
$$
并要求它们满足单位、结合和对称相干。

**答案 28.5.** 对 Cartesian 方块，先用单位
$$
g^*f_*\to f'_*f'^*g^*f_*,
$$
再用 $f'^*g^*\simeq g'^*f^*$，最后用余单位 $f^*f_*\to\operatorname{id}$，得到
$$
g^*f_*\to f'_*g'^*.
$$

**答案 28.6.** 非常基变换可取为右伴随比较
$$
g'^*f^!\to f'^!g^*
$$
的左伴随转置。即在映射空间等价
$$
\operatorname{Map}(g^*f_!A,B)\simeq\operatorname{Map}(A,f^!g_*B)
$$
与相应方块相干下，得到 $g^*f_!A\to f'_!g'^*A$。

**答案 28.7.** 投影公式是自然等价
$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB.
$$
普通推前版本把 $f_!$ 换为 $f_*$。

**答案 28.8.** $\mathcal D(Y)$-线性正是要求 $f_!$ 与 $\mathcal D(Y)$-作用相容：
$$
f_!(A\otimes_X f^*B)\simeq f_!A\otimes_YB.
$$
这就是投影公式。

**答案 28.9.** 对 $f:X\to Y$、$g:Y\to Z$，
$$
(gf)_!(A\otimes f^*g^*C)
\simeq g_!f_!(A\otimes f^*g^*C)
\simeq g_!(f_!A\otimes g^*C)
\simeq g_!f_!A\otimes C.
$$
最后一项等于 $(gf)_!A\otimes C$。

**答案 28.10.** 若 $f$ proper，则 $f_!\simeq f_*$；拉回 $f'$ 仍 proper 时 $f'_!\simeq f'_*$。非常基变换
$$
g^*f_!\simeq f'_!g'^*
$$
因此识别为普通基变换
$$
g^*f_*\simeq f'_*g'^*.
$$

**答案 28.11.** 对 $j:U\hookrightarrow X$ 和闭补 $i:Z\hookrightarrow X$，recollement 给出
$$
j_!j^*K\to K\to i_*i^*K
$$
和
$$
i_*i^!K\to K\to j_*j^*K.
$$

**答案 28.12.** 若 $j^*K=0$ 且 $i^*K=0$，则第一条 recollement 余纤维序列两端为零：
$$
0\to K\to0.
$$
稳定范畴中这推出 $K=0$，所以 $j^*$ 与 $i^*$ 联合保守。

**答案 28.13.** 若 $p_X:X\to *$，dualizing object 为
$$
\omega_X=p_X^!\mathbb 1_*.
$$
Verdier duality functor 为
$$
\mathbb D_X(K)=\underline{\operatorname{Hom}}_X(K,\omega_X).
$$

**答案 28.14.** 若 $K$ dualizable，则内部 Hom 满足
$$
\underline{\operatorname{Hom}}_X(K,M)\simeq K^\vee\otimes_XM.
$$
取 $M=\omega_X$ 得
$$
\mathbb D_X(K)\simeq K^\vee\otimes_X\omega_X.
$$

**答案 28.15.** Equipment 中 Beck-Chevalley 条件比较“沿方块两条路径先拉后推或先推后拉”的结果。六操作中的
$$
g^*f_*\simeq f'_*g'^*,\qquad g^*f_!\simeq f'_!g'^*
$$
就是这一比较在稳定 presentable $\infty$-范畴值 sheaf 理论中的版本。

**答案 28.16.** 对恒等态射，$\operatorname{id}^*$、$\operatorname{id}_!$ 都是恒等函子。投影公式变为
$$
A\otimes_X\operatorname{id}^*B=A\otimes_XB\simeq A\otimes_XB=\operatorname{id}_!A\otimes_XB,
$$
因此是恒等同构。

**答案 28.17.** 若 $f,g$ proper，则 $f_!\simeq f_*$、$g_!\simeq g_*$. 由复合相干
$$
(gf)_!\simeq g_!f_!,\qquad (gf)_*\simeq g_*f_*.
$$
代入 proper compatibility 得 $(gf)_!\simeq(gf)_*$。

**答案 28.18.** Recollement 给出余纤维序列
$$
j_!j^*K\to K\to i_*i^*K.
$$
若 $j^*K=0$，则第一项为零，所以序列为
$$
0\to K\to i_*i^*K.
$$
稳定范畴中这推出 $K\simeq i_*i^*K$。

## 第二十九章

**答案 29.1.** Relative category 是一对 $(\mathcal C,W)$，其中 $\mathcal C$ 是普通范畴，$W$ 是含所有对象和恒等态射的宽子范畴；$W$ 中态射称为 weak equivalences。

**答案 29.2.** $\infty$-categorical localization 是函子 $N\mathcal C\to\mathcal C[W^{-1}]$，把 $W$ 送为等价，并满足：对任意 $\infty$-范畴 $D$，从 $\mathcal C[W^{-1}]$ 到 $D$ 的函子等价于从 $N\mathcal C$ 到 $D$ 且把 $W$ 送为等价的函子。

**答案 29.3.** 若 $L$ 与 $L'$ 都满足泛性质，则由 $L$ 的泛性质得 $L\to L'$，由 $L'$ 的泛性质得 $L'\to L$。两个复合预合成到 $N\mathcal C$ 后等于恒等；由全忠实性，复合等价于恒等，所以 $L\simeq L'$。

**答案 29.4.** $W$ saturated，若 $f\in W$ 当且仅当 $f$ 在 ordinary localization $\mathcal C[W^{-1}]_{\operatorname{ord}}$ 中成为同构。等价地，$W$ 已包含所有被局部化强制为等价的态射。

**答案 29.5.** Simplicial category 是 $\mathbf{sSet}$-富范畴：对象成集合，Hom 为单纯集 $\operatorname{Map}_{\mathcal A}(x,y)$，复合为单纯集映射并满足富范畴公理。

**答案 29.6.** Simplicial functor $F:\mathcal A\to\mathcal B$ 是 Dwyer-Kan equivalence，若每个映射单纯集映射是弱同伦等价，且 $\pi_0F:\pi_0\mathcal A\to\pi_0\mathcal B$ 本质满。

**答案 29.7.** 映射单纯集弱等价给出 $\pi_0$ 上 Hom 集同构，因此 $\pi_0F$ 完全忠实；再加本质满，$\pi_0F$ 是范畴等价。

**答案 29.8.** 模型范畴 $\mathcal M$ 的 underlying $\infty$-category 是相对范畴 $(\mathcal M,W_\mathcal M)$ 的 $\infty$-局部化：
$$
\mathcal M_\infty=\mathcal M[W_\mathcal M^{-1}].
$$

**答案 29.9.** 同伦范畴只记录 $\pi_0$ 级 Hom。$\infty$-范畴等价还必须比较高阶映射空间。Quillen 等价通过 cofibrant-fibrant derived mapping spaces 或 hammock localization 比较这些映射空间。

**答案 29.10.** Coherent nerve
$$
N_{\operatorname{hc}}:\mathbf{sCat}\to\mathbf{sSet}
$$
把 simplicial category 转为 quasi-category；当 Hom 为 Kan 复形时，它保留同伦相干复合信息。

**答案 29.11.** 若 Hom 单纯集离散，则同伦相干 $n$-单纯形没有非平凡高维选择，只是普通可复合箭头串。因此 coherent nerve 与普通 nerve 一致。

**答案 29.12.** Simplicial space $X:\Delta^{op}\to\mathcal S$ 满足 Segal 条件，若对 $n\ge2$，
$$
X_n\simeq X_1\times_{X_0}\cdots\times_{X_0}X_1.
$$
这表示 $n$-单纯形由 $n$ 条可复合 $1$-态射控制。

**答案 29.13.** Complete Segal space 是满足 Segal 条件的 simplicial space，且退化映射 $X_0\to X_{\operatorname{eq}}$ 为等价，其中 $X_{\operatorname{eq}}$ 是在同伦范畴中可逆的 $1$-态射空间。

**答案 29.14.** Rezk nerve 把 relative category 送到 complete Segal space 模型，使只给定 weak equivalences 的数据能表示完整的同伦理论。

**答案 29.15.** 若构造由 localization 泛性质刻画，则任何模型中得到的对象都满足同一泛性质。由 localization 的唯一性，不同模型给出等价结果。

**答案 29.16.** 若 $F:(\mathcal C,W)\to(\mathcal D,V)$ 把 $W$ 送入 $V$，则复合
$$
N\mathcal C\to N\mathcal D\to\mathcal D[V^{-1}]
$$
把 $W$ 送为等价。由 $\mathcal C[W^{-1}]$ 的泛性质，它唯一因子化为
$$
\mathcal C[W^{-1}]\to\mathcal D[V^{-1}].
$$
两个因子化若预合成到 $N\mathcal C$ 后相同，则由泛性质中的全忠实性相同，所以选择空间可缩。

**答案 29.17.** 若 $W$ 是所有同构，则任意函子 $N\mathcal C\to D$ 自动把 $W$ 送到等价。故 localization 泛性质给出
$$
\operatorname{Fun}(\mathcal C[W^{-1}],D)\simeq\operatorname{Fun}(N\mathcal C,D)
$$
对所有 $D$ 成立。由 Yoneda 判别，$\mathcal C[W^{-1}]\simeq N\mathcal C$。

**答案 29.18.** 对复合 $\mathcal A\xrightarrow F\mathcal B\xrightarrow G\mathcal C$，同伦范畴上的等价满足 $2$-out-of-$3$。映射空间部分也满足 $2$-out-of-$3$。
唯一需注意的是从 $F$ 和 $GF$ 推出 $G$ 时，要用 $\pi_0F$ 本质满，把 $b,b'\in\mathcal B$ 替换为与某些 $Fx,Fx'$ 等价的对象。
映射空间为 Kan 复形时，与等价对象前后复合给出映射空间弱等价，于是把 $G$ 在 $\operatorname{Map}_{\mathcal B}(b,b')$ 上的判断化为像内对象的判断。
故 Dwyer-Kan equivalences 满足 $2$-out-of-$3$。

## 第三十章

**答案 30.1.** Exact sequence 是正合函子列 $A\to B\to C$，其中 $A\to B$ 全忠实，复合 $A\to C$ 为零，并且 $\operatorname{Kar}(B/A)\to C$ 是等价。

**答案 30.2.** 对稳定全子范畴 $A\subseteq B$，商函子 $B\to B/A$ 按定义把 $A$ 送为零。再作幂等完备化得到 $A\to B\to\operatorname{Kar}(B/A)$，第三个条件成为恒等，因此是 exact sequence。

**答案 30.3.** Flasque 稳定范畴是存在正合自函子 $T:C\to C$，使 $\operatorname{id}_C\oplus T\simeq T$ 的稳定范畴。

**答案 30.4.** 自然等价 $\operatorname{id}_C\oplus T\simeq T$ 经过加性不变量给出
$$
\operatorname{id}_{E(C)}+E(T)=E(T)
$$
作为 $E(C)$ 的自映射等式。稳定范畴的 Hom 是加性群，消去 $E(T)$ 得 $\operatorname{id}_{E(C)}=0$，故 $E(C)\simeq0$。

**答案 30.5.** dg quotient $\mathcal B/\mathcal A$ 是 dg 函子 $q:\mathcal B\to\mathcal B/\mathcal A$，使 $\mathcal A$ 中对象变为零，并且任意把 $\mathcal A$ 送为零的 dg 函子唯一地经它因子化，唯一性在导出映射空间中理解。

**答案 30.6.** Drinfeld quotient 对每个 $a\in\mathcal A$ 添加次数 $-1$ 的 $\varepsilon_a$，满足 $d\varepsilon_a=\operatorname{id}_a$。所以 $\operatorname{id}_a$ 在 Hom 复形中是边界，$H^0$ 中为零；对象 $a$ 因而成为零对象。

**答案 30.7.** Additive invariant 是保持滤过余极限且把 split-exact sequences 送为直和分解的函子
$$
E:\operatorname{Cat}^{\operatorname{perf}}_\infty\to\mathcal D.
$$

**答案 30.8.** Localizing invariant 是保持滤过余极限且把 exact sequences
$$
A\to B\to C
$$
送为纤维序列 $E(A)\to E(B)\to E(C)$ 的函子。

**答案 30.9.** Split-exact sequence 是 exact sequence 的特殊情形。Localizing invariant 把它送为纤维序列；由于序列 split，该纤维序列 split，得到直和分解。因此它是 additive invariant。

**答案 30.10.** 对 exact sequence $A\to B\to C$，非连通代数 $K$-理论给出谱纤维序列
$$
K(A)\to K(B)\to K(C).
$$

**答案 30.11.** $\operatorname{Mot}_{\operatorname{add}}$ 配有 $U_{\operatorname{add}}$，使左伴随 $\operatorname{Mot}_{\operatorname{add}}\to\mathcal D$ 等价于取值于 $\mathcal D$ 的 additive invariants。

**答案 30.12.** $\operatorname{Mot}_{\operatorname{loc}}$ 配有 $U_{\operatorname{loc}}$，使左伴随 $\operatorname{Mot}_{\operatorname{loc}}\to\mathcal D$ 等价于取值于 $\mathcal D$ 的 localizing invariants。

**答案 30.13.** 任意 localizing invariant $E$ 因子化为 $\overline E\circ U_{\operatorname{loc}}$。若 $U_{\operatorname{loc}}(A)\simeq U_{\operatorname{loc}}(B)$，应用 $\overline E$ 得 $E(A)\simeq E(B)$。

**答案 30.14.** Hochschild chains 可写为恒等 bimodule 的导出 trace：
$$
HH(\mathcal A)=\mathcal A\otimes^{\mathbb L}_{\mathcal A^{op}\otimes\mathcal A}\mathcal A.
$$

**答案 30.15.** $M_n(R)$ 与 $R$ Morita 等价，因此 $\operatorname{Perf}(M_n(R))\simeq\operatorname{Perf}(R)$。Morita 不变的 $K$ 和 $HH$ 给出
$$
K(M_n(R))\simeq K(R),\qquad HH(M_n(R))\simeq HH(R).
$$

**答案 30.16.** Localizing invariant 把 exact sequence $A\to B\to C$ 送为纤维序列
$$
E(A)\to E(B)\to E(C).
$$
若 $E(A)=0$，则 $E(B)\to E(C)$ 的纤维为零，故该态射是等价。

**答案 30.17.** $A\to A\oplus B\to B$ 是 split-exact sequence。Additive invariant 按定义把它送为直和分解，故
$$
E(A\oplus B)\simeq E(A)\oplus E(B).
$$

**答案 30.18.** Derived Morita equivalence 给出
$$
\operatorname{Perf}(\mathcal A)\simeq\operatorname{Perf}(\mathcal B)
$$
作为小幂等完备稳定 $\infty$-范畴等价。Additive 或 localizing invariant 是这类范畴上的函子，因而把等价对象送到等价对象。

## 第三十一章

**答案 31.1.** $D^b_c(X,\Lambda)$ 是 $D^b(X,\Lambda)$ 中 cohomology sheaves 沿给定有限层化的每个 stratum 局部常值且 stalk 有有限生成同调的对象组成的全子范畴。

**答案 31.2.** 对有限层化按 strata 数归纳。取开 stratum $j:U\hookrightarrow X$ 和闭补 $i:Z\hookrightarrow X$。若所有 stratum restrictions 为零，则 $j^*K=0$ 且 $i^*K=0$。由 recollement 序列 $j_!j^*K\to K\to i_*i^*K$ 得 $K=0$。

**答案 31.3.** 标准 t-结构为
$$
D^{\le0}=\{K\mid H^i(K)=0,\ i>0\},\qquad
D^{\ge0}=\{K\mid H^i(K)=0,\ i<0\}.
$$

**答案 31.4.** Middle perverse t-structure 给出
$$
H^i(i_\alpha^*K)=0\quad(i>-\dim_\mathbb C S_\alpha),
$$
和
$$
H^i(i_\alpha^!K)=0\quad(i<-\dim_\mathbb C S_\alpha).
$$

**答案 31.5.** 一点空间唯一 stratum 的维数为 $0$，且 $i^*=i^!=\operatorname{id}$。Perverse 条件要求 cohomology 只在 $0$ 次非零，因此 heart 是有限维 $\Lambda$-模范畴。

**答案 31.6.** Recollement 下 t-结构粘合指
$$
K\in D^{\le0}(X)\iff j^*K\in D^{\le0}(U),\ i^*K\in D^{\le0}(Z),
$$
且
$$
K\in D^{\ge0}(X)\iff j^*K\in D^{\ge0}(U),\ i^!K\in D^{\ge0}(Z).
$$

**答案 31.7.** $K$ perverse 当且仅当 $j^*K$ perverse，$i^*K\in{}^pD^{\le0}(Z)$，且 $i^!K\in{}^pD^{\ge0}(Z)$。

**答案 31.8.** 中间延拓为
$$
j_{!*}P=\operatorname{im}({}^pj_!P\to{}^pj_*P)
$$
在 perverse heart 中的 image。

**答案 31.9.** 若 $0\ne Q\subseteq j_{!*}P$，则 $j^*Q\subseteq P$。因 $P$ simple，$j^*Q$ 为 $0$ 或 $P$。第一种使 $Q$ 闭支撑，矛盾；第二种使 quotient 闭支撑，也矛盾，故 $Q=j_{!*}P$。

**答案 31.10.** Verdier 对偶交换 perverse t-结构两半：
$$
\mathbb D({}^pD^{\le0})={}^pD^{\ge0},\qquad
\mathbb D({}^pD^{\ge0})={}^pD^{\le0}.
$$
因此它给出 $\operatorname{Perv}(X)^{op}\simeq\operatorname{Perv}(X)$。

**答案 31.11.** Verdier 对偶交换 $j_!$ 与 $j_*$，并把 perverse heart 反等价到自身。因此它把 image
$$
\operatorname{im}({}^pj_!P\to{}^pj_*P)
$$
送到
$$
\operatorname{im}({}^pj_!\mathbb D_UP\to{}^pj_*\mathbb D_UP),
$$
即 $j_{!*}(\mathbb D_UP)$。

**答案 31.12.** Nearby cycles 描述一般纤维靠近特殊纤维的极限信息；vanishing cycles 描述退化中真正消失或新产生的奇异信息。适当平移后它们保持 perverse sheaves，并通过标准三角连接 $i^*$、$\psi_f$ 和 $\phi_f$。

**答案 31.13.** 对 $H\in\operatorname{Perv}(Z)$，recollement 给出 $j^*i_*H=0$ 且 $i^*i_*H=i^!i_*H=H$，所以粘合条件说明 $i_*H$ perverse。若 $K\in\operatorname{Perv}(X)$ 且 $j^*K=0$，则三角
$$
j_!j^*K\to K\to i_*i^*K
$$
给出 $K\simeq i_*i^*K$。再由 $i^!K\simeq i^*K$ 和粘合条件可知 $i^*K\in\operatorname{Perv}(Z)$。因此闭支撑 perverse sheaves 正是 $i_*\operatorname{Perv}(Z)$。

## 第三十二章

**答案 32.1.** 谱 $E$ 的 Bousfield class 是
$$
\langle E\rangle=\{X\in\mathbf{Sp}\mid E\wedge X\simeq0\}.
$$
它记录 $E$ 看不见的谱。

**答案 32.2.** 若 $E\simeq F\wedge G$ 且 $F\wedge X\simeq0$，则
$$
E\wedge X\simeq F\wedge G\wedge X\simeq0.
$$
所以所有 $F$-acyclic 都是 $E$-acyclic，$\langle E\rangle\le\langle F\rangle$。

**答案 32.3.** 因为
$$
(\bigvee_iE_i)\wedge X\simeq\bigvee_i(E_i\wedge X),
$$
左边为零当且仅当每个 $E_i\wedge X$ 为零。因此楔和的 acyclics 是各 acyclics 的交，对应 Bousfield classes 的 join。

**答案 32.4.** 固定素数 $p$，
$$
K(n)_*\cong\mathbb F_p[v_n^{\pm1}],\qquad |v_n|=2(p^n-1).
$$

**答案 32.5.** 有限 $p$-local 谱 $F$ 的 chromatic type 为 $n$，若 $K(m)_*F=0$ 对 $m<n$，且 $K(n)_*F\ne0$。

**答案 32.6.** 厚子范畴定理说，$p$-local finite spectra 的 thick subcategories 由 chromatic type 分类，形成按 type 排列的链。

**答案 32.7.** $v_n$-self map 是 $\Sigma^dF\to F$，在 $K(n)_*$ 上为同构并在其他高度上呈幂零行为。其 telescope 是迭代该自映射的余极限。

**答案 32.8.** Telescope conjecture 断言有限局部化 $L_n^f$ 与由 telescope 谱 $T(0),\dots,T(n)$ 给出的 Bousfield localization 一致，等价形式依具体记号而定。

**答案 32.9.** 若猜想成立，则 $L_n^f$ 的 acyclics 由
$$
(T(0)\vee\cdots\vee T(n))\wedge X\simeq0
$$
检测，也就是由每个 $T(i)\wedge X$ 是否为零检测。

**答案 32.10.** Chromatic fracture square 形如
$$
\begin{array}{c}
L_nX\to L_{K(n)}X\\
\downarrow\quad\downarrow\\
L_{n-1}X\to L_{n-1}L_{K(n)}X.
\end{array}
$$
它是同伦拉回方块。

**答案 32.11.** 若 $L_{n-1}X=0$ 且 $L_{K(n)}X=0$，则右下角也为零，拉回为 $0\times_00=0$。所以 $L_nX=0$。反向由投影到两个角得到。

**答案 32.12.** Chromatic localization 是以 $E=K(n)$、$E(n)$、$T(n)$ 或其楔和定义的 $E$-acyclics 为核的 Bousfield localization。

**答案 32.13.** 若 smashing localization 的 acyclics 由 compact objects 生成，则 Neeman-Thomason 型定理描述 quotient 的 compact objects：
$$
(\mathbf{Sp}/A)^\omega\simeq\operatorname{Kar}(\mathbf{Sp}^\omega/A^\omega).
$$
因此有限谱的厚子范畴控制大范畴局部化后的紧对象。

**答案 32.14.** $f:X\to Y$ 是 $E$-equivalence，若
$$
E\wedge\operatorname{cofib}(f)\simeq0.
$$
若 $\langle E\rangle=\langle F\rangle$，则 $\operatorname{cofib}(f)$ 为 $E$-acyclic 当且仅当为 $F$-acyclic，所以两者给出相同的局部等价类。

**答案 32.15.** 若 $\langle E\rangle\le\langle F\rangle$ 且 $f$ 是 $F$-equivalence，则 $\operatorname{cofib}(f)$ 为 $F$-acyclic。由偏序定义，它也是 $E$-acyclic，故 $f$ 是 $E$-equivalence。

## 第三十三章

**答案 33.1.** 对光滑 $X$，$D_X$ 是由 $\mathcal O_X$ 与切向量场 $\mathcal T_X$ 生成的 filtered sheaf of rings，满足 $\xi f-f\xi=\xi(f)$。

**答案 33.2.** 左 $D_X$-module 给出切向量场对 $M$ 的作用，满足 Leibniz 公式，因此等价于 connection；$D_X$ 的 Lie bracket 关系对应曲率为零，即 flatness。反向由 flat connection 延拓出 $D_X$-作用。

**答案 33.3.** 对 coherent $D_X$-module $M$，选 good filtration，取 associated graded 为 $\operatorname{gr}D_X\simeq\operatorname{Sym}\mathcal T_X$ 上的模。其在 $T^*X$ 中的支撑为 $\operatorname{Char}(M)$。

**答案 33.4.** Bernstein inequality 说非零 coherent $D_X$-module 满足 $\dim\operatorname{Char}(M)\ge\dim X$。若等号成立，则 $M$ holonomic。

**答案 33.5.** 点空间上 $D_X=\mathbb C$，所以 regular holonomic $D_X$-modules 就是有限维复向量空间。

**答案 33.6.** 对左 $D_X$-module $M$，
$$
\operatorname{DR}_X(M)=
[M\to\Omega_X^1\otimes M\to\cdots\to\Omega_X^{\dim X}\otimes M][\dim X].
$$

**答案 33.7.** 平凡 connection 给出通常解析 de Rham complex。Poincaré lemma 说明它 quasi-isomorphic 于 $\mathbb C_X$，所以按本章约定
$$
\operatorname{DR}_X(\mathcal O_X)\simeq\mathbb C_X[\dim X].
$$

**答案 33.8.** Riemann-Hilbert correspondence 给出
$$
D^b_{\operatorname{rh}}(D_X)\simeq D^b_c(X,\mathbb C),
$$
并在 heart 层把 regular holonomic $D_X$-modules 对应到 perverse sheaves。

**答案 33.9.** de Rham 或 solution functor 的平移约定使 regular holonomic $D_X$-module 的标准 heart 被送入 perverse t-结构 heart；因此 heart 层得到 regular holonomic modules 与 $\operatorname{Perv}(X)$ 的等价。

**答案 33.10.** Proper $f$ 下 sheaf 侧 $f_!=f_*$。Riemann-Hilbert 等价与六操作相容，所以 $D$-module 的 proper direct image 对应 sheaf 的 direct image。

**答案 33.11.** Kashiwara equivalence 说闭嵌入 $i:Z\hookrightarrow X$ 下，$D_Z$-modules 等价于支撑在 $Z$ 上的 $D_X$-modules。

**答案 33.12.** Recollement 中闭部分由支撑在闭子空间的对象给出。Kashiwara equivalence 正是 $D$-module 理论中把闭部分识别为 $Z$ 上 $D_Z$-modules 的定理。

**答案 33.13.** 设 $\Phi:\mathcal C\simeq\mathcal D$，把
$$
\mathcal C_{\le0}=\Phi^{-1}(\mathcal D_{\le0}),\qquad
\mathcal C_{\ge0}=\Phi^{-1}(\mathcal D_{\ge0})
$$
作为定义。平移闭合、正交性和截断三角都由 $\mathcal D$ 侧通过 $\Phi$ 和准逆运输回来。因此这是一组 t-结构；heart 是两半交，故 $\Phi$ 限制为 heart 等价。

**答案 33.14.** 若 $F\dashv G$，并用等价 $\Phi_X,\Phi_Y$ 共轭得到 $\widetilde F=\Phi_YF\Phi_X^{-1}$、$\widetilde G=\Phi_XG\Phi_Y^{-1}$，则
$$
\operatorname{Map}(\widetilde F A,B)
\simeq
\operatorname{Map}(F\Phi_X^{-1}A,\Phi_Y^{-1}B)
\simeq
\operatorname{Map}(\Phi_X^{-1}A,G\Phi_Y^{-1}B)
\simeq
\operatorname{Map}(A,\widetilde G B).
$$
所以 $\widetilde F\dashv\widetilde G$。Riemann-Hilbert 下六操作相容性的形式部分正是这种伴随和函子结构的等价运输。

## 第三十四章

**答案 34.1.** 常用模型包括 simplicial commutative rings、特征 $0$ 下的非正 commutative dg algebras、connective $E_\infty$-rings。

**答案 34.2.** Derived affine scheme 是 $\operatorname{Spec}A$，其中 $A$ 是 connective $E_\infty$-ring；$\operatorname{dAff}=(\operatorname{CAlg}^{cn})^{op}$。

**答案 34.3.** 普通环 $R$ 给出离散 $E_\infty$-ring $HR$。离散对象之间的映射空间退化为普通环同态集合，因此普通仿射概形全忠实嵌入派生仿射概形。

**答案 34.4.** Prestack 是函子 $F:\operatorname{dAff}^{op}\to\mathcal S$。若它对给定拓扑的覆盖满足 hyperdescent，则称为 derived stack。

**答案 34.5.** 因 $\operatorname{dAff}=(\operatorname{CAlg}^{cn})^{op}$，
$$
\operatorname{Map}_{\operatorname{dAff}}(\operatorname{Spec}B,\operatorname{Spec}A)
\simeq
\operatorname{Map}_{\operatorname{CAlg}^{cn}}(A,B).
$$

**答案 34.6.** 对 $\operatorname{Spec}A$，
$$
\operatorname{QCoh}(\operatorname{Spec}A)=\operatorname{Mod}_A.
$$

**答案 34.7.** 对 derived stack $X$，
$$
\operatorname{QCoh}(X)=\lim_{\operatorname{Spec}A\to X}\operatorname{Mod}_A.
$$
极限沿所有仿射测试对象到 $X$ 的图形取。

**答案 34.8.** 若 $X=\operatorname{Spec}A$，overcategory 有终对象 $\operatorname{Spec}A\to X$，所以极限等于该终对象处的值 $\operatorname{Mod}_A$。

**答案 34.9.** Cotangent complex $L_A$ 表示导子：
$$
\operatorname{Map}_{\operatorname{Mod}_A}(L_A,M)\simeq\operatorname{Der}(A,M).
$$
相对版本 $L_{B/A}$ 表示 $A$-线性导子。

**答案 34.10.** 对 $A\to B\to C$，导子限制给纤维序列
$$
\operatorname{Der}_B(C,M)\to\operatorname{Der}_A(C,M)\to\operatorname{Der}_A(B,M).
$$
用 cotangent complex 表示并对所有 $M$ 应用 Yoneda，得到
$$
C\otimes_BL_{B/A}\to L_{C/A}\to L_{C/B}.
$$

**答案 34.11.** Formal moduli problem 是 $F:\operatorname{Art}_k\to\mathcal S$，满足 $F(k)=*$，并把小拉回方块送为拉回方块。

**答案 34.12.** Lurie-Pridham 定理说，在特征 $0$ 下，formal moduli problems 的 $\infty$-范畴等价于 dg Lie algebras 的合适 $\infty$-范畴。

**答案 34.13.** 点 $x:\operatorname{Spec}k\to X$ 的一阶变形由
$$
\operatorname{Map}_k(x^*L_X,M)
$$
控制。因此切复形是 $x^*L_X$ 的线性对偶，记作 $T_xX$。

**答案 34.14.** $\operatorname{QCoh}$ 适合准凝聚复形和张量几何；$\operatorname{IndCoh}$ 更适合奇异空间、Grothendieck duality、! pullback 和分布型对象。光滑情形二者接近，奇异情形差别关键。

**答案 34.15.** 对任意测试对象 $\operatorname{Spec}T$，
$$
\operatorname{Map}(\operatorname{Spec}T,\operatorname{Spec}(A\otimes_BC))
\simeq
\operatorname{Map}_{\operatorname{CAlg}}(A\otimes_BC,T).
$$
由推出泛性质，右侧等于
$$
\operatorname{Map}(A,T)\times_{\operatorname{Map}(B,T)}\operatorname{Map}(C,T),
$$
这正是到 $\operatorname{Spec}A\times_{\operatorname{Spec}B}\operatorname{Spec}C$ 的映射空间。由 Yoneda 得结论。

**答案 34.16.** 由表示性
$$
\operatorname{Der}_A(B,M)\simeq\operatorname{Map}_{\operatorname{Mod}_B}(L_{B/A},M).
$$
若 $L_{B/A}=0$，所有映射空间可缩。反过来，若右侧对所有 $M$ 可缩，则 $L_{B/A}$ 和零对象表示同一函子；由 Yoneda，$L_{B/A}\simeq0$。

**答案 34.17.** 态射 $f:X\to Y$ 把 $Y$ 上一阶变形拉回为 $X$ 上一阶变形。由 cotangent complex 的表示性，这给出
$$
x^*f^*L_Y\simeq y^*L_Y\to x^*L_X.
$$
对 $k$-module 取线性对偶，得到切复形映射 $T_xX\to T_yY$。

## 第三十五章

**答案 35.1.** Monad 是函子 $T:C\to C$ 连同单位 $\eta:\operatorname{id}\to T$ 和乘法 $\mu:T^2\to T$，满足结合律和单位律的同伦相干形式。

**答案 35.2.** 伴随 $F:C\rightleftarrows D:G$ 产生 monad $GF$；单位来自伴随单位，乘法为 $GFGF\xrightarrow{G\varepsilon F}GF$。

**答案 35.3.** Comparison functor $K:D\to\operatorname{Alg}_{GF}(C)$ 把 $Y$ 送到 $GY$，其 $GF$-作用为 $GFGY\xrightarrow{G\varepsilon_Y}GY$。

**答案 35.4.** Split augmented simplicial object 是带额外退化或 contracting homotopy 的增广单纯对象，使其同伦相干地收缩到增广目标。

**答案 35.5.** 额外退化给出单纯对象到常值增广目标的同伦收缩。几何实现是 colimit，保持该收缩，因此 $|X_\bullet|\simeq X_{-1}$。

**答案 35.6.** Barr-Beck-Lurie 定理说：若 $G$ 保守并保持 $G$-split simplicial objects 的几何实现，则 $D\to\operatorname{Alg}_{GF}(C)$ 是等价。

**答案 35.7.** 若 $D\simeq\operatorname{Alg}_T(C)$，遗忘函子反映等价，因为 algebra morphism 是等价当且仅当其底层 $C$ 中态射是等价。

**答案 35.8.** 自由-遗忘伴随 $A\otimes-:\mathbf{Sp}\rightleftarrows\operatorname{Mod}_A:U$ 的复合 monad 是 $A\otimes-$，其代数正是 $A$-modules。

**答案 35.9.** Functor $F:C\to D$ comonadic，若 $C\to\operatorname{Coalg}_{FG}(D)$ 是等价，其中 $FG$ 是由伴随产生的 comonad。

**答案 35.10.** Cech nerve $U_\bullet$ 是 $U\times_X\cdots\times_XU$ 组成的增广单纯对象。Descent data 是 $U_\bullet$ 上对象及其所有高阶 cocycle 相容。

**答案 35.11.** 若 $f^*$ comonadic，则 $\mathcal D(X)$ 等价于 comonad $f^*f_*$ 的 coalgebras。该 coalgebra 范畴由 Cech cobar construction 的 totalization 给出，即 Cech descent data。

**答案 35.12.** 对 faithfully flat $A\to B$，数据是 $B$-模 $M$，同构
$$
B\otimes_A M\simeq M\otimes_A B
$$
在 $B\otimes_AB$ 上，且在 $B\otimes_AB\otimes_AB$ 上满足 cocycle condition。

**答案 35.13.** Comparison functor 把 $Y\in D$ 送为 $GY$，其 $GF$-代数结构由 $GFGY\xrightarrow{G\varepsilon_Y}GY$ 给出。遗忘函子只忘掉这个结构，保留底层对象 $GY$，所以 $U\circ K\simeq G$。

**答案 35.14.** 若 $E:D'\simeq D$ 且 $G:D\to C$ monadic，则 $G\circ E$ 的左伴随是 $E^{-1}F$，产生的 monad 仍等价于 $GF$。Comparison functor $D'\to\operatorname{Alg}_{GF}(C)$ 是 $E$ 后接 $D\simeq\operatorname{Alg}_{GF}(C)$，因此是等价。

**答案 35.15.** 恒等覆盖的 Cech nerve 是常值单纯对象 $X_\bullet=X$。应用系数系统后得到常值 cosimplicial 对象 $\mathcal D(X)$。常值 cosimplicial 对象的 totalization 是其常值项，所以 descent 等价为恒等。

## 第三十六章

**答案 36.1.** Neutral Tannakian category 是刚性 $k$-线性阿贝尔对称幺半范畴 $\mathcal C$，配 faithful exact $k$-线性对称幺半 fiber functor $\omega:\mathcal C\to\operatorname{Vect}_k^{fd}$。

**答案 36.2.** 张量自同构群函子为
$$
\operatorname{Aut}^{\otimes}(\omega)(R)=\operatorname{Aut}^{\otimes}(\omega_R),
$$
其中 $\omega_R$ 是标量扩张后的 fiber functor。

**答案 36.3.** 经典 Tannaka duality 说 $G=\operatorname{Aut}^{\otimes}(\omega)$ 是仿射群概形，且
$$
\mathcal C\simeq\operatorname{Rep}_k^{fd}(G)
$$
作为对称幺半范畴。

**答案 36.4.** Matrix coefficient coalgebra 为
$$
\mathcal O(G)=\int^{X\in\mathcal C}\omega(X)^\vee\otimes\omega(X).
$$

**答案 36.5.** 刚性提供对偶、评价和余评价，使张量自同构在 $X$ 和 $X^\vee$ 上相互决定；矩阵系数因而能控制整个群函子。

**答案 36.6.** 自然映射为
$$
X(A)\to\operatorname{Fun}^{L,\otimes}(\operatorname{QCoh}(X),\operatorname{Mod}_A),
$$
把点 $x:\operatorname{Spec}A\to X$ 送到 pullback $x^*$。

**答案 36.7.** 保小余极限的对称幺半函子 $\operatorname{Mod}_R\to\operatorname{Mod}_A$ 由 $R$ 的像决定，等价于 $E_\infty$-ring map $R\to A$。对应函子为 $-\otimes_RA$。

**答案 36.8.** 若 $\operatorname{QCoh}(X)\simeq\operatorname{Tot}\operatorname{QCoh}(U_\bullet)$，则从 $\operatorname{QCoh}(X)$ 到 $\operatorname{Mod}_A$ 的张量函子等价于 $U_\bullet$ 上相容的张量函子，即点的 descent data。

**答案 36.9.** Classifying stack $BG$ 把 $A$ 送到 $\operatorname{Spec}A$ 上 $G$-torsors 的空间。

**答案 36.10.** 在适当假设下，
$$
\operatorname{QCoh}(BG)\simeq\operatorname{Rep}(G)
$$
作为对称幺半 presentable $\infty$-范畴；经典有限维部分恢复有限维表示。

**答案 36.11.** Fiber functor 对应基点 $*\to BG$。其张量自同构群是 loop group $\Omega_*BG\simeq G$，所以带 fiber functor 的 $\operatorname{QCoh}(BG)$ 恢复 $G$。

**答案 36.12.** 对每个 $A$，张量等价 $\Phi:\operatorname{QCoh}(X)\simeq\operatorname{QCoh}(Y)$ 给出预合成等价
$$
\operatorname{Fun}^{L,\otimes}_{good}(\operatorname{QCoh}(Y),\operatorname{Mod}_A)
\simeq
\operatorname{Fun}^{L,\otimes}_{good}(\operatorname{QCoh}(X),\operatorname{Mod}_A).
$$
由高阶 Tannaka，这等价于 $Y(A)\simeq X(A)$。对所有 $A$ 自然，故 functor of points 等价，$X\simeq Y$。

**答案 36.13.** 对点 $x:\operatorname{Spec}A\to X$，复合
$$
\operatorname{QCoh}(Y)\xrightarrow{f^*}\operatorname{QCoh}(X)\xrightarrow{x^*}\operatorname{Mod}_A
$$
是 $f\circ x$ 对应的 pullback。高阶 Tannaka 说明这些张量函子恢复所有 $A$-点，因此 $f^*$ 决定 $f$。

## 第三十七章

**答案 37.1.** tt-category 是本质小幂等完备三角范畴，配有精确对称幺半结构 $\otimes$，且张量对每个变量保持三角。

**答案 37.2.** Thick tensor ideal 是 thick subcategory $I\subseteq T$，满足若 $x\in I$ 且 $t\in T$，则 $x\otimes t\in I$。

**答案 37.3.** 取所有包含给定对象族的 thick tensor ideals 的交。交仍对三角、直和项和张量封闭，因此是最小 thick tensor ideal。

**答案 37.4.** Proper thick tensor ideal $\mathfrak p$ prime，若 $x\otimes y\in\mathfrak p$ 蕴含 $x\in\mathfrak p$ 或 $y\in\mathfrak p$。

**答案 37.5.** $\operatorname{Spc}(T)$ 是 prime thick tensor ideals 的集合。对象 $x$ 的支撑为
$$
\operatorname{supp}(x)=\{\mathfrak p\mid x\notin\mathfrak p\}.
$$

**答案 37.6.** 对 prime $\mathfrak p$，$x\otimes y\notin\mathfrak p$ 当且仅当 $x\notin\mathfrak p$ 且 $y\notin\mathfrak p$。因此
$$
\operatorname{supp}(x\otimes y)=\operatorname{supp}(x)\cap\operatorname{supp}(y).
$$

**答案 37.7.** Thomason subset 是 quasi-compact open subsets 的补的并。Spectral/noetherian 情形中它与常见 specialization-closed 条件相容。

**答案 37.8.** Balmer 分类定理说 rigid tt-category 中 radical thick tensor ideals 与 $\operatorname{Spc}(T)$ 的 Thomason subsets 对应，$I$ 送到 $\bigcup_{x\in I}\operatorname{supp}(x)$。

**答案 37.9.** 对交换环 $R$，
$$
\operatorname{Spc}(\operatorname{Perf}(R))\cong\operatorname{Spec}R.
$$
Perfect complex 的支撑对应局部化后非零的素理想集合。

**答案 37.10.** 若 $P$ perfect 且所有 $P_\mathfrak p=0$，则每个 cohomology module 的所有局部化为零。Noetherian 情形中这推出 cohomology modules 为零，故 $P=0$。

**答案 37.11.** 有限 $p$-local spectra 的 thick tensor ideals 由 chromatic type 分类；Balmer primes 与高度层相联系，反映 Morava $K(n)$ 的检测层级。

**答案 37.12.** 若 $I$ 是 tensor ideal，商中被倒置的态射张量任意对象后仍被倒置。因此张量由 Verdier quotient 泛性质下降到 $T/I$。

**答案 37.13.** 若 prime $\mathfrak p$ 不在 $\operatorname{supp}(x)\cup\operatorname{supp}(z)$ 中，则 $x,z\in\mathfrak p$。因为 $\mathfrak p$ 是 triangulated subcategory，三角中两项在 $\mathfrak p$ 蕴含第三项 $y$ 也在 $\mathfrak p$。故 $\mathfrak p\notin\operatorname{supp}(y)$。

**答案 37.14.** 由张量支撑公式，
$$
\operatorname{supp}(x^{\otimes n})
=\operatorname{supp}(x)\cap\cdots\cap\operatorname{supp}(x)
=\operatorname{supp}(x).
$$

**答案 37.15.** 对 prime $\mathfrak q\subset T'$，原像 $F^{-1}(\mathfrak q)$ 对三角、直和项和张量封闭，且不含单位。若 $x\otimes y$ 落入原像，则 $F(x)\otimes F(y)\in\mathfrak q$，由 prime 性得 $F(x)\in\mathfrak q$ 或 $F(y)\in\mathfrak q$。所以原像是 prime。支撑基开集满足逆像公式 $\operatorname{supp}(x)\mapsto\operatorname{supp}(F(x))$，故得到连续映射。

## 第三十八章

**答案 38.1.** $THH(C)$ 是小稳定幂等完备 $\infty$-范畴 $C$ 的谱值 Hochschild trace，即 Morita $(\infty,2)$-范畴中恒等 bimodule 的 trace。

**答案 38.2.** 对 $E_1$-ring $R$，cyclic bar construction 为 $[n]\mapsto R^{\otimes(n+1)}$，几何实现给出 $THH(R)$。

**答案 38.3.** Morita equivalence 识别双模理论和恒等 bimodule，因此识别恒等 bimodule 的 trace，故 $THH$ Morita invariant。

**答案 38.4.** Cyclic bar construction 有循环对称性；这些循环算子组合成 $S^1$-作用。因此 $THH$ 自然是带圆作用的谱。

**答案 38.5.** Cyclotomic spectrum 是带 $\mathbb T$-作用的谱 $X$，并配有 Frobenius 映射 $X\to X^{tC_p}$，满足相干条件。

**答案 38.6.** $THH(C)$ 不仅有圆作用，还自然带 Frobenius/Tate 型 cyclotomic 结构，因此提升为 cyclotomic spectrum。

**答案 38.7.** $p$-complete 约定下，
$$
TC(X;p)=\operatorname{fib}(X^{h\mathbb T}\xrightarrow{\operatorname{can}-\varphi}X^{t\mathbb T}).
$$

**答案 38.8.** $TC$ 是 cyclotomic spectra 上的函子；若 $X\simeq Y$ 为 cyclotomic 等价，则应用 $TC$ 得 $TC(X)\simeq TC(Y)$。

**答案 38.9.** Cyclotomic trace 是自然变换
$$
\operatorname{tr}_{cycl}:K(C)\to TC(C).
$$

**答案 38.10.** Dennis trace 是 $K(C)\to THH(C)$。Cyclotomic trace 可看作 Dennis trace 与 $THH$ 的 cyclotomic refinement 结合后到 $TC$ 的提升。

**答案 38.11.** Dundas-Goodwillie-McCarthy 定理说，在 nilpotent 相对情形中，相对 $K$-理论与相对 $TC$ 的 $p$-完成等价。

**答案 38.12.** Trace methods 先把 $K$-理论映到 Morita/localizing invariant $THH$，再利用圆作用、Tate construction 和 Frobenius 形成 $TC$；这把 $K$-计算转化为更稳定的谱论固定点计算。

**答案 38.13.** 对 exact functor $C\to D$，
$$
K(C,D)=\operatorname{fib}(K(C)\to K(D)),\qquad
TC(C,D)=\operatorname{fib}(TC(C)\to TC(D)).
$$

**答案 38.14.** 自然变换 $K\to TC$ 给出交换方块
$$
\begin{array}{c}
K(C)\to K(D)\\
\downarrow\quad\downarrow\\
TC(C)\to TC(D).
\end{array}
$$
稳定范畴中交换方块诱导纤维之间的映射，因此得到 $K(C,D)\to TC(C,D)$。

**答案 38.15.** $THH$ localizing，故 exact sequence $A\to B\to C$ 给出纤维序列
$$
THH(A)\to THH(B)\to THH(C).
$$
若 $THH(A)=0$，则 $THH(B)\to THH(C)$ 的纤维为零，故为等价。

## 第三十九章

**答案 39.1.** Reduced functor 是满足 $F(0)\simeq0$ 的函子。

**答案 39.2.** $1$-excisive functor 把 homotopy pushout squares 送到 homotopy pullback squares。

**答案 39.3.** 正合函子保持有限余极限，所以把 pushout square 送到 pushout square。稳定目标中 pushout square 等价于 pullback square，因此它 $1$-excisive。

**答案 39.4.** $F$ 为 $n$-excisive，若它把 strongly homotopy cocartesian $(n+1)$-cubes 送到 homotopy cartesian $(n+1)$-cubes。

**答案 39.5.** $P_nF$ 是从 $F$ 到 $n$-excisive functor 的 universal approximation：任意 $F\to G$ 且 $G$ 为 $n$-excisive，唯一因子化经 $P_nF$。

**答案 39.6.** Goodwillie tower 是 $\cdots\to P_nF\to P_{n-1}F\to\cdots$。第 $n$ 层为
$$
D_nF=\operatorname{fib}(P_nF\to P_{n-1}F).
$$

**答案 39.7.** $n$-homogeneous functor 是 $n$-excisive 且 $P_{n-1}F\simeq0$ 的函子。

**答案 39.8.** 对 reduced $F$，
$$
\operatorname{cr}_2F(X,Y)=\operatorname{fib}(F(X\vee Y)\to F(X)\times F(Y)).
$$

**答案 39.9.** 若 $F$ reduced 且 $1$-excisive，则 $X\vee Y$ 的 pushout square 被送到 pullback：
$$
F(X\vee Y)\simeq F(X)\times_{F(0)}F(Y)\simeq F(X)\times F(Y).
$$
因此二重 cross-effect 为零。

**答案 39.10.** $\operatorname{cr}_nF$ 由所有子集楔和形成的立方图全纤维定义。置换变量置换该立方图，因此给出自然 $\Sigma_n$-作用。

**答案 39.11.** 对 spaces 到 spectra 的合适 reduced finitary functor，
$$
D_nF(X)\simeq(\partial_nF\wedge X^{\wedge n})_{h\Sigma_n}.
$$

**答案 39.12.** Goodwillie chain rule 说
$$
\partial_*(F\circ G)\simeq\partial_*F\circ\partial_*G
$$
其中右侧为 symmetric sequences 的 composition product。

**答案 39.13.** Tower 在 $X$ 处收敛，若
$$
F(X)\to\lim_nP_nF(X)
$$
为等价。

## 第四十章

**答案 40.1.** $\operatorname{Sm}_S$ 是 $S$ 上光滑有限型概形范畴；
$$
\mathcal P(\operatorname{Sm}_S)=\operatorname{Fun}(\operatorname{Sm}_S^{op},\mathcal S).
$$

**答案 40.2.** Nisnevich sheaves of spaces 是 $\operatorname{Sm}_S$ 上对 Nisnevich 覆盖满足下降的 space-valued sheaves。

**答案 40.3.** Motivic spaces $\mathbf H(S)$ 是 Nisnevich sheaves of spaces 关于所有 $X\times\mathbb A^1\to X$ 的 accessible localization。

**答案 40.4.** 预层范畴 presentable；Nisnevich sheaf 化是 accessible localization；再作 $\mathbb A^1$ accessible localization 仍 presentable。因此 $\mathbf H(S)$ presentable。

**答案 40.5.** $F$ 为 $\mathbb A^1$-invariant，若对所有 $X$，
$$
F(X)\simeq F(X\times\mathbb A^1).
$$

**答案 40.6.** 局部对象对 $X\times\mathbb A^1\to X$ 取映射空间为等价。由 Yoneda，这等价于 $F(X)\simeq F(X\times\mathbb A^1)$。

**答案 40.7.** Tate sphere 为
$$
T=\mathbb A^1/(\mathbb A^1\setminus0)\simeq S^1\wedge\mathbb G_m.
$$

**答案 40.8.** Stable motivic homotopy category 是
$$
\mathbf{SH}(S)=\operatorname{Sp}_T(\mathbf H_*(S)).
$$

**答案 40.9.** 外部输入是：复点 functor 与 Nisnevich descent、$\mathbb A^1$-局部化和 $T$-稳定化相容。给定该相容性后，由局部化泛性质先得到 motivic spaces 到 spaces 的函子，再由稳定化泛性质得到 $\mathbf{SH}(\mathbb C)\to\mathbf{Sp}$。

**答案 40.10.** Motivic 六操作为
$$
f^*,f_*,f_!,f^!,-\otimes-,\underline{\operatorname{Hom}},
$$
满足基变换、投影公式、proper compatibility、purity 和 localization。

**答案 40.11.** 对开闭分解 $j:U\hookrightarrow X$、$i:Z\hookrightarrow X$，
$$
j_!j^*E\to E\to i_*i^*E
$$
是 motivic localization triangle。

**答案 40.12.** Homotopy purity 说闭嵌入 $i:Z\hookrightarrow X$ 的商
$$
X/(X\setminus Z)
$$
等价于法丛 $N_{Z/X}$ 的 Thom space。

**答案 40.13.** 零向量丛总空间为 $X$，补为空，所以
$$
\operatorname{Th}(0_X)=X/\varnothing\simeq X_+.
$$

**答案 40.14.** 若 $\mathcal G$ 是 compact generators，则 $X=0$ 当且仅当
$$
\operatorname{Map}(\Sigma^mG,X)\simeq *
$$
对所有 $G\in\mathcal G$ 和 $m\in\mathbb Z$ 成立。因为与 $X$ 正交的对象形成 localizing subcategory，若它包含所有生成子就包含整个范畴；于是 $X$ 与自身正交，恒等态射为零。态射是否为等价可对其纤维应用同一判别。Realization functor 是否保守是额外信息，不能仅由 compact generation 推出。

## 第四十一章

**答案 41.1.** $\operatorname{Sub}_{\mathcal C}(X)$ 是所有单态 $U\hookrightarrow X$ 的同构类，按因子化排序。

**答案 41.2.** 态射 $f:X\to Y$ 把子对象 $V\hookrightarrow Y$ 拉回为 $f^*V\hookrightarrow X$。恒等态射的拉回为恒等，复合态射的拉回由 pullback 粘合性质等于迭代拉回，所以得到反变函子。

**答案 41.3.** Regular category 是有有限极限、每个态射有 image factorization，且 regular epimorphisms 在 pullback 下稳定的范畴。

**答案 41.4.** 对 $U\hookrightarrow X$，令 $\exists_f(U)$ 为 $U\to X\xrightarrow fY$ 的 image。则
$$
\exists_f(U)\le V\Longleftrightarrow U\le f^*V,
$$
故 $\exists_f\dashv f^*$。

**答案 41.5.** Cartesian closed category 是有有限积且每个 $-\times A$ 有右伴随 $(-)^A$ 的范畴。

**答案 41.6.** Heyting implication $P\Rightarrow Q$ 由伴随条件
$$
R\le(P\Rightarrow Q)\Longleftrightarrow R\wedge P\le Q
$$
刻画。

**答案 41.7.** Locally Cartesian closed category 是有有限极限且每个 slice $\mathcal C/X$ 都 Cartesian closed 的范畴。

**答案 41.8.** $\Sigma_f(U\to X)=U\to X\xrightarrow fY$。Slice 态射 $\Sigma_fU\to V$ 等价于 $U\to X\times_YV$，故 $\Sigma_f\dashv f^*$。

**答案 41.9.** $\Pi_f$ 是 $f^*$ 的右伴随；在类型论中它把 $X$ 上依赖于 $f$ 的族送到 $Y$ 上的依赖函数类型。

**答案 41.10.** Comprehension category 是 fibration $p:\mathcal T\to\mathcal C$ 配到箭头范畴的 comprehension 函子，解释上下文、类型和上下文扩张。

**答案 41.11.** 类型 $A$ 在上下文 $\Gamma$ 中给出投影 $\Gamma.A\to\Gamma$；项就是选择每个上下文元素上的一个元素，即 section $\Gamma\to\Gamma.A$。

**答案 41.12.** 在 groupoid 语义中，$x$ 与 $y$ 的恒等证据是从 $x$ 到 $y$ 的可逆箭头；路径对象为箭头对象，恒等箭头给出对角线因子化。

**答案 41.13.** Univalent universe 是 fibration $p:\mathcal U_\bullet\to\mathcal U$，使恒等类型 $\operatorname{Id}_{\mathcal U}(A,B)$ 与等价类型 $\operatorname{Equiv}(A,B)$ 等价。

**答案 41.14.** 几何态射逆像 $f^*$ 保持有限极限且作为左伴随保持小余极限；几何逻辑只用有限合取、任意析取和存在量词，因此解释被 $f^*$ 保持。

## 第四十二章

**答案 42.1.** $\operatorname{Disk}_n$ 是由有限个 $\mathbb R^n$ 不交并组成、态射为空间化嵌入、幺半结构为不交并的对称幺半 $\infty$-范畴。

**答案 42.2.** $E_n$-代数是对称幺半函子 $A:\operatorname{Disk}_n\to C$。

**答案 42.3.** 因为 $A$ 对称幺半，
$$
A\left(\bigsqcup_I\mathbb R^n\right)\simeq\bigotimes_I A(\mathbb R^n).
$$

**答案 42.4.** 因子化同调定义为
$$
\int_MA=(\operatorname{Lan}_{\operatorname{Disk}_n\hookrightarrow\operatorname{Mfld}_n}A)(M)
\simeq\operatorname*{colim}_{\operatorname{Disk}_{n/M}}A(U).
$$

**答案 42.5.** $\operatorname{id}_{\mathbb R^n}$ 是 $\operatorname{Disk}_{n/\mathbb R^n}$ 的终对象，所以余极限等于 $A(\mathbb R^n)$。

**答案 42.6.** 因子化同调是对称幺半左 Kan 延拓，故保持不交并对应的幺半积：
$$
\int_{M\sqcup N}A\simeq\int_MA\otimes\int_NA.
$$

**答案 42.7.** 若 $M=M_-\cup_{N\times\mathbb R}M_+$，则
$$
\int_MA\simeq
\left(\int_{M_-}A\right)\otimes_{\int_{N\times\mathbb R}A}
\left(\int_{M_+}A\right).
$$

**答案 42.8.** 圆盘值给出局部输入；不交并给出张量；每次 collar-gluing 用 excision 化为相对张量积，所以可递归计算。

**答案 42.9.** 对 $E_1$-代数 $A$，
$$
\int_{S^1}A\simeq HH(A).
$$

**答案 42.10.** $S^1$ 的旋转作用在 $\operatorname{Disk}_{1/S^1}$ 上，因此作用在余极限 $\int_{S^1}A$ 上；经 $\int_{S^1}A\simeq HH(A)$ 得到 $HH(A)$ 的圆作用。

**答案 42.11.** $E_n$-空间 $A$ grouplike，若 $\pi_0(A)$ 在诱导乘法下为群。

**答案 42.12.** 非阿贝尔 Poincare 对偶说，对 grouplike $E_n$-空间 $A$，
$$
\int_MA\simeq\operatorname{Map}_c(M,B^nA).
$$

**答案 42.13.** 当 $M=\mathbb R^n$，右侧为 $\operatorname{Map}_*(S^n,B^nA)\simeq\Omega^nB^nA\simeq A$，与 $\int_{\mathbb R^n}A\simeq A$ 一致。

**答案 42.14.** Factorization algebra 是带有不交开集到大开集的多重乘法并满足 Weiss cosheaf descent 的 prefactorization algebra；$\mathbb R^n$ 上局部常值者与 $E_n$-代数等价。

## 第四十三章

**答案 43.1.** $\operatorname{ProFin}$ 的覆盖族是有限族 $\{S_i\to S\}$，使 $\coprod_iS_i\to S$ 为满射。

**答案 43.2.** Condensed set 是 $\operatorname{ProFin}$ 上的 set-valued sheaf；condensed abelian group 是 abelian group-valued sheaf。

**答案 43.3.** 对集合 $A$，令 $\underline A(S)=\operatorname{Map}_{cts}(S,A_{disc})$。连续映射到离散目标可沿 profinite 满覆盖粘合，因此是 sheaf。

**答案 43.4.** 自然变换 $\underline A\to\underline B$ 由点 $*$ 上的函数 $A\to B$ 决定；任意函数又由复合诱导自然变换，所以全忠实。

**答案 43.5.** 拓扑空间 $T$ 的凝聚化为 $\underline T(S)=\operatorname{Map}_{cts}(S,T)$。

**答案 43.6.** 若 $T$ 离散，则 $\operatorname{Map}_{cts}(S,T)$ 正是离散集合嵌入的定义。

**答案 43.7.** 自由 condensed abelian group $\mathbb Z[X]$ 是遗忘函子 $\operatorname{Cond}(\mathbf{Ab})\to\operatorname{Cond}(\mathbf{Set})$ 的左伴随作用于 $X$。

**答案 43.8.** 它有足够 colimits、生成元，且 filtered colimits exact；因此可进行通常的 Abelian 和 derived 同调代数。

**答案 43.9.** Solidification 是反射性对称幺半局部化
$$
(-)^{\mathrm{solid}}:\operatorname{Cond}(\mathbf{Ab})\to\operatorname{Solid}.
$$

**答案 43.10.** Solid tensor product 定义为
$$
M\otimes^{\mathrm{solid}} N=(M\otimes N)^{\mathrm{solid}}.
$$

**答案 43.11.** $M\otimes^{\mathrm{solid}} N$ 是某对象经 solidification 后的值，故属于 solid objects 的本质像。

**答案 43.12.** 对 solid commutative algebra $A$，solid $A$-module 是 $\operatorname{Solid}$ 中的 $A$-module。

**答案 43.13.** Solidification 是左伴随；导出后仍为左伴随，因此保持小余极限。

**答案 43.14.** 拓扑对象通过 $S\mapsto\operatorname{Map}_{cts}(S,T)$ 变为 sheaf，连续运算变为逐点运算；于是问题进入 condensed sheaves 与 modules 的范畴论框架。

## 第四十四章

**答案 44.1.** 有限极限语法范畴的对象为上下文中的公式 $\{\vec x\mid\varphi\}$，态射为可证唯一存在的函数式关系，并按理论可证等价取商。

**答案 44.2.** 空有限积理论中对象只记录有限变量上下文 $n$；从 $n$ 到 $m$ 的代换等价于函数 $m\to n$，故得到有限集范畴的对偶骨架。

**答案 44.3.** Regular 逻辑允许有限合取和存在量词，并按可证蕴含解释为子对象排序。

**答案 44.4.** Coherent category 是 regular category，并且每个子对象格有有限并且 pullback 保持这些有限并。

**答案 44.5.** 同一对象上公式 $\varphi,\psi$ 的解释为子对象 $U,V$；析取解释为 $U\vee V$。

**答案 44.6.** 分类 topos $\mathcal E_T$ 满足
$$
\operatorname{Geom}(\mathcal F,\mathcal E_T)\simeq\operatorname{Mod}_T(\mathcal F)
$$
自然于 Grothendieck topos $\mathcal F$。

**答案 44.7.** 若两个 topos 都表示 $\operatorname{Mod}_T(-)$，则它们表示同一 2-函子；由 2-Yoneda，在等价意义下唯一。

**答案 44.8.** 泛模型是分类 topos 中对应恒等几何态射的 $T$-模型。

**答案 44.9.** $\mathcal F$ 中模型 $M$ 对应几何态射 $f:\mathcal F\to\mathcal E_T$，且 $M\simeq f^*U_T$。

**答案 44.10.** Tripos 是 $P:\mathcal C^{op}\to\mathbf{Heyt}$，带纤维 Heyting 结构、量词伴随、Beck-Chevalley 和 generic predicate 等结构。

**答案 44.11.** Generic predicate 是 $\Omega$ 上谓词 $\top_\Omega$，使任意谓词由某个 classifying map 拉回得到。

**答案 44.12.** 对 topos $\mathcal E$，取 $P(X)=\operatorname{Sub}(X)$；子对象分类子给 generic predicate，内部 Heyting 结构和量词给 tripos 结构。

**答案 44.13.** Tripos-to-topos 把 tripos 生成 elementary topos；对象可由 partial equivalence relations 表示，态射由功能性关系表示。

**答案 44.14.** 来自 topos 的 tripos 已编码子对象、逻辑和分类子；PER 在 topos 中有有效商，因此 tripos-to-topos 恢复原 topos。

**答案 44.15.** 任一模型 $M$ 由某几何态射 $f:\mathcal F\to\mathcal E_T$ 满足 $M\simeq f^*U_T$ 得到。几何态射的逆像 $f^*$ 保持有限极限和余极限，因此保持几何公式的解释。故泛模型中成立的几何 sequent 拉回后在 $M$ 中成立。

## 第四十五章

**答案 45.1.** 关系 $R:X\nrightarrow Y$ 是子对象 $R\hookrightarrow X\times Y$。

**答案 45.2.** 复合为
$$
S\circ R=\exists_{\pi_{XZ}}\bigl(\pi_{XY}^*R\wedge\pi_{YZ}^*S\bigr)\hookrightarrow X\times Z.
$$

**答案 45.3.** 在集合中该子集由所有 $(x,z)$ 组成，使存在 $y$ 满足 $(x,y)\in R$ 且 $(y,z)\in S$，即通常关系复合。

**答案 45.4.** 态射 $f:X\to Y$ 的图像关系是 $(\operatorname{id}_X,f):X\to X\times Y$ 的 image。

**答案 45.5.** $\Gamma_g\circ\Gamma_f$ 由存在 $y=f(x)$ 且 $z=g(y)$ 描述，等价于 $z=gf(x)$，故为 $\Gamma_{gf}$。

**答案 45.6.** Regular completion 是 regular category $\mathcal C_{\mathrm{reg}}$ 与 lex 函子 $\mathcal C\to\mathcal C_{\mathrm{reg}}$，使
$$
\operatorname{Reg}(\mathcal C_{\mathrm{reg}},\mathcal R)\simeq\operatorname{Lex}(\mathcal C,\mathcal R).
$$

**答案 45.7.** Exact category 是 regular category，其中每个内部等价关系都是某个态射的 kernel pair。

**答案 45.8.** Exact completion 是 exact category $\mathcal C_{\mathrm{ex}}$ 与 regular functor $\mathcal C\to\mathcal C_{\mathrm{ex}}$，泛地把 regular functor 延拓到 exact functor。

**答案 45.9.** 若 $\mathcal C$ 已 exact，则它自身满足 exact completion 的泛性质；由表示对象唯一性，$\mathcal C_{\mathrm{ex}}\simeq\mathcal C$。

**答案 45.10.** 内部等价关系 $R\rightrightarrows X$ effective，若它是某个 $q:X\to Q$ 的 kernel pair。

**答案 45.11.** Exact category 中每个等价关系为 kernel pair；regular epi 和 kernel pair 在 pullback 下稳定，因此商稳定。

**答案 45.12.** Allegory 是 locally posetal 2-category，带关系反向 involution 和 meet/关系演算公理。

**答案 45.13.** 若 $R\hookrightarrow X\times Y$，则 $R^\circ\hookrightarrow Y\times X$ 由乘积交换 $X\times Y\cong Y\times X$ 得到。

**答案 45.14.** 拉回后取交解释 $R(x,y)\wedge S(y,z)$，沿 $X\times Y\times Z\to X\times Z$ 取 image 解释 $\exists y$，所以得到关系复合。

**答案 45.15.** 左单位为
$$
\exists x'\,(x=x'\wedge R(x',y))\Longleftrightarrow R(x,y),
$$
右单位同理。范畴上，对角线给出的等式关系经 pullback 后不改变原关系，沿投影取 image 仍回到 $R$。

**答案 45.16.** $(T\circ S)\circ R$ 与 $T\circ(S\circ R)$ 都解释
$$
\exists y\,\exists z\,(R(x,y)\wedge S(y,z)\wedge T(z,w)).
$$
Regular category 中有限积、pullback、有限交和 image 运算满足 regular 逻辑的替换与存在量词规则，因此两侧给出同一子对象。

## 第四十六章

**答案 46.1.** 典型伴随串为
$$
\Pi\dashv\operatorname{Disc}\dashv\Gamma\dashv\operatorname{Codisc}.
$$

**答案 46.2.** $\Pi$ 取 shape，$\operatorname{Disc}$ 给离散对象，$\Gamma$ 取全局截面，$\operatorname{Codisc}$ 给余离散对象。

**答案 46.3.** 对伴随 $L\dashv R$，$L$ 全忠实当且仅当单位 $\operatorname{id}\to RL$ 为等价；这是伴随的标准判别。

**答案 46.4.** $\int=\operatorname{Disc}\Pi$，$\flat=\operatorname{Disc}\Gamma$，$\sharp=\operatorname{Codisc}\Gamma$。

**答案 46.5.** 若 $\Gamma\operatorname{Disc}\simeq\operatorname{id}$，则
$$
\flat\flat=\operatorname{Disc}\Gamma\operatorname{Disc}\Gamma\simeq\operatorname{Disc}\Gamma=\flat.
$$

**答案 46.6.** Left exact modality 是保持有限极限的反射性局部化，等价于 left exact 幂等 monad。

**答案 46.7.** Pullback 是有限极限；left exact localization 按定义保持有限极限，故保持 pullback。

**答案 46.8.** 恒等类型由对角线和 pullback/path object 结构解释；left exact modality 保持有限极限，因此与恒等类型构造相容。

**答案 46.9.** Modal type theory 是依赖类型论加模态算子 $\bigcirc$ 和单位 $A\to\bigcirc A$，满足反射、幂等和替换相容规则。

**答案 46.10.** Differential cohesive $\infty$-topos 是 cohesive $\infty$-topos，另带 infinitesimal/de Rham 模态以编码无穷小结构。

**答案 46.11.** $X_{\mathrm{dR}}$ 是把无穷小加厚方向局部化后的 de Rham shape。

**答案 46.12.** 若 $X$ 已对无穷小加厚局部，即是该局部化的局部对象，则局部化单位 $X\to X_{\mathrm{dR}}$ 为等价。

**答案 46.13.** 在合适群对象假设下，
$$
H^n(X;A)=\pi_0\operatorname{Map}_{\mathcal H}(X,B^nA).
$$

**答案 46.14.** 由 $\Pi\dashv\operatorname{Disc}$，
$$
\operatorname{Map}_{\mathcal H}(X,\operatorname{Disc}B^nA)
\simeq
\operatorname{Map}_{\mathcal S}(\Pi X,B^nA),
$$
取 $\pi_0$ 得到 shape 上的 cohomology。

**答案 46.15.** 若 $D:K\to\mathcal H$ 是有限图形且各 $D(k)$ 局部，则 left exactness 给出
$$
L(\lim_KD)\simeq\lim_KLD\simeq\lim_KD.
$$
因此 $\lim_KD$ 的局部化单位为等价，局部对象对有限极限封闭。

## 第四十七章

**答案 47.1.** 层化空间是拓扑空间 $X$ 配连续映射 $X\to P$ 到偏序集的 Alexandrov 拓扑，纤维为 strata。

**答案 47.2.** Conically stratified space 是每点邻域同胚于 $\mathbb R^k\times C(L)$ 且层化由链接 $L$ 的层化诱导的层化空间。

**答案 47.3.** Exit path 是路径 $\gamma$，使层标号随时间只能沿偏序增大。

**答案 47.4.** $\operatorname{Exit}(X)$ 的对象为点，$n$-单纯形为保持 exit 条件的 $\Delta^n\to X$。

**答案 47.5.** 单层时 exit 条件自动满足，exit simplicial set 等于 singular complex，即 fundamental $\infty$-groupoid。

**答案 47.6.** 若闭层 $Z$ 小于开层 $U$，路径可从 $Z$ 到 $U$；反向要求 $U\le Z$，不满足。

**答案 47.7.** Constructible sheaf 是在每个 stratum 上限制为 locally constant 的 sheaf。

**答案 47.8.** 对好的 conically stratified space，
$$
\operatorname{Shv}_{cbl}(X;\mathcal S)\simeq\operatorname{Fun}(\operatorname{Exit}(X),\mathcal S).
$$

**答案 47.9.** 单层时 $\operatorname{Exit}(X)\simeq\Pi_\infty(X)$，constructible 即 locally constant，故恢复局部系统分类。

**答案 47.10.** 限制到 $\operatorname{Exit}(U)$ 和 $\operatorname{Exit}(Z)$ 给两部分数据；从 $Z$ 进入 $U$ 的 exit morphisms 给跨层传输映射。

**答案 47.11.** Perverse sheaf 是 constructible derived sheaf 中满足支撑和余支撑条件、落入 perverse heart 的对象。

**答案 47.12.** Exit-path 范畴编码局部系统和跨层 monodromy；perverse 条件是在这些数据上增加同调维数限制。

**答案 47.13.** 层化因子化同调使用层化 $\operatorname{Disk}$-范畴和各层代数系数，并满足层化 excision。

**答案 47.14.** 单层时层化 $\operatorname{Disk}$-范畴就是普通 $\operatorname{Disk}_n$，左 Kan 延拓定义相同，因此恢复普通因子化同调。

**答案 47.15.** 若 exit-simplex 的像落在单个 stratum $X_p$，层标号恒为 $p$，exit 条件自动满足；反过来 $X_p$ 的任意 singular simplex 作为 $X$ 中 simplex 也满足 exit 条件。因此二者 simplices 相同，得到 $X_p$ 的 singular complex。

## 第四十八章

**答案 48.1.** $\operatorname{Alg}_n(C)$ 的对象为 $E_n$-代数，1-态射为 $E_{n-1}$-双模，更高态射递归由更低阶双模和同伦给出。

**答案 48.2.** $n=1$ 时对象为结合代数，1-态射为双模，复合为相对张量积，即普通 Morita bicategory。

**答案 48.3.** Proper 指底层对象有限或可对偶；smooth 指 $A$ 作为 $A^{op}\otimes A$-module 为 perfect。

**答案 48.4.** 稳定线性 Morita 语境中 fully dualizable 对象通常由 smooth 和 proper 这两个有限性条件控制。

**答案 48.5.** 取 $P=k^n$ 与 $Q=(k^n)^*$，有
$$
P\otimes_{M_n(k)}Q\simeq k,\qquad Q\otimes_kP\simeq M_n(k),
$$
故 Morita 等价。

**答案 48.6.** Morita trace 是 Morita $(\infty,2)$-范畴中恒等双模 $\operatorname{id}_A$ 的 trace。

**答案 48.7.** 对合适 $E_1$-代数，
$$
\operatorname{Tr}(\operatorname{id}_A)\simeq HH(A)\simeq\int_{S^1}A.
$$

**答案 48.8.** Morita 等价对象的恒等 1-态射在共轭下对应，trace 不变；由 trace 与 $HH$ 的等价，$HH$ Morita 不变。

**答案 48.9.** Higher Hochschild object 可由
$$
\int_{S^k\times\mathbb R^{n-k}}A
$$
等因子化同调表达。

**答案 48.10.** $k=1,n=1$ 时空间为 $S^1$，故得到 $\int_{S^1}A\simeq HH(A)$。

**答案 48.11.** 增广 $E_n$-代数是 $E_n$-代数 $A$ 配态射 $\epsilon:A\to\mathbb 1$。

**答案 48.12.** $E_n$-Koszul dual 可写为 $A^!=\operatorname{End}_A(\mathbb 1)$，也可由 $n$-重 bar/cobar 构造给出。

**答案 48.13.** 若 $A=\mathbb 1$，则
$$
A^!=\operatorname{End}_{\mathbb 1}(\mathbb 1)\simeq\mathbb 1.
$$

**答案 48.14.** Bar 构造把乘法迭代转成余乘法型结构，cobar 反向恢复乘法；在收敛条件下，局部代数数据由对偶余代数控制。

**答案 48.15.** 自反性由恒等双模 $A:{}_AA_A$ 给出；若 $M:{}_AM_B$ 与 $N:{}_BN_A$ 互逆，则同一数据反向给出对称性；若 $A\sim B$ 与 $B\sim C$ 分别由可逆双模给出，则复合双模的相对张量积给出 $A\sim C$，结合律保证逆双模复合后的评价仍为恒等双模。

## 第四十九章

**答案 49.1.** 预 derivator 是严格 2-函子 $\mathbb D:\mathbf{Cat}^{op}\to\mathbf{CAT}$。

**答案 49.2.** 对 $I\xrightarrow uJ\xrightarrow vK$，2-函子反变给出
$$
(vu)^*=\mathbb D(vu)=\mathbb D(u)\mathbb D(v)=u^*v^*.
$$

**答案 49.3.** 核心公理包括：小 coproducts 送 products、点值联合保守、每个 $u^*$ 有左右伴随 $u_!\dashv u^*\dashv u_*$、Kan 延拓满足同伦点态公式。

**答案 49.4.** $u_!$ 是同伦左 Kan 延拓，$u_*$ 是同伦右 Kan 延拓。

**答案 49.5.** 唯一函子 $u:I\to *$ 的左伴随 $u_!$ 把 $I$-图送到同伦余极限，右伴随 $u_*$ 送到同伦极限。

**答案 49.6.** 左 Kan 延拓点态公式为
$$
j^*u_!X\simeq\operatorname{hocolim}_{(u/j)}X|_{(u/j)}.
$$

**答案 49.7.** 若 $u$ 有拟逆 $v$，则 $u^*v^*\cong(uv)^*\cong\operatorname{id}$ 且 $v^*u^*\cong(vu)^*\cong\operatorname{id}$。

**答案 49.8.** Stable derivator 是 pointed derivator，且 cocartesian squares 与 cartesian squares 一致。

**答案 49.9.** 稳定性定义即说明同伦 pushout square 同时为同伦 pullback square，反之亦然。

**答案 49.10.** 稳定 derivator 的基础范畴 $\mathbb D(*)$ 有典范三角范畴结构。

**答案 49.11.** 对 $\infty$-category $C$，定义
$$
\mathbb D_C(I)=h\operatorname{Fun}(N(I),C).
$$

**答案 49.12.** $N(*)=\Delta^0$，$\operatorname{Fun}(\Delta^0,C)\simeq C$，故 $\mathbb D_C(*)\simeq hC$。

**答案 49.13.** Derivator 公理把图形范畴的 coproduct 送为范畴的 product。空图形是空 coproduct，故其像为空 product 即终范畴；二元 coproduct $I\amalg J$ 的像为二元 product，故 $\mathbb D(I\amalg J)\simeq\mathbb D(I)\times\mathbb D(J)$。

## 第五十章

**答案 50.1.** Groupoid-valued prestack 是伪函子 $F:\mathcal C^{op}\to\mathbf{Grpd}$。

**答案 50.2.** Stack 条件要求对每个覆盖 $U_\bullet\to U$，$F(U)\to\operatorname{Desc}(F,U_\bullet)$ 为 groupoids 的等价。

**答案 50.3.** Set 可视为离散 groupoid；sheaf 的唯一粘合条件正是离散 groupoid 情形的 descent 等价。

**答案 50.4.** Descent datum 为局部对象 $x_i$、重叠同构 $\phi_{ij}:x_i|_{U_{ij}}\to x_j|_{U_{ij}}$，满足三重交 cocycle 条件。

**答案 50.5.** Stack 条件的本质满性给出全局粘合对象，完全忠实性给出唯一性到唯一同构。

**答案 50.6.** $G$-torsor 是带右 $G$-作用、局部同构于 $G$ 正则作用的 sheaf。

**答案 50.7.** $BG(U)$ 是 $U$ 上 $G|_U$-torsors 的 groupoid。

**答案 50.8.** $H^1(U,G)$ 可识别为 $G$-torsors 的同构类，即 $\pi_0BG(U)$。

**答案 50.9.** 平凡群的 torsor 局部且全局均为单点 sheaf，故每个 $BG(U)$ 为终 groupoid。

**答案 50.10.** Gerbe 是局部非空且任意两个局部对象进一步局部同构的 stack；$A$-banded gerbe 还指定 automorphism sheaf 与 $A$ 的相容识别。

**答案 50.11.** $A$-banded gerbes 的等价类由 $H^2(U,A)$ 分类。

**答案 50.12.** 局部平凡化后，重叠比较给 $g_{ij}\in G(U_{ij})$，三重交条件为 $g_{ij}g_{jk}=g_{ik}$。

**答案 50.13.** Higher stack 是满足超下降的 space-valued 或 $\infty$-groupoid-valued sheaf。

**答案 50.14.** Groupoid 逐点取 nerve 得到 1-truncated spaces；groupoid descent 变为 1-truncated space descent。

**答案 50.15.** Stack 条件给 $F(U)\to\operatorname{Desc}(F,U_\bullet)$ 的完全忠实性。相容的局部同构族正是 descent groupoid 中两个 descent data 之间的态射，因此唯一来自全局同构，说明 isomorphism presheaf 满足 sheaf 条件。

## 第五十一章

**答案 51.1.** Descent datum 是 $X\to E$ 配同构 $\pi_1^*X\simeq\pi_2^*X$，满足三重纤维积上的 cocycle 条件。

**答案 51.2.** Descent category 的对象为带 descent datum 的对象，态射为与 datum 相容的态射。

**答案 51.3.** 对 $Y\to B$，$p^*Y=E\times_BY$；在 $E\times_BE$ 上两种拉回均同构于 $E\times_BE\times_BY$，给典范 descent datum。

**答案 51.4.** $p$ effective descent，若 $p^*:\mathcal C_{/B}\to\operatorname{Desc}(p)$ 为等价。

**答案 51.5.** 若 $p$ 是同构，则 slice 拉回是等价，且 descent datum 平凡，因此 $p$ effective descent。

**答案 51.6.** Grothendieck topos 中 epimorphisms 是 effective descent morphisms。

**答案 51.7.** 若 $p^*$ monadic 且 monad 代数范畴等价于 descent category，则 $\mathcal C_{/B}\simeq\operatorname{Desc}(p)$。

**答案 51.8.** 范畴 Galois 结构由反射伴随 $I\dashv H$ 及合适 extension/fibration 类组成，并要求 pullback 稳定和反射相容。

**答案 51.9.** Trivial covering 是由反射子范畴对象拉回得到的 extension；covering 是经某 effective descent morphism 拉回后 trivial 的 extension。

**答案 51.10.** Trivial covering 是 pullback 得到的；再次 pullback 由 pullback 粘合性质仍为同类 pullback。

**答案 51.11.** Normal extension 是 covering $p$，使 $p^*p$ trivial 且 $p$ 满足有效下降。

**答案 51.12.** Galois groupoid 是 kernel pair $E\times_BE\rightrightarrows E$ 在 Galois 结构下反射得到的内部 groupoid。

**答案 51.13.** 合适 Galois 结构中，normal extensions over $B$ 与相应 Galois groupoids 的 actions 等价。

**答案 51.14.** 对有限 Galois 扩张 $L/K$，$L\otimes_KL\cong\prod_{\sigma\in G}L$；descent datum 等价于每个 $\sigma$ 的半线性作用，cocycle 即群作用律。

**答案 51.15.** 同构 $e:E'\simeq E$ 诱导 slice 范畴等价 $\mathcal C_{/E}\simeq\mathcal C_{/E'}$，并把 $p$ 的 descent data 等价地运输为 $pe$ 的 descent data。因此两个比较函子在这些等价下对应，一个为等价当且仅当另一个为等价。

## 第五十二章

**答案 52.1.** 由 $I\xleftarrow{s}E\xrightarrow pB\xrightarrow tJ$ 定义 $P=\Sigma_t\Pi_ps^*:\mathcal C_{/I}\to\mathcal C_{/J}$。

**答案 52.2.** 当 $I=J=1$，$\Pi_p$ 在每个 $b$ 上给 $X^{E_b}$，$\Sigma_t$ 对 $b\in B$ 求和，得 $P(X)=\sum_{b\in B}X^{E_b}$。

**答案 52.3.** Container 是 shapes 集合 $B$ 与 positions 映射 $E\to B$，扩张为 $X\mapsto\sum_{b\in B}X^{E_b}$。

**答案 52.4.** 一元多项式 $1\leftarrow E\to B\to1$ 唯一非平凡数据正是 $E\to B$，即 container。

**答案 52.5.** Species 是函子 $F:\mathbf{FinBij}\to\mathbf{Set}$。

**答案 52.6.** 解析函子为
$$
\widehat F(X)=\sum_{n\ge0}F[n]\times_{\Sigma_n}X^n.
$$

**答案 52.7.** 若 $F[n]=1$，则 $\widehat F(X)=\sum_nX^n/\Sigma_n$，即有限多重集。

**答案 52.8.** $\Sigma_n$ 同时重排结构标号和标签位置；取商即遗忘具体编号。

**答案 52.9.** W-type 是多项式函子 $P$ 的初代数 $\alpha:P(W)\to W$。

**答案 52.10.** $1+X$-代数是点 $1\to A$ 与后继 $A\to A$；初此类代数正是自然数对象。

**答案 52.11.** 多项式单子是多项式函子配单位 $\eta:\operatorname{id}\to P$ 和乘法 $\mu:P^2\to P$，且二者为多项式自然变换。

**答案 52.12.** List functor 的单位给长度 1 列表，乘法拼接列表的列表；空列表和拼接结合律给单子律。

**答案 52.13.** 对 pullback $X\times_ZY$，逐个 shape 有
$$
(X\times_ZY)^{E_b}\cong X^{E_b}\times_{Z^{E_b}}Y^{E_b}.
$$
而 $P(X)\times_{P(Z)}P(Y)$ 中两边元素必须有同一 shape $b$，故它分解为上述 pullback 的不交并，等于 $P(X\times_ZY)$。

## 第五十三章

**答案 53.1.** $\infty$-cosmos 是满足 isofibration、equivalence、cotensor、pullback 等公理的 simplicially enriched category。

**答案 53.2.** Quasi-categories、complete Segal spaces、合适 marked simplicial sets 和 simplicial categories 都给例子。

**答案 53.3.** $\mathcal K_2$ 与 $\mathcal K$ 同对象，Hom category 为 $h\operatorname{map}_{\mathcal K}(A,B)$。

**答案 53.4.** Quasi-category 的同伦范畴以 0-单纯形为对象，1-单纯形按 2-单纯形同伦取商；内角填充保证复合，取商后为普通范畴。

**答案 53.5.** Equivalence 是在 homotopy 2-category $\mathcal K_2$ 中成为等价的态射。

**答案 53.6.** Isofibration 是 $\infty$-cosmos 公理指定的 fibration-like maps，支持等价提升并在 pullback 下稳定。

**答案 53.7.** Isofibration 的 pullback 稳定是 $\infty$-cosmos 公理之一。

**答案 53.8.** $\infty$-cosmos 中 adjunction 是 $\mathcal K_2$ 中的 adjunction，配单位、余单位并满足三角恒等式。

**答案 53.9.** 若 $f\dashv u$ 且 $c$ 表示 $D$ 的 colimit，则
$$
\mathcal K_2(fc,y)\cong\mathcal K_2(c,uy)\cong\lim\mathcal K_2(D-,uy)\cong\lim\mathcal K_2(fD-,y).
$$

**答案 53.10.** Module/profunctor $A\nrightarrow B$ 是从 $A$ 到 $B$ 的 bimodule 型对象，可由 span、comma 或映射对象抽象表示。

**答案 53.11.** 态射 $f:A\to B$ 诱导 representable module $(a,b)\mapsto\operatorname{map}_B(fa,b)$。

**答案 53.12.** $\infty$-cosmos 抽取不同 $\infty$-category 模型共有结构，使伴随、极限、Kan 延拓等定理可跨模型转移。

**答案 53.13.** Equivalence 按定义是在 homotopy 2-category 中为等价。2-category 中等价 1-态射由拟逆和单位余单位给出；拟逆可反向复合，且若两个复合因子中任意两个为等价，第三个也由拟逆复合得到。因此满足 $2$-out-of-$3$。

## 第五十四章

**答案 54.1.** $f\perp g$ 指任意以 $f$ 为左边、$g$ 为右边的交换方块存在唯一对角填充。

**答案 54.2.** 若 $\mathcal S\subseteq\mathcal T$，正交于 $\mathcal T$ 中所有态射当然正交于 $\mathcal S$ 中所有态射，故 ${}^\perp\mathcal T\subseteq{}^\perp\mathcal S$；右正交同理。

**答案 54.3.** 正交因子化系统是态射类 $(\mathcal E,\mathcal M)$，每个态射分解为 $me$，且 $\mathcal E={}^\perp\mathcal M$、$\mathcal M=\mathcal E^\perp$。

**答案 54.4.** 两个分解 $me=m'e'$ 之间由 $e\perp m'$ 与 $e'\perp m$ 得到互逆比较态射；唯一填充保证复合为恒等。

**答案 54.5.** 每个函数分解为满射到像再单射入陪域。满射-单射方块中，填充由选择原像定义，单射保证良定义和唯一。

**答案 54.6.** $X$ 为 $\mathcal S$-局部，若对每个 $s:A\to B$，$\mathcal C(B,X)\to\mathcal C(A,X)$ 为双射。

**答案 54.7.** $s\perp(X\to1)$ 的填充正是把任意 $A\to X$ 唯一延拓为 $B\to X$，等价于上述 Hom 映射双射。

**答案 54.8.** 弱因子化系统是 $(\mathcal L,\mathcal R)$，每个态射分解为 $rl$，且 $\mathcal L$、$\mathcal R$ 由相互提升性质刻画，提升不要求唯一。

**答案 54.9.** 正交提升给唯一填充，特别给存在填充；遗忘唯一性即得弱因子化系统。

**答案 54.10.** 若 $f$ 是 $g\in\mathcal L$ 的 retract，则任意 $f$ 对 $r\in\mathcal R$ 的方块可扩张为 $g$ 的方块，取填充后沿 retract 投回得到填充；故 $f\in\mathcal L$。右类对偶。

**答案 54.11.** 同构与任意态射正交，所以属于两类。若 $e_1,e_2\in\mathcal E$，则对任意 $m\in\mathcal M$ 的提升问题可先用 $e_1\perp m$ 再用 $e_2\perp m$ 逐步唯一填充，故 $e_2e_1\in\mathcal E$；$\mathcal M$ 的复合封闭对偶。

## 第五十五章

**答案 55.1.** Sketch 是小范畴配一族指定锥和指定余锥。

**答案 55.2.** Sketch 在 $\mathcal C$ 中的模型是函子 $\mathcal S\to\mathcal C$，把指定锥送为极限锥、指定余锥送为余极限余锥。

**答案 55.3.** 空 sketch 没有额外条件，因此模型就是所有函子 $\mathcal S\to\mathcal C$，态射为自然变换。

**答案 55.4.** 有限积理论是带有限积的小范畴；模型是保持有限积的函子。

**答案 55.5.** 群对象由 $m:G\times G\to G$、$e:1\to G$、$i:G\to G$ 及群公理交换图组成，这些只用有限积。

**答案 55.6.** Doctrine 指定一类结构及保持该结构的函子，如 finite product、finite limit、regular、coherent、geometric doctrine。

**答案 55.7.** 若 doctrine 更强，则保持其结构的函子自动保持较弱 doctrine 的结构，因此模型类包含关系反向。

**答案 55.8.** Essentially algebraic theory 允许部分运算，其定义域由有限极限条件给出；等价地可由有限极限 sketch 表示。

**答案 55.9.** 小范畴的复合只定义在 $M\times_OM$ 上，这是 pullback 定义域，故是 essentially algebraic。

**答案 55.10.** 用 sorts $O,M$、源靶 $s,t$、恒等 $e$、pullback $M\times_OM$ 和复合 $c$，再加入结合与单位交换图，即得到小范畴的有限极限 sketch。

**答案 55.11.** 小 sketch 在 locally presentable category 中的模型范畴在合适小性条件下 locally presentable。

**答案 55.12.** 模型是满足保持指定锥/余锥条件的函子；模型间态射仍是普通自然变换，所以模型范畴是函子范畴的 full subcategory。

**答案 55.13.** 若 $M:\mathcal S\to\mathcal C$ 把指定锥送为极限锥，且 $F:\mathcal C\to\mathcal D$ 保持这些极限，则 $FM$ 也把它们送为极限锥；指定余锥同理。自然变换后合成仍为自然变换，故得到模型范畴间函子。

## 第五十六章

**答案 56.1.** 幂等态射是满足 $e^2=e$ 的自态射 $e:X\to X$。

**答案 56.2.** 幂等 $e$ 分裂，若存在 $r:X\to Y$、$s:Y\to X$，使 $rs=\operatorname{id}_Y$ 且 $sr=e$。

**答案 56.3.** 两个分裂 $e=sr=s'r'$ 给 $u=r's$ 与 $v=rs'$；计算得 $vu=\operatorname{id}$、$uv=\operatorname{id}$。

**答案 56.4.** $\operatorname{Kar}(\mathcal C)$ 的对象为 $(X,e)$，其中 $e$ 幂等；态射 $f:(X,e)\to(Y,d)$ 满足 $f=dfe$。

**答案 56.5.** 若 $p$ 是 $(X,e)$ 上幂等，则 $(X,p)$ 存在，$p:(X,e)\to(X,p)$ 与 $p:(X,p)\to(X,e)$ 给出分裂。

**答案 56.6.** 从 $(X,\operatorname{id})$ 到 $(Y,\operatorname{id})$ 的态射条件为空条件，故 Hom 集与 $\mathcal C(X,Y)$ 相同。

**答案 56.7.** 幂等完备范畴是每个幂等都分裂的范畴。

**答案 56.8.** 嵌入全忠实；任意 $(X,e)$ 中 $e$ 在 $\mathcal C$ 分裂，所以 $(X,e)$ 同构于某个 $(Y,\operatorname{id})$，故本质满。

**答案 56.9.** 绝对余极限是被所有函子保持的余极限。

**答案 56.10.** 分裂 coequalizer 的泛性质由有限等式和 splitting data 验证；任意函子保持等式与复合，故保持它。

**答案 56.11.** 普通范畴 Cauchy complete，若所有绝对余极限存在；普通小范畴中等价于幂等完备。

**答案 56.12.** Karoubi 包络自由加入所有幂等分裂：到任意幂等完备范畴的函子都唯一延拓到 $\operatorname{Kar}(\mathcal C)$。

**答案 56.13.** $\operatorname{Kar}(\mathcal C)$ 已幂等完备，因此由幂等完备范畴嵌入其 Karoubi 包络为等价，得到
$$
\operatorname{Kar}(\operatorname{Kar}(\mathcal C))\simeq\operatorname{Kar}(\mathcal C).
$$

## 附录 A

**答案 A.1.** $\mathbf{Set}_{\mathcal U}$ 的对象集可取为
$\mathcal U$。若它是 $\mathcal U$-小的，则存在
$x\in\mathcal U$ 与 $\mathcal U$ 双射。由于
$\mathcal P(x)\in\mathcal U$ 且 $\mathcal U$ 传递，有注入
$\mathcal P(x)\hookrightarrow\mathcal U\cong x$，与 Cantor 定理矛盾。

**答案 A.2.** 若 $\mathcal C,\mathcal D$ 为 $\mathcal U$-小，函子由
$\operatorname{Ob}(\mathcal C)\to\operatorname{Ob}(\mathcal D)$
和
$\operatorname{Mor}(\mathcal C)\to\operatorname{Mor}(\mathcal D)$
两函数编码，并受源、靶、恒等和复合等式约束。两个函数集均为
$\mathcal U$-小，满足这些等式的子集仍为 $\mathcal U$-小。

**答案 A.3.** 选择原则用于对每个 $D\in\mathcal D$ 同时选择
$G(D)\in\mathcal C$ 和同构 $F(GD)\cong D$。默认
$\operatorname{Ob}(\mathcal D)$ 为 $\mathcal V$-小集合，所以这是
$\mathcal V$-小族上的选择；后续态射作用由完全忠实性唯一决定，不再需要选择。若对象超出固定 universe 成为真类，则该论证会需要本书没有假设的全局选择。

## 附录 B

**答案 B.1.** 例 $\alpha(0)=0,\alpha(1)=2,\alpha(2)=2$。像为 $\{0,2\}$，先满射 $[2]\to[1]$ 给 $0\mapsto0,1,2\mapsto1$，再单射 $[1]\to[4]$ 给 $0\mapsto0,1\mapsto2$。

**答案 B.2.** 对 $k\in[n-2]$，两边都把 $k$ 送到 $[n]$ 中漏掉 $i$ 和 $j$ 后的第 $k$ 个元素；逐点相等。

**答案 B.3.** $\Lambda_1^3$ 是 $\Delta^3$ 的四个二维面中去掉第 1 个面的并。第 $i$ 个面为漏掉顶点 $i$ 的面，所以缺失面是 $(0,2,3)$；保留的三个面是 $(1,2,3)$、$(0,1,3)$、$(0,1,2)$。

## 附录 C

**答案 C.1.** 设 $(P,p_A,p_B)$ 与 $(Q,q_A,q_B)$ 都是
$A,B$ 的积。由 $Q$ 的泛性质有唯一
$u:P\to Q$ 满足 $q_Au=p_A,q_Bu=p_B$；由 $P$ 的泛性质有唯一
$v:Q\to P$ 满足 $p_Av=q_A,p_Bv=q_B$。于是 $vu$ 与
$\operatorname{id}_P$ 经 $p_A,p_B$ 后相同，故 $vu=\operatorname{id}_P$；
同理 $uv=\operatorname{id}_Q$。任何保持两投影的 $P\to Q$ 都由
$Q$ 的泛性质等于 $u$，所以唯一性只针对保持投影的同构。

**答案 C.2.** 自由群 $F(S)$ 表示函子
$G\mapsto\mathbf{Set}_{\mathcal U}(S,U G)$。表示映射
$$
\mathbf{Grp}(F(S),G)\to\mathbf{Set}_{\mathcal U}(S,UG)
$$
把群同态限制到生成元 $S$；逆映射把集合函数唯一延拓为群同态。对
$h:G\to H$，限制 $h\bar f$ 得 $Uh\circ f$，故该双射对 $G$ 自然。

**答案 C.3.** 双射由 currying 给出：
$$
\mathbf{Set}_{\mathcal U}(X\times A,Y)
\cong\mathbf{Set}_{\mathcal U}(X,Y^A).
$$
它把 $f$ 送到 $\widehat f(x)(a)=f(x,a)$，逆映射把 $g$ 送到
$\check g(x,a)=g(x)(a)$。两式逐点互逆。若 $u:X'\to X$，则
$\widehat{f(u\times\operatorname{id}_A)}=\widehat f\,u$；若
$v:Y\to Y'$，则 $\widehat{vf}=v^A\widehat f$，所以双射对两个变量自然。
单位是 $x\mapsto(a\mapsto(x,a))$ 的转置形式，余单位为评价映射
$Y^A\times A\to Y$。

## 附录 E

**答案 E.1.** $\Delta^0$ 各有一个顶点。join 把第一个顶点放在第二个顶点之前，并添加一条从前者到后者的边，因此得到有两个有序顶点的 $\Delta^1$。

**答案 E.2.** $C_{/x}$ 的对象是映射 $\Delta^0\star\Delta^0\cong\Delta^1\to C$，其在右端顶点为 $x$；因此对象是 $C$ 中所有指向 $x$ 的边 $y\to x$。

**答案 E.3.** Kan complex 对所有 horn $\Lambda_i^n\to\Delta^n$ 有填充；quasi-category 只要求内 horn 填充。因此 Kan complex 自动满足 quasi-category 条件。

**答案 E.4.** $C^\natural$ 标记所有等价边；$C^\sharp$ 标记全部 $1$-单纯形；$C^\flat$ 只标记退化边。三者满足 $C^\flat\subseteq C^\natural\subseteq C^\sharp$。

**答案 E.5.** 映射空间定义说某条 lift 对所有测试对象诱导同伦拉回，即具有同伦泛性质；horn lifting 定义把同一泛性质展开为对所有有限单纯形边界数据的填充条件。模型范畴理论证明二者等价。

**答案 E.6.** marked simplicial set 标记边，即 $1$-单纯形；scaled simplicial set 标记 $2$-单纯形。前者适合记录等价或 Cartesian 边，后者适合记录 $2$-态射层面的可逆性和相干关系。

**答案 E.7.** 它应记录三个对象、两条可复合 $1$-态射、第三条比较用的 $1$-态射，以及一个从某个复合到该比较态射的 $2$-态射；thin 标记说明该 $2$-态射被视为相干等式或可逆比较。

**答案 E.8.** 普通有向图只能记录 $f$ 和 $g$ 两条箭头。伴随还需要单位 $\eta:\operatorname{id}\Rightarrow gf$、余单位 $\varepsilon:fg\Rightarrow\operatorname{id}$ 以及三角恒等式，这些都是 $2$-态射及其关系，因此必须使用 $2$-维数据。
