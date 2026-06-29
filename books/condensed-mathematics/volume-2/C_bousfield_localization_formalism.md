# 附录 C：Bousfield localization 与解析化的形式定理

## C.0 目标

第二卷正文多次使用 localization、solidification 和 analyticization。本附录把这些操作的抽象形式写成可引用的定理。这样做的目的不是把 Scholze 的深层定理重新证明一遍，而是把可形式化的范畴论骨架和真正需要输入的数学内容分开。

本附录采用以下约定：

1. 稳定范畴指带有限极限、有限余极限且 pushout 与 pullback 相容的 stable $\infty$-category，或其同伦三角范畴版本。
2. “可展示稳定范畴”表示 presentable stable category；它有所有小极限、小余极限，并满足可访问性条件。
3. 若读者只使用三角范畴语言，可把“cofiber”读作“cone”，把“fiber sequence”读作“distinguished triangle”。

## C.1 由一族对象定义的局部对象

设 $\mathcal C$ 是可展示稳定范畴，$\mathcal K\subset\mathcal C$ 是一个集合的对象族。

**定义 C.1.1.** 对象 $X\in\mathcal C$ 称为 $\mathcal K$-局部，如果对所有 $K\in\mathcal K$，

$$
\operatorname{Map}_{\mathcal C}(K,X)\simeq *
$$

即同伦映射空间可缩。若使用导出范畴符号，这等价于

$$
R\operatorname{Hom}_{\mathcal C}(K,X)\simeq0.
$$

记 $\mathcal C_{\mathcal K\text{-loc}}\subset\mathcal C$ 为全子范畴。

**命题 C.1.2.** $\mathcal C_{\mathcal K\text{-loc}}$ 对小极限、shift 和扩张封闭。

**证明.** 对固定 $K$，函子 $\operatorname{Map}(K,-)$ 保持极限。因此若 $(X_i)$ 都是局部对象，则

$$
\operatorname{Map}(K,\varprojlim_iX_i)
\simeq
\varprojlim_i\operatorname{Map}(K,X_i)
\simeq *
$$

所以小极限仍局部。shift 的情形来自稳定范畴中

$$
\operatorname{Map}(K,X[1])\simeq\operatorname{Map}(K[-1],X)
$$

以及局部化子范畴对 shift 的封闭性；等价地，$R\operatorname{Hom}(K,X[1])\simeq R\operatorname{Hom}(K,X)[1]$。对 fiber sequence

$$
X'\to X\to X''
$$

应用 $\operatorname{Map}(K,-)$ 得到 fiber sequence。若其中两个映射空间可缩，则第三个也可缩。证毕。

**定理 C.1.3（存在性输入，presentable Bousfield localization）.** 在上述假设下，包含函子

$$
i:\mathcal C_{\mathcal K\text{-loc}}\hookrightarrow\mathcal C
$$

有左伴随

$$
L_{\mathcal K}:\mathcal C\to\mathcal C_{\mathcal K\text{-loc}}.
$$

并且单位映射 $X\to iL_{\mathcal K}X$ 的 cofiber 属于由 $\mathcal K$ 生成的 localizing subcategory。

**证明边界.** 这是可展示范畴中的 Bousfield localization 定理。本书使用其结论，不在第二卷重证。它依赖 adjoint functor theorem、可访问局部化和由集合生成的 localizing subcategory。

## C.2 局部等价与局部化核

**定义 C.2.1.** 态射 $f:X\to Y$ 称为 $\mathcal K$-局部等价，如果对每个 $\mathcal K$-局部对象 $Z$，诱导映射

$$
\operatorname{Map}(Y,Z)\to\operatorname{Map}(X,Z)
$$

为等价。

**命题 C.2.2.** 态射 $f:X\to Y$ 是 $\mathcal K$-局部等价，当且仅当 $L_{\mathcal K}f$ 是等价。

**证明.** 若 $L_{\mathcal K}f$ 是等价，则对任意局部对象 $Z$，

$$
\operatorname{Map}(Y,Z)\simeq
\operatorname{Map}(L_{\mathcal K}Y,Z)
\to
\operatorname{Map}(L_{\mathcal K}X,Z)
\simeq
\operatorname{Map}(X,Z)
$$

是等价。反过来，取 $Z=L_{\mathcal K}X$ 和 $Z=L_{\mathcal K}Y$，由 Yoneda 判别可知 $L_{\mathcal K}f$ 在局部子范畴中诱导所有映射空间等价，因此是等价。证毕。

**命题 C.2.3.** 若 $f:X\to Y$ 的 cofiber 属于由 $\mathcal K$ 生成的 localizing subcategory，则 $f$ 是 $\mathcal K$-局部等价。

**证明.** 设 $C=\operatorname{cofib}(f)$。对局部对象 $Z$，fiber sequence

$$
X\to Y\to C
$$

诱导

$$
\operatorname{Map}(C,Z)\to\operatorname{Map}(Y,Z)\to\operatorname{Map}(X,Z).
$$

若 $C$ 属于 $\langle\mathcal K\rangle_{\operatorname{loc}}$，则由局部对象定义和映射空间把余极限转为极限可得 $\operatorname{Map}(C,Z)\simeq *$。于是中间箭头是等价。证毕。

## C.3 幺半结构的下降

设 $\mathcal C$ 是可展示稳定对称幺半范畴，张量积分别保持小余极限。

**定义 C.3.1.** localizing subcategory $\mathcal N\subset\mathcal C$ 称为张量理想，如果对任意 $N\in\mathcal N$ 和任意 $X\in\mathcal C$，

$$
N\otimes X\in\mathcal N.
$$

**定理 C.3.2（幺半 Bousfield localization）.** 若 $\mathcal N=\langle\mathcal K\rangle_{\operatorname{loc}}$ 是张量理想，则局部范畴 $\mathcal C_{\mathcal K\text{-loc}}$ 上存在唯一的对称幺半结构，使得 localization

$$
L_{\mathcal K}:\mathcal C\to\mathcal C_{\mathcal K\text{-loc}}
$$

成为对称幺半函子。具体地，

$$
X\otimes_{\operatorname{loc}}Y
=
L_{\mathcal K}(iX\otimes iY).
$$

**证明.** 因为 $\mathcal N$ 是张量理想，若 $X\to X'$ 是局部等价，则 $X\otimes Y\to X'\otimes Y$ 仍是局部等价。于是 $\otimes$ 可沿 localization 下降。结合律、交换律和单位约束由 $\mathcal C$ 中相应约束经 $L_{\mathcal K}$ 得到；coherence 条件由原范畴中的 coherence 和 localization 的函子性继承。证毕。

这一定理是 solid 张量积和 analytic 张量积的抽象来源。真正困难的是验证相应的核 $\mathcal N$ 是张量理想。

## C.4 solid 派生范畴的形式定义

令

$$
\mathcal C=D(\mathbf{CondAb}).
$$

对每个 profinite 集合 $S$，有 Dirac 映射

$$
\mathbb Z[\underline S]\to\mathbb Z^\square[S].
$$

记其 cone 为

$$
K_S=\operatorname{Cone}(\mathbb Z[\underline S]\to\mathbb Z^\square[S]).
$$

**定义 C.4.1.** 派生 solid 范畴定义为

$$
D_\square(\mathbb Z)
=
D(\mathbf{CondAb})_{\{K_S\}\text{-loc}}.
$$

也就是说，$C\in D(\mathbf{CondAb})$ 属于 $D_\square(\mathbb Z)$ 当且仅当对所有 profinite $S$，

$$
R\operatorname{Hom}(K_S,C)\simeq0.
$$

**输入定理 C.4.2（Scholze）.** 对上述 $\{K_S\}$，localization 存在，并且其核为张量理想。因此 $D_\square(\mathbb Z)$ 带有对称幺半结构

$$
M\otimes_{\mathbb Z}^{L,\square}N
=
L^\square(M\otimes_{\mathbb Z}^LN).
$$

**本书证明的部分.** 一旦接受输入定理 C.4.2，第二卷中 solid 环、solid 模和相对 solid 张量积只是对称幺半范畴中的代数对象、模对象和相对张量积。

## C.5 analytic ring 的形式定义

设 $A$ 是凝聚交换环，$\mathcal M$ 是在 profinite 或极不连通测试对象上给出的测度对象赋值。对每个测试对象 $S$，有 Dirac 映射

$$
A[\underline S]\to\mathcal M[S].
$$

记

$$
K_S^{\mathcal M}
=
\operatorname{Cone}(A[\underline S]\to\mathcal M[S]).
$$

**定义 C.5.1（本书使用的解析对象判别）.** $C\in D(A)$ 称为 $(A,\mathcal M)$-解析，如果对所有测试对象 $S$，

$$
R\operatorname{Hom}_A(K_S^{\mathcal M},C)\simeq0.
$$

记解析对象全子范畴为

$$
D(A,\mathcal M)\subset D(A).
$$

**输入定理 C.5.2（analytic ring 条件）.** 若 $(A,\mathcal M)$ 是 Scholze 意义下的 analytic ring，则：

1. $D(A,\mathcal M)$ 是 $D(A)$ 的反射局部子范畴。
2. 反射函子记为
   $$
   L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M).
   $$
3. localization 核与 $A$-模张量积相容，因此 $D(A,\mathcal M)$ 继承解析张量积
   $$
   M\otimes_{(A,\mathcal M)}^LN
   =
   L_{(A,\mathcal M)}(M\otimes_A^LN).
   $$

**严格性说明.** 若只给出 $A[\underline S]\to\mathcal M[S]$，还不能保证 C.5.2。analytic ring 的技术条件正是为了保证反射存在、局部化核可控、张量积下降和几何局部化相容。

## C.6 解析化的泛性质

**命题 C.6.1.** 设 $(A,\mathcal M)$ 是 analytic ring。对任意 $M\in D(A)$ 和任意解析对象 $C\in D(A,\mathcal M)$，有自然等价

$$
R\operatorname{Hom}_{D(A,\mathcal M)}
(L_{(A,\mathcal M)}M,C)
\simeq
R\operatorname{Hom}_{D(A)}(M,C).
$$

**证明.** 这是左伴随 $L_{(A,\mathcal M)}$ 与包含函子 $i:D(A,\mathcal M)\hookrightarrow D(A)$ 的伴随定义。由于 $i$ 全忠实，左侧可等同于 $D(A)$ 中的

$$
R\operatorname{Hom}_{D(A)}(iL_{(A,\mathcal M)}M,iC).
$$

伴随给出它与 $R\operatorname{Hom}_{D(A)}(M,iC)$ 的自然等价。证毕。

**推论 C.6.2.** 若 $M\to N$ 是 analytic localization 等价，则对任意解析对象 $C$，

$$
R\operatorname{Hom}_A(N,C)\to R\operatorname{Hom}_A(M,C)
$$

是等价。

**证明.** analytic localization 等价表示 $L_{(A,\mathcal M)}M\to L_{(A,\mathcal M)}N$ 是等价。代入命题 C.6.1。证毕。

## C.7 liquid 的位置

$p$-liquid 实向量空间不是 Banach 空间范畴的另一个名字。它是 analytic ring

$$
(\mathbb R,\mathcal M_{<p})
$$

给出的解析模范畴中的对象。严格使用时应写明：

1. 参数 $p$ 或 $<p$。
2. 测试对象 $S$ 的类别。
3. 测度对象 $\mathcal M_{<p}[S]$。
4. Hom 判别所在的范畴。
5. 张量积是否经过 $L_{(\mathbb R,\mathcal M_{<p})}$。

## C.8 本附录小结

第二卷中出现的三类操作有同一个形式骨架：

1. 选一族 cone $K_S$。
2. 定义局部对象为 $R\operatorname{Hom}(K_S,-)=0$。
3. 使用 Bousfield localization 得到反射函子。
4. 验证 localization 核是张量理想。
5. 将普通张量积局部化，得到 solid/analytic/liquid 张量积。

其中第 1、2 步是定义，第 3、4 步通常是深层输入，第 5 步是范畴论推论。
