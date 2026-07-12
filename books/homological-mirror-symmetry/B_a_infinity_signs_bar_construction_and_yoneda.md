# 附录 B：$A_\infty$ 符号、bar construction 与 Yoneda embedding

## B.1 Suspension、张量次序与 coderivation

**约定 B.1.** 本附录沿用 $V[1]^i=V^{i+1}$，故齐次元素 $a$ 的
suspension $x=sa$ 满足
$$
|x|=|a|-1.
$$
对可复合态射 $a_i:X_{i-1}\to X_i$，张量固定按
$$
x_d\otimes\cdots\otimes x_1
$$
书写。所有符号都使用 suspended degree $|x_i|$；空和的指数约定为
$0$。

**定义 B.2.** 对一组分次 morphism spaces，按对象匹配的 reduced tensor
coalgebra 是
$$
T^c(s\mathcal A)=\bigoplus_{d\ge1}
\bigoplus_{X_0,\ldots,X_d}
s\operatorname{hom}(X_{d-1},X_d)\otimes\cdots\otimes
s\operatorname{hom}(X_0,X_1).
$$
其 reduced deconcatenation coproduct 在长度 $d$ 的张量上为
$$
\Delta(x_d\otimes\cdots\otimes x_1)
=\sum_{i=1}^{d-1}
(x_d\otimes\cdots\otimes x_{i+1})\otimes
(x_i\otimes\cdots\otimes x_1).
$$

给定次数 $+1$ 的 Taylor components
$$
b_s:(s\mathcal A)^{\otimes s}\longrightarrow s\mathcal A,
\qquad s\ge1,
$$
它们唯一延拓为次数 $+1$ 的 coderivation $b$。本书把延拓公式固定为
$$
\begin{aligned}
b(x_d,\ldots,x_1)
=\sum_{\substack{r+s+t=d\\s\ge1}}
(-1)^{\epsilon(r,s,t)}(&x_d,\ldots,x_{r+s+1},\\
&b_s(x_{r+s},\ldots,x_{r+1}),x_r,\ldots,x_1),
\end{aligned}
\tag{B.1}
$$
其中
$$
\epsilon(r,s,t)=\sum_{j=r+s+1}^{d}|x_j|.
\tag{B.2}
$$
也就是说，次数为 $1$ 的 $b_s$ 越过其左侧的 suspended inputs 时产生
Koszul 符号。Desuspension 约定为
$$
\mu^s=s^{-1}\circ b_s\circ s^{\otimes s};
$$
张量积映射使用标准 Koszul 规则。于是 $\mu^s$ 的次数为 $2-s$。

## B.2 完整 Stasheff 恒等式

**命题 B.3（无曲率的 suspended $A_\infty$ 恒等式）.** 采用约定 B.1--B.2，
$b^2=0$ 当且仅当对每个 $d\ge1$ 和每组齐次可复合 suspended inputs
$x_d,\ldots,x_1$，有
$$
\sum_{\substack{r+s+t=d\\s\ge1}}
(-1)^{\epsilon(r,s,t)}
b_{r+1+t}
(x_d,\ldots,x_{r+s+1},
b_s(x_{r+s},\ldots,x_{r+1}),
x_r,\ldots,x_1)=0.
\tag{B.3}
$$

**证明.** 令 $\pi_1:T^c(s\mathcal A)\to s\mathcal A$ 为长度 $1$ 的投影。
把公式 (B.1) 连续应用两次。要使结果长度为 $1$，第二次作用必须把
第一次作用后剩余的整个张量送到 $s\mathcal A$；选择第一次被合成的
连续区间，恰好由 $r,s,t$ 唯一参数化。第一次的 Koszul 符号就是
(B.2)，所以 $\pi_1b^2$ 正是 (B.3) 左端。Coderivation 由其到 cogenerators
的投影唯一决定，故 $b^2=0$ 等价于所有长度上的 $\pi_1b^2=0$。证毕。

**推论 B.4（低阶恒等式）.** 对齐次 $x_i\in s\mathcal A$，公式 (B.3)
在长度 $1,2,3$ 分别为
$$
b_1b_1(x_1)=0,
\tag{B.4}
$$
$$
b_1b_2(x_2,x_1)+b_2(b_1x_2,x_1)
+(-1)^{|x_2|}b_2(x_2,b_1x_1)=0,
\tag{B.5}
$$
以及
$$
\begin{aligned}
0={}&b_1b_3(x_3,x_2,x_1)
+b_2(b_2(x_3,x_2),x_1)
+(-1)^{|x_3|}b_2(x_3,b_2(x_2,x_1))\\
&+b_3(b_1x_3,x_2,x_1)
+(-1)^{|x_3|}b_3(x_3,b_1x_2,x_1)\\
&+(-1)^{|x_3|+|x_2|}b_3(x_3,x_2,b_1x_1).
\end{aligned}
\tag{B.6}
$$

**证明.** 分别列出 $r+s+t=d$ 的全部三元组并代入 (B.3)。每一项的
指数由该内层运算左侧的 $x_j$ 次数之和给出，没有未指定的符号。证毕。

**反例 B.5（不能删去 suspended degree）.** 若把 (B.5) 的最后一个符号
无条件改成 $+1$，则当 $|x_2|$ 为奇数时得到无符号 Leibniz 规则。这与
复形张量积微分的 Koszul 规则不相容，因此一般 dg category 不再给出
满足该错误恒等式的 $A_\infty$ category。符号不是排版选择，而是
定义的一部分。

## B.3 Curvature 与完备化

**定义 B.6.** Curved 情况使用含长度 $0$ 项并按能量过滤完备化的
coaugmented tensor coalgebra
$$
\widehat T^c(s\mathcal A)=\prod_{d\ge0}(s\mathcal A)^{\widehat\otimes d}.
$$
除 $b_s$（$s\ge1$）外，还给每个对象 $X$ 一个次数 $+1$ 元素
$$
b_0(X)\in s\operatorname{hom}_{\mathcal A}(X,X).
$$
要求各 $b_s$ 对完备、分离的能量过滤连续，并要求整个 Taylor family
过滤局部有限：对每个 valuation quotient 和每个输入，延拓公式中只有
有限多个 Taylor terms 非零。Filtered Fukaya 模型中，这由正能量的
$b_0$ 与 gapped energy 条件保证。于是先在每个 quotient 中计算有限和，
随后才可取逆极限；单独写张量长度的直积并不自动给出收敛性。

本书把 arity-zero desuspension 的额外符号固定为
$$
b_0(X)=-s\mu^0_X,
\qquad\text{即}\qquad
\mu^0_X=-s^{-1}b_0(X).
\tag{B.7a}
$$
这个符号使两个 scalar curvatures 之间的一阶运算满足
target-minus-source 公式；它是 convention，不由 degree 单独决定。
公式 (B.1)--(B.3) 的 curved 版本把指标范围改为
$d\ge0$、$s\ge0$、$r+s+t=d$。当 $s=0$ 时，$b_0(X_r)$ 插在
$x_{r+1}$ 与 $x_r$ 之间，且
$$
\epsilon(r,0,t)=\sum_{j=r+1}^{d}|x_j|.
\tag{B.7b}
$$
完备化只保证 Maurer--Cartan 变形等无穷和收敛；固定外部输入数的
Stasheff 恒等式仍是有限和。

**推论 B.7（curved 零输入与一输入方程）.** 对
$x\in s\operatorname{hom}(X_0,X_1)$，curved 恒等式给出
$$
b_1(b_0(X))=0
\tag{B.8}
$$
以及
$$
b_1b_1(x)+b_2(b_0(X_1),x)
+(-1)^{|x|}b_2(x,b_0(X_0))=0.
\tag{B.9}
$$
特别地，$b_0=0$ 时 $b_1^2=0$。这里不存在未写出的“higher curvature
insertions”；一个外部输入的恒等式只有 (B.9) 中三项。

**证明.** 在允许 $s=0$ 的 (B.3) 中分别取 $d=0$ 与 $d=1$。当 $d=1$、
$s=0$ 时只有 $r=0,1$ 两个插入位置，符号分别为 $(-1)^{|x|}$ 与
$1$。证毕。

## B.4 $A_\infty$ functor

**定义 B.8.** 非弯曲 $A_\infty$ functor 的 suspended 数据是次数 $0$ 的
coalgebra morphism
$$
F:T^c(s\mathcal A)\longrightarrow T^c(s\mathcal B),
\qquad F b_{\mathcal A}=b_{\mathcal B}F.
$$
它由次数 $0$ 的 Taylor components
$f_d:(s\mathcal A)^{\otimes d}\to s\mathcal B$ 决定；desuspension 后的
$F^d$ 次数为 $1-d$。

**命题 B.9（functor 的一、二输入方程）.** 对齐次 $x_2,x_1$，有
$$
f_1b_1^{\mathcal A}=b_1^{\mathcal B}f_1
\tag{B.10}
$$
和
$$
\begin{aligned}
&f_1b_2^{\mathcal A}(x_2,x_1)
+f_2(b_1^{\mathcal A}x_2,x_1)
+(-1)^{|x_2|}f_2(x_2,b_1^{\mathcal A}x_1)\\
&\qquad=b_1^{\mathcal B}f_2(x_2,x_1)
+b_2^{\mathcal B}(f_1x_2,f_1x_1).
\end{aligned}
\tag{B.11}
$$
因此 $F^1$ 是 cochain map，而它保持二元复合的失败由 $F^2$ 给出明确
链同伦。

**证明.** 分别取 $Fb_{\mathcal A}=b_{\mathcal B}F$ 在长度 $1$、$2$
输入上的长度 $1$ 分量。左端使用 (B.1)，右端使用 coalgebra morphism
在长度 $2$ 上的两个分拆 $f_2$ 与 $f_1\otimes f_1$，即得到
(B.10)--(B.11)。证毕。

## B.5 Yoneda

**定义 B.10.** 对小、严格含单位 $A_\infty$ category $\mathcal A$，右
Yoneda module 为
$$
Y_X(-)=\operatorname{hom}_{\mathcal A}(-,X),
$$
其高阶 module maps 由 $\mathcal A$ 的 $\mu^d$ 给出。

**外部输入定理 B.11（$A_\infty$ Yoneda）.** 对小、严格含单位
$k$-线性 $A_\infty$ category，Yoneda functor
$$
Y:\mathcal A\longrightarrow\operatorname{Mod}(\mathcal A)
$$
在每个 morphism complex 上为 quasi-isomorphism，因而
cohomologically fully faithful。

**证明路线（外部输入）.** 对 representable module 的 closed module
morphism，在单位处取值给出逆映射；完整证明还要验证全部高阶 module
方程、同伦和单位退化条件。本书不重建完整 $A_\infty$ module category，
故把该结果作为外部输入。来源定位见 `SOURCES.md` 与
`ONLINE_THEOREM_LOCATOR.md`。

## 本附录小结

本书的全部 $A_\infty$ 符号由 (B.1)--(B.3) 固定。低阶恒等式、curvature
方程和 functor 方程都从同一 suspended 公式推出；后文不得再用未解释的
“$\pm$”承担证明责任。
