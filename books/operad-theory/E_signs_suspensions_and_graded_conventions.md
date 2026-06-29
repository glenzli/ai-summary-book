# 附录 E：符号、悬挂与分次约定

本附录固定链复形、张量积、悬挂和 Hochschild 复形中的基本符号。全书采用同调分次：微分次数为 $-1$。若某章使用上同调分次，会在该章单独说明转换。

## E.1 Koszul sign rule

**约定 E.1.** 若 $x,y$ 是齐次元素，其次数记为 $|x|,|y|$。在对称幺半范畴 $\mathbf{Ch}_k$ 中，交换映射
$$
\tau:V\otimes W\to W\otimes V
$$
定义为
$$
\tau(x\otimes y)=(-1)^{|x||y|}y\otimes x.
$$

**命题 E.2.** 交换映射满足 $\tau_{W,V}\tau_{V,W}=\operatorname{id}_{V\otimes W}$。

**证明.** 对齐次张量 $x\otimes y$，
$$
\tau_{W,V}\tau_{V,W}(x\otimes y)
=\tau_{W,V}\big((-1)^{|x||y|}y\otimes x\big)
=(-1)^{|x||y|}(-1)^{|y||x|}x\otimes y.
$$
指数和为 $2|x||y|$，故符号为 $1$。$\square$

**约定 E.3.** 若次数为 $|f|$ 的线性映射 $f:V\to V'$ 与次数为 $|g|$ 的线性映射 $g:W\to W'$ 作用在张量上，则
$$
(f\otimes g)(x\otimes y)=(-1)^{|g||x|}f(x)\otimes g(y).
$$

该约定保证 tensor product of chain maps 与 Koszul braiding 相容。

## E.2 Tensor differential

**定义 E.4.** 若 $(V,d_V)$ 与 $(W,d_W)$ 是链复形，则 $V\otimes W$ 的微分定义为
$$
d_{V\otimes W}(x\otimes y)
=d_Vx\otimes y+(-1)^{|x|}x\otimes d_Wy.
$$

**命题 E.5.** $d_{V\otimes W}^2=0$。

**证明.** 对齐次 $x\otimes y$，
$$
d^2(x\otimes y)
=d(d_Vx\otimes y)+(-1)^{|x|}d(x\otimes d_Wy).
$$
第一项为
$$
d_V^2x\otimes y+(-1)^{|x|-1}d_Vx\otimes d_Wy.
$$
第二项为
$$
(-1)^{|x|}d_Vx\otimes d_Wy
 +(-1)^{|x|}(-1)^{|x|}x\otimes d_W^2y.
$$
由于 $d_V^2=d_W^2=0$，剩余两项的系数为
$$
(-1)^{|x|-1}+(-1)^{|x|}=0.
$$
故 $d^2=0$。$\square$

## E.3 Suspension and desuspension

**定义 E.6.** 悬挂 $sV$ 定义为
$$
(sV)_n=V_{n-1}.
$$
若 $x\in V$ 齐次，则对应元素记为 $sx\in sV$，其次数
$$
|sx|=|x|+1.
$$
微分定义为
$$
d_{sV}(sx)=-s(d_Vx).
$$

去悬挂 $s^{-1}V$ 定义为
$$
(s^{-1}V)_n=V_{n+1},
$$
元素记为 $s^{-1}x$，次数为 $|x|-1$，微分定义为
$$
d_{s^{-1}V}(s^{-1}x)=-s^{-1}(d_Vx).
$$

**命题 E.7.** $sV$ 与 $s^{-1}V$ 都是链复形。

**证明.** 对 $sx$，
$$
d_{sV}^2(sx)=d_{sV}(-s d_Vx)=s d_V^2x=0.
$$
对 $s^{-1}x$ 同理：
$$
d_{s^{-1}V}^2(s^{-1}x)=s^{-1}d_V^2x=0.
$$
$\square$

**约定 E.8.** 迭代悬挂写作 $s^rV$。若 $r$ 为整数，则
$$
|s^rx|=|x|+r.
$$

## E.4 Hom complex

**定义 E.9.** 链复形 $V,W$ 的 Hom complex 定义为
$$
\underline{\operatorname{Hom}}(V,W)_r
=
\prod_p\operatorname{Hom}_k(V_p,W_{p+r}).
$$
若 $f$ 是次数 $r$ 的齐次线性映射，则 Hom differential 为
$$
d(f)=d_W\circ f-(-1)^r f\circ d_V.
$$

**命题 E.10.** Hom differential 满足 $d^2=0$。

**证明.** 设 $|f|=r$。则 $|d(f)|=r-1$。展开：
$$
d^2(f)
=d_W(d_Wf-(-1)^rfd_V)
-(-1)^{r-1}(d_Wf-(-1)^rfd_V)d_V.
$$
由 $d_W^2=d_V^2=0$，只剩
$$
-(-1)^r d_W f d_V-(-1)^{r-1}d_W f d_V.
$$
两项系数相加为 $0$，故 $d^2(f)=0$。$\square$

## E.5 Operadic suspension

**定义 E.11.** 对 dg symmetric sequence $M$，operadic suspension $\Lambda M$ 定义为
$$
(\Lambda M)(n)=s^{1-n}M(n)\otimes\operatorname{sgn}_n.
$$
其逆 operadic desuspension 为
$$
(\Lambda^{-1}M)(n)=s^{n-1}M(n)\otimes\operatorname{sgn}_n.
$$

**说明 E.12.** 不同文献可能使用 cohomological grading，此时公式中的 $s^{1-n}$ 会表现为相反的位移。本书在同调分次中使用定义 E.11。涉及 $A_\infty/L_\infty$ 的逐项恒等式时，必须先把文献中的 suspension convention 转换到本附录。

**命题 E.13.** 若 $M$ 是 concentrated in arity $n$ 的 dg symmetric sequence，则 operadic suspension 改变总次数 $1-n$ 并张量符号表示。

**证明.** 这是定义 E.11 的直接展开。元素 $m\in M(n)$ 被送到 $s^{1-n}m\otimes\epsilon$，次数变为 $|m|+1-n$；$\Sigma_n$ 作用同时乘以 $\operatorname{sgn}_n$。$\square$

## E.6 Hochschild cochains: graded convention

设 $A$ 是 dg associative algebra。定义
$$
C^n(A,A)=\underline{\operatorname{Hom}}(A^{\otimes n},A).
$$
若 $f\in C^p(A,A)$ 是内部次数 $|f|$ 的齐次 cochain，称 $p$ 为 arity。

**定义 E.14.** 对 $f\in C^p(A,A)$、$g\in C^q(A,A)$，第 $i$ 个插入定义为
$$
(f\circ_i g)(a_1,\ldots,a_{p+q-1})
=
(-1)^{|g|(|a_1|+\cdots+|a_{i-1}|)}
f(a_1,\ldots,a_{i-1},g(a_i,\ldots,a_{i+q-1}),a_{i+q},\ldots).
$$

**说明 E.15.** 定义 E.14 只记录内部次数穿过前面输入时产生的 Koszul 符号。若使用 suspended Hochschild complex，还会出现由 arity suspension 产生的额外符号。本附录 E.7 给出 suspended brace 的统一算法；最终版若展开 graded Deligne conjecture，需要把该算法与所选链级 $E_2$ 模型逐项对照。

**定义 E.16.** Graded cup product 可取为
$$
(f\smile g)(a_1,\ldots,a_{p+q})
=
(-1)^{|g|(|a_1|+\cdots+|a_p|)}
f(a_1,\ldots,a_p)\,g(a_{p+1},\ldots,a_{p+q}).
$$

**警告 E.17.** Hochschild cohomological degree、arity degree、internal chain degree 和 suspended degree 是四个不同计数。Gerstenhaber bracket 的次数取决于使用哪一种总次数。第十一章采用传统 Hochschild cohomology 约定，因此 bracket 在 cohomological grading 中为 $-1$。

## E.7 Suspended Hochschild braces

本节给出本书后续使用的分次 brace 符号算法。核心原则是：brace 符号先在 suspended 输入 $sA$ 上定义，再通过悬挂同构转回 $A$。这样 arity 产生的符号和内部次数产生的 Koszul 符号不会混在一条未解释的公式中。

**定义 E.18.** 设 $A$ 是 dg associative algebra。定义 suspended Hochschild cochains
$$
\widetilde C^p(A,A)
=
\underline{\operatorname{Hom}}\big((sA)^{\otimes p},sA\big).
$$
若 $f:A^{\otimes p}\to A$ 是内部次数 $|f|$ 的齐次 cochain，则它对应的 suspended cochain 记为
$$
\widetilde f=s\circ f\circ (s^{-1})^{\otimes p}.
$$
其次数为
$$
|\widetilde f|=|f|+1-p.
$$

**定义 E.19.** 对 homogeneous
$$
F\in\widetilde C^p(A,A),\qquad
G\in\widetilde C^q(A,A),
$$
定义第 $i$ 个 suspended insertion，其中 $0\le i\le p-1$：
$$
\begin{aligned}
&(F\widetilde\circ_i G)(x_1,\ldots,x_{p+q-1})\\
&=
(-1)^{|G|(|x_1|+\cdots+|x_i|)}
F(x_1,\ldots,x_i,G(x_{i+1},\ldots,x_{i+q}),x_{i+q+1},\ldots,x_{p+q-1}),
\end{aligned}
$$
其中 $x_j\in sA$ 齐次。总 insertion 定义为
$$
F\widetilde\circ G
=
\sum_{i=0}^{p-1}F\widetilde\circ_iG.
$$
所有 arity signs 已包含在 $|\widetilde G|=|g|+1-q$ 中，因此这里不再额外写 $(-1)^{(q-1)i}$。

**定义 E.20.** 设
$$
F\in\widetilde C^p(A,A),\qquad
G_j\in\widetilde C^{q_j}(A,A),\quad 1\le j\le r.
$$
令
$$
N=p-r+\sum_{j=1}^r q_j.
$$
一个插入型由整数
$$
0\le i_1\le i_2\le\cdots\le i_r\le p-r
$$
给出，其中 $i_j$ 表示在 $F$ 中第 $j$ 个插入块之前保留的未插入输入个数。令
$$
b_j=i_j+1+\sum_{t<j}q_t
$$
为 $G_j$ 在总输入序列中的起始位置。定义
$$
\begin{aligned}
&F\{G_1,\ldots,G_r\}(x_1,\ldots,x_N)\\
&=
\sum_{0\le i_1\le\cdots\le i_r\le p-r}
(-1)^{\epsilon(i_\bullet;x_\bullet)}
F(x_1,\ldots,
G_1(x_{b_1},\ldots,x_{b_1+q_1-1}),
\ldots,
G_r(x_{b_r},\ldots,x_{b_r+q_r-1}),
\ldots,x_N),
\end{aligned}
$$
其中省略号表示未被插入块占据的 $x$ 按原顺序放入 $F$。令
$$
c_1<\cdots<c_{p-r}
$$
为没有被任何区间
$$
[b_j,b_j+q_j-1]
$$
占据的输入位置。符号指数定义为
$$
\epsilon(i_\bullet;x_\bullet)
=
\sum_{j=1}^r
|G_j|\big(|x_{c_1}|+\cdots+|x_{c_{i_j}}|\big),
$$
其中当 $i_j=0$ 时对应的和为 $0$。
当 $r=0$ 时规定 $F\{\}=F$。

**命题 E.21.** 定义 E.20 在 $r=1$ 时给出定义 E.19 的总 insertion。

**证明.** 当 $r=1$ 时，$N=p+q_1-1$，插入型由 $0\le i_1\le p-1$ 给出，且 $b_1=i_1+1$。定义 E.20 的符号为
$$
|G_1|(|x_{c_1}|+\cdots+|x_{c_{i_1}}|)
=|G_1|(|x_1|+\cdots+|x_{i_1}|),
$$
这正是定义 E.19 中第 $i_1$ 个 insertion 的符号。对所有 $i_1$ 求和得到总 insertion。$\square$

**命题 E.22.** Suspended braces 满足 brace identity；因而 $\widetilde C^\*(A,A)$ 是 brace algebra。

**证明.** 计算
$$
(F\{F_1,\ldots,F_m\})\{G_1,\ldots,G_n\}.
$$
每一项最终给出一个表达式：若干 $G_j$ 被插入某个 $F_a$ 内部，其余 $G_j$ 被放在 $F$ 的外部空隙中。按这些最终位置重新分类，即得到
$$
\sum
F\{G_1,\ldots,F_1\{G_{i_1},\ldots\},\ldots,F_m\{G_{i_m},\ldots\},\ldots,G_n\}.
$$
符号方面，两边都由同一条 Koszul 规则计算：每个 homogeneous operation 穿过它左边且直接进入外层 operation 的 suspended inputs 时，贡献其次数乘以那些输入次数之和。把两步插入的穿越次数相加，等于一次性按最终表达式插入时的穿越次数。故两边符号一致。$\square$

**定义 E.23.** Suspended Gerstenhaber bracket 定义为
$$
[F,G]_{\operatorname{sus}}
=
F\widetilde\circ G
-(-1)^{|F||G|}G\widetilde\circ F.
$$
转回未悬挂 cochains 后，若 $f\in C^p(A,A)$、$g\in C^q(A,A)$ 内部次数均为 $0$，则
$$
|\widetilde f|=1-p,\qquad |\widetilde g|=1-q,
$$
从而 bracket 的交换符号为
$$
(-1)^{(p-1)(q-1)},
$$
这与第十二章未分次公式一致。

**警告 E.24.** 若直接在 $A$ 上写 unsuspended brace 公式，必须同时记录内部次数、arity suspension 和输入元素次数三类符号。定义 E.18-E.23 给出本书的标准算法：先悬挂，按 suspended formula 计算，再去悬挂。

## E.8 Low-order $A_\infty$ signs

本书正文把 $A_\infty$ 结构定义为 $A_\infty=\Omega\operatorname{Ass}^¡$ 的代数。若展开为 maps
$$
m_n:A^{\otimes n}\to A,\qquad |m_n|=n-2
$$
则低阶关系为：

1. $m_1^2=0$；
2. $m_1m_2=m_2(m_1\otimes 1)+(-1)^{|a|}m_2(1\otimes m_1)$ 在元素 $a\otimes b$ 上展开；
3. $m_2(m_2\otimes1)-m_2(1\otimes m_2)$ 由含 $m_3$ 的边界项同伦控制。

**说明 E.25.** 第 3 条的完整符号依赖采用 suspended 或 unsuspended convention。正文不把 unsuspended 全公式作为定义；需要具体计算时，应先选定一种 convention，并把它与定义 E.11 对齐。

## E.9 本附录小结

本书的默认符号链为：同调分次、微分次数 $-1$、Koszul braiding、张量微分定义 E.4、悬挂定义 E.6、Hom differential 定义 E.9、operadic suspension 定义 E.11、suspended brace 公式定义 E.18-E.23。任何从文献引入的 $A_\infty$、$L_\infty$、brace 或 Hochschild 符号都必须转换到这些约定后再进入核心证明。
