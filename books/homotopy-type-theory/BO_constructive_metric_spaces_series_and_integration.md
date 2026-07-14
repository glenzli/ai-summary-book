# 附录 BO：构造性度量空间、级数与积分

在构造性分析中，把距离先当作实数值函数再任意选择误差界，常会隐藏 locatedness 或 choice。这里统一采用有理开球关系
$$
\mathsf B_q(x,y)
\quad(q:\mathbb Q_{>0}),
$$
读作“$x,y$ 的距离小于 $q$”。Cauchy、收敛、连续和压缩映射都直接用这一个关系书写。这样，Banach 不动点证明中的几何级数估计仍是有理数计算，而从 mere completeness 取得极限的合法性可以在命题截断规则内逐步检查。

## BO.1 有理开球预度量

**定义 BO.1（有理开球预度量空间）.** 一个有理开球预度量空间由集合
$X:\mathcal U_i$、命题值关系
$$
\mathsf B:
\mathbb Q_{>0}\to X\to X\to\mathcal U_j
$$
以及以下数据组成；对每个 $q>0$ 和 $x,y,z:X$：

1. 反身性：$\mathsf B_q(x,x)$；
2. 对称性：$\mathsf B_q(x,y)\to\mathsf B_q(y,x)$；
3. 三角性：
   $\mathsf B_q(x,y)\to\mathsf B_r(y,z)\to
   \mathsf B_{q+r}(x,z)$；
4. 单调性：若 $q<r$，则
   $\mathsf B_q(x,y)\to\mathsf B_r(x,y)$；
5. 有限距离：
   $$
   \left\|\sum_{q:\mathbb Q_{>0}}\mathsf B_q(x,y)\right\|;
   $$
6. 分离性：
   $$
   \left(\prod_{q:\mathbb Q_{>0}}\mathsf B_q(x,y)\right)
   \to(x=y).
   $$

每个 $\mathsf B_q(x,y)$ 都要求是命题。有限距离只断言某个有理半径存在，不选择全局距离值；它在不动点唯一性中不可省略。若已有实数值度量 $d$，可取
$\mathsf B_q(x,y)\coloneqq(d(x,y)<q)$，上列公理由实数度量律给出。

**定义 BO.2（Cauchy 序列与极限）.** 对序列 $a:\mathbb N\to X$，定义
$$
\mathsf{isCauchy}(a)
\coloneqq
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{N:\mathbb N}
\prod_{m,n:\mathbb N}
(N\le m)\to(N\le n)\to
\mathsf B_\varepsilon(a_m,a_n)
\right\|.
$$
点 $x:X$ 是 $a$ 的极限，若
$$
\mathsf{Lim}(a,x)
\coloneqq
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{N:\mathbb N}
\prod_{n:\mathbb N}(N\le n)\to
\mathsf B_\varepsilon(a_n,x)
\right\|.
$$

**定义 BO.3（mere sequential completeness）.** $X$ sequentially complete，若
$$
\mathsf{isComplete}(X)
\coloneqq
\prod_{a:\mathbb N\to X}
\mathsf{isCauchy}(a)\to
\left\|\sum_{x:X}\mathsf{Lim}(a,x)\right\|.
$$
完备性只给出“仅仅存在极限”，不是一个任意选择序列极限的算子。

**命题 BO.4（极限唯一）.** 若 $\mathsf{Lim}(a,x)$ 且
$\mathsf{Lim}(a,y)$，则 $x=y$。

**证明（书内证明）.** 固定 $q:\mathbb Q_{>0}$。分别以 $q/2$ 应用两条收敛证明；目标 $\mathsf B_q(x,y)$ 是命题，故可消去其中的命题截断，取得指标 $N_x,N_y$。令
$n\coloneqq\max(N_x,N_y)$。由对称性与三角性，
$$
\mathsf B_{q/2}(x,a_n),
\quad
\mathsf B_{q/2}(a_n,y)
\quad\Longrightarrow\quad
\mathsf B_q(x,y).
$$
这对每个 $q>0$ 成立，分离性给出 $x=y$。$\square$

**推论 BO.5（极限的 mere existence 可消去）.** 对固定序列 $a$，类型
$$
\mathsf{Limit}(a)
\coloneqq
\sum_{x:X}\mathsf{Lim}(a,x)
$$
是命题。因此若 $X$ complete 且 $a$ Cauchy，则可从
$\|\mathsf{Limit}(a)\|$
消去得到实际项 $\mathsf{Limit}(a)$。

**证明.** $\mathsf{Lim}(a,x)$ 是命题，因为它由命题值依赖积与命题截断组成。给定两个极限对 $(x,L_x),(y,L_y)$，命题 BO.4 给出 $p:x=y$；沿 $p$ transport 后的两份极限证明属于同一命题，因而相等。$\Sigma$ 路径刻画给出两对相等，所以 $\mathsf{Limit}(a)$ 是命题。命题截断允许向命题目标消去，故 mere completeness 的输出可在这里去截断。$\square$

这个推论不是一般选择原则：只有因为极限连同其收敛证明组成命题，消去才合法。

## BO.2 连续与一致收敛

设 $(X,\mathsf B^X)$、$(Y,\mathsf B^Y)$ 为有理开球预度量空间。

**定义 BO.6（点态连续）.** 函数 $f:X\to Y$ 在 $x:X$ 连续，若
$$
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{\delta:\mathbb Q_{>0}}
\prod_{y:X}
\mathsf B^X_\delta(x,y)\to
\mathsf B^Y_\varepsilon(fx,fy)
\right\|.
$$
若它在每个 $x$ 连续，则称 $f$ 连续。

**定义 BO.7（一致收敛）.** 函数列 $f_n:X\to Y$ 一致收敛到 $f:X\to Y$，若
$$
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{N:\mathbb N}
\prod_{n:\mathbb N}(N\le n)\to
\prod_{x:X}
\mathsf B^Y_\varepsilon(f_n(x),f(x))
\right\|.
$$

**定理 BO.8（一致极限保持连续）.** 若每个 $f_n$ 连续，且 $f_n$ 一致收敛到 $f$，则 $f$ 连续。

**证明（书内证明）.** 固定 $x:X$ 和 $\varepsilon>0$。连续性目标是命题，所以可对一致收敛在 $\varepsilon/3$ 处的截断消去，取得 $N$，再对 $f_N$ 在 $x$ 的连续性消去，取得适用于 $\varepsilon/3$ 的 $\delta$。若
$\mathsf B^X_\delta(x,y)$，则有
$$
\mathsf B^Y_{\varepsilon/3}(f(x),f_N(x)),
\quad
\mathsf B^Y_{\varepsilon/3}(f_N(x),f_N(y)),
\quad
\mathsf B^Y_{\varepsilon/3}(f_N(y),f(y));
$$
第一式使用一致收敛与对称性。两次三角性给出
$\mathsf B^Y_\varepsilon(f(x),f(y))$。$\square$

## BO.3 级数作为部分和序列

设 $V$ 是带加法群结构的 complete 有理开球预度量空间，并假设球关系平移不变，且满足
$$
\mathsf B_q(u,0)\to\mathsf B_r(v,0)
\to\mathsf B_{q+r}(u+v,0).
$$

**定义 BO.9（级数收敛）.** 对 $a:\mathbb N\to V$，令
$$
S_N\coloneqq\sum_{k<N}a_k.
$$
称 $\sum_ka_k$ 收敛到 $s:V$，若
$\mathsf{Lim}(S,s)$。

**命题 BO.10（级数 Cauchy 判别）.** 在上述完备性假设下，级数有极限，当且仅当
$$
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{N:\mathbb N}
\prod_{m,n:\mathbb N}
(N\le n)\to(n\le m)\to
\mathsf B_\varepsilon
\left(\sum_{k=n}^{m-1}a_k,0\right)
\right\|.
$$

**证明（书内证明）.** 由平移不变性，
$\mathsf B_\varepsilon(S_m,S_n)$
等价于题中 tail sum 与 $0$ 的球关系，所以右侧正是部分和序列的 Cauchy 条件。若级数已有极限，任意收敛序列都是 Cauchy：对 $\varepsilon/2$ 取共同尾部并用三角性。反向由 complete 得
$\|\mathsf{Limit}(S)\|$，再由推论 BO.5 合法消去为实际极限。$\square$

**命题 BO.11（有理 majorant 的 M-test）.** 给定类型
$D:\mathcal U_k$、函数列
$f:\mathbb N\to(D\to V)$ 与有理数列
$M:\mathbb N\to\mathbb Q_{\ge0}$。假设对每个
$n:\mathbb N$、$x:D$ 和 $q:\mathbb Q_{>0}$，若 $M_n<q$，则
$$
\mathsf B_q(f_n(x),0).
$$
再假设 $\sum_nM_n$ 的有理 tails 满足
$$
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{N:\mathbb N}
\prod_{m,n:\mathbb N}
(N\le n)\to(n\le m)\to
\left(\sum_{k=n}^{m-1}M_k<\varepsilon\right)
\right\|.
$$
令
$$
S_N(x)\coloneqq\sum_{k<N}f_k(x).
$$
则函数列 $(S_N)$ 一致 Cauchy。若 $V$ complete，则极限函数总类型
$$
\sum_{s:D\to V}
\prod_{x:D}\mathsf{Lim}(\lambda N.S_N(x),s(x))
$$
可收缩，并且其中心的函数分量 $s$ 是 $(S_N)$ 的一致极限。

**证明（书内证明）.** 对固定 tail，用有限次加法球估计得到：只要
$q_k>M_k$，就有
$$
\mathsf B_{\sum_{k=n}^{m-1}q_k}
\left(\sum_{k=n}^{m-1}f_k(x),0\right).
$$
利用有理数稠密性选择 $q_k$，使其有限和仍小于给定
$\varepsilon$；majorant tail 给出与 $x$ 无关的 $N$。这证明一致 Cauchy。对每个 $x$ 用 BO.10 和 BO.5 取唯一极限。函数外延性与逐点极限唯一性说明上述极限函数总类型可收缩；一致 tail 估计直接给出其中心 $s$ 满足 $(S_N)$ 一致收敛到 $s$。$\square$

## BO.4 Banach 不动点定理

**定义 BO.12（压缩映射）.** 对 $T:X\to X$ 和
$c:\mathbb Q_{>0}$，定义
$$
\mathsf{isContraction}_c(T)
\coloneqq
\prod_{x,y:X}\prod_{q:\mathbb Q_{>0}}
\mathsf B_q(x,y)\to\mathsf B_{cq}(Tx,Ty).
$$
称 $T$ 为压缩映射，若
$$
\sum_{c:\mathbb Q_{>0}}(c<1)\times
\mathsf{isContraction}_c(T)
$$
有项。这个定义全程使用球关系，没有把 $\mathsf B$ 当作可相乘的距离值。

**定义 BO.13（不动点类型）.** 定义
$$
\mathsf{Fix}(T)
\coloneqq
\sum_{x:X}(Tx=x).
$$
因为 $X$ 是集合，第二分量是命题，故 $\mathsf{Fix}(T)$ 也是集合。

**定理 BO.14（Banach 不动点）.** 设 $X$ 是 complete 有理开球预度量空间，并假设它 merely inhabited：
$$
\|X\|.
$$
若 $T:X\to X$ 是压缩映射，则
$$
\mathsf{isContr}(\mathsf{Fix}(T)).
$$
特别地，$T$ 有且只有一个不动点。

**证明（书内证明）.** 结论 $\mathsf{isContr}(\mathsf{Fix}(T))$ 在函数外延性下是命题，所以可先对 $\|X\|$ 消去并固定 $x_0:X$。写压缩常数为 $0<c<1$，递归定义
$$
x_{n+1}\coloneqq T(x_n).
$$
有限距离给出
$$
\left\|\sum_{q_0:\mathbb Q_{>0}}
\mathsf B_{q_0}(x_0,x_1)\right\|.
$$
最终目标仍是命题，故可消去并固定这样的 $q_0$。

由自然数归纳和压缩条件，
$$
\mathsf B_{c^nq_0}(x_n,x_{n+1}).
$$
有限次使用三角性得到，对 $k\ge1$，
$$
\mathsf B_{q_0c^n(1+c+\cdots+c^{k-1})}
(x_n,x_{n+k}).
$$
给定 $\varepsilon>0$，有理几何级数的 Archimedean 估计给出 $N$，使
$$
\frac{q_0c^N}{1-c}<\varepsilon.
$$
上式半径不超过 $q_0c^N/(1-c)$；结合单调性、对称性，得到任意
$m,n\ge N$ 时 $\mathsf B_\varepsilon(x_m,x_n)$。因此 $(x_n)$ Cauchy。

完备性只先给出
$$
\left\|\sum_{x:X}\mathsf{Lim}(x_\bullet,x)\right\|.
$$
由推论 BO.5，括号内的极限总类型是命题，所以此处可以且只在此处消去截断，取得 $(x,L)$。压缩条件还给出 $T$ 的连续性：要证明
$\mathsf B_\varepsilon(Tu,Tv)$，从
$\mathsf B_{\varepsilon/c}(u,v)$
应用压缩律即可。于是 $T(x_n)=x_{n+1}$ 收敛到 $T(x)$；另一方面，移位序列 $(x_{n+1})$ 由 $L$ 直接收敛到 $x$。命题 BO.4 的极限唯一性给出
$$
p:T(x)=x.
$$

还需证明中心唯一。设 $(y,r):\mathsf{Fix}(T)$。有限距离仅仅给出某个
$q>0$ 使 $\mathsf B_q(x,y)$；目标 $x=y$ 是集合 $X$ 中的路径命题，所以可消去该截断。对每个 $n$ 迭代压缩，并沿 $p,r$ 重写端点，得到
$$
\mathsf B_{c^nq}(x,y).
$$
给定 $\varepsilon>0$，选 $n$ 使 $c^nq<\varepsilon$，再用单调性得
$\mathsf B_\varepsilon(x,y)$。分离性给出 $x=y$。由于不动点方程是命题，子类型路径刻画把这条路径提升为
$(x,p)=(y,r)$。因此 $(x,p)$ 是 $\mathsf{Fix}(T)$ 的收缩中心。$\square$

非空假设不可删除：空类型配备唯一自映射时，完备性和压缩条件都可空泛成立，但不存在不动点。Mere inhabitation 已足够，是因为最终的可收缩性结论为命题；证明没有从 $\|X\|$ 构造一个可在结论外观察的任意点。

## BO.5 Riemann 积分

在 Cauchy 实数 $\mathbb R_C$ 上取
$$
\mathsf B_q(u,v)\coloneqq(|u-v|<q).
$$
设 $a\le b$，记 $L\coloneqq b-a$。

**定义 BO.15（tagged partition）.** 区间 $[a,b]$ 的 tagged partition 是有限数据
$$
a=x_0\le x_1\le\cdots\le x_n=b,
\qquad
t_i\in[x_i,x_{i+1}].
$$
其 mesh 为有限集合中的最大长度
$$
\mathsf{mesh}(P)\coloneqq
\max_{i<n}(x_{i+1}-x_i),
$$
Riemann 和为
$$
S(f,P)\coloneqq
\sum_{i<n}f(t_i)(x_{i+1}-x_i).
$$
有限最大值的存在使用实数序的 located 比较；退化情形 $a=b$ 时所有和为 $0$。

**定义 BO.16（Riemann 可积）.** 函数
$f:[a,b]\to\mathbb R_C$ Riemann 可积，若
$$
\sum_{I:\mathbb R_C}
\prod_{\varepsilon:\mathbb Q_{>0}}
\left\|
\sum_{\delta:\mathbb Q_{>0}}
\prod_P
(\mathsf{mesh}(P)<\delta)
\to |S(f,P)-I|<\varepsilon
\right\|
$$
有项。分离性保证积分值 $I$ 唯一。

**定理 BO.17（一致连续函数可积）.** 假设 $[a,b]$ 上的有限划分具有 common refinement，Cauchy 实数 sequentially complete，并可为每个正有理数选择更细的 dyadic mesh。若
$f:[a,b]\to\mathbb R_C$ 一致连续，则 $f$ Riemann 可积。

**证明（书内证明核）.** 对第 $n$ 个误差 $2^{-n}$，由一致连续性取得振幅界，并选择 mesh 足够小的规范 dyadic tagged partition $P_n$。若 $m,n$ 足够大，把 $P_m,P_n$ 拉到 common refinement；每个小区间上更换 tag 造成的误差小于预定振幅乘区间长度，有限求和后由总长度 $L$ 控制。因此
$S(f,P_n)$ 是 Cauchy 序列。由 BO.5 从 mere completeness 合法取得唯一极限 $I$。

给定 $\varepsilon>0$，选择一个已经进入 $\varepsilon/3$ 尾部的 $P_n$，再取足够小的 $\delta$。任意 mesh 小于 $\delta$ 的 tagged partition $P$ 与 $P_n$ 作 common refinement；同一振幅估计给出
$$
|S(f,P)-S(f,P_n)|<2\varepsilon/3,
$$
而极限尾估计给出
$|S(f,P_n)-I|<\varepsilon/3$。三角不等式得到定义 BO.16。全部选择都发生在命题值的“存在 $\delta$”目标内；有限组合、Archimedean 和 locatedness 假设已列在定理中。$\square$

## BO.6 微积分基本定理的边界

构造性微积分必须区分点态可微、具有统一余项控制的强可微，以及导数值域的 locatedness。若 $F$ 具有强导数 $f$，其定义应给出统一误差：对每个 $\varepsilon>0$，存在 $\delta>0$，使 $0<|h|<\delta$ 时
$$
|F(x+h)-F(x)-f(x)h|<\varepsilon|h|
$$
在规定区间内一致成立。将此估计沿划分求和，线性项 telescopes 为
$F(b)-F(a)$，余项由区间总长度控制；再用 BO.17 的积分唯一性可推出
$$
\int_a^b f=F(b)-F(a).
$$
本附录没有建立强导数、端点处理和上述统一余项的完整理论，因此这段只说明后续定理所需的精确输入，不把经典点态证明直接移植为书内定理。
