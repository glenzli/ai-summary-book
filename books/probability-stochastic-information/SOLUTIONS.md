# 习题解答

下列解答与正文使用同一符号和证明边界。涉及外部输入的题目只调用正文已经精确陈述的版本；其余推导均在此完成。

## 第 0 章

**练习 0.1.** 值域为 $\{0,1\}$，且

$$
f^{-1}(\{0\})=\{2,4\},\qquad
f^{-1}(\{1\})=\{1,3\}.
$$

由推前测度定义，

$$
\nu(\{0\})=\frac{|\{2,4\}|}{4}=\frac12,\qquad
\nu(\{1\})=\frac{|\{1,3\}|}{4}=\frac12.
$$

再加上 $\nu(\varnothing)=0$、$\nu(\{0,1\})=1$，这就逐个确定了值域幂集上的分布。

**练习 0.2.** 令 $U,Y_1,Y_2$ 为均匀 Bernoulli 随机变量，其中 $Y_1,Y_2$ 独立。取

$$
X_1=U,\qquad X_2=U.
$$

四个变量的一维边缘分布都是 Bernoulli$(1/2)$。然而

$$
\mathbb P(X_1=X_2)=1,
\qquad
\mathbb P(Y_1=Y_2)=\frac12.
$$

因此 $(X_1,X_2)$ 与 $(Y_1,Y_2)$ 的二维联合分布不同。

**练习 0.3.** “有限字母表互信息非负”是书内定理，因为第 8 章把互信息写成 KL 散度并证明 Gibbs 不等式。“有限字母表平稳遍历信源满足 AEP”调用 Shannon--McMillan--Breiman 外部输入 EI-10。“某压缩器在英文文本上表现好”不是单凭概率模型可判真的数学定理；它需要指定数据集、码长或错误指标及实验协议，属于经验模型判断。

## 第 1 章

**练习 1.1.** 若 $A_n\in\mathcal F$，则 $A_n^c\in\mathcal F$，并且 $\bigcup_nA_n^c\in\mathcal F$。由 De Morgan 律，

$$
\bigcap_{n=1}^{\infty}A_n
=\left(\bigcup_{n=1}^{\infty}A_n^c\right)^c\in\mathcal F.
$$

所以 $\sigma$-代数对可列交封闭。

**练习 1.2.** 任何包含 $\{1\}$ 的 $\sigma$-代数也必须包含其补集 $\{2,3\}$、空集和全集。因此

$$
\sigma(\{\{1\}\})
=\{\varnothing,\{1\},\{2,3\},\{1,2,3\}\}.
$$

右端本身满足补集和可列并封闭，所以没有还需加入的集合。

**练习 1.3.** 在 $(\mathbb N,2^{\mathbb N})$ 上取计数测度 $\mu$，令

$$
A_n=\{n,n+1,n+2,\ldots\}.
$$

则 $A_n\downarrow\varnothing$，但每个 $\mu(A_n)=+\infty$，并不趋于 $\mu(\varnothing)=0$。这里恰有 $\mu(A_1)=+\infty$，说明连续性从上定理中的有限测度假设不能直接删除。

## 第 2 章

**练习 2.1.** 令

$$
\mathcal C=\{B\in\mathcal B(\mathbb R):X^{-1}(B)\in\mathcal F\}.
$$

原像保持补集与可列并，所以 $\mathcal C$ 是 $\sigma$-代数。假设说明每个半无限区间 $(-\infty,a]$ 属于 $\mathcal C$。这些区间生成 $\mathcal B(\mathbb R)$，故 $\mathcal B(\mathbb R)\subseteq\mathcal C$。因此每个 Borel 集的原像可测，$X$ 是实值随机变量。

**练习 2.2.** 按离散期望定义，

$$
\mathbb E X
=1\cdot\frac12+2\cdot\frac13+3\cdot\frac16
=\frac53,
$$

而

$$
\mathbb E X^2
=1^2\cdot\frac12+2^2\cdot\frac13+3^2\cdot\frac16
=\frac{10}{3}.
$$

**练习 2.3.** 令 $U,V$ 独立且均匀取 $\{0,1\}$，并置

$$
X=U,\qquad Y=V,\qquad Z=U\oplus V.
$$

例如，对任意 $x,z\in\{0,1\}$，给定 $U=x$ 后恰有一个 $v$ 使 $x\oplus v=z$，故

$$
\mathbb P(X=x,Z=z)=\frac14
=\mathbb P(X=x)\mathbb P(Z=z).
$$

同理可证 $(X,Y)$、$(Y,Z)$ 独立。然而 $X\oplus Y\oplus Z=0$ 恒成立，所以事件 $\{X=0,Y=0,Z=1\}$ 的概率为 $0$，而三个边缘概率之积为 $1/8$；三者不共同独立。

## 第 3 章

**练习 3.1.** 对非负随机变量 $|X|$ 使用 Markov 不等式：

$$
\mathbb P(|X|\ge a)\le\frac{\mathbb E|X|}{a}.
$$

$X\in L^1$ 保证分子有限，故右端在 $a\to\infty$ 时趋于 $0$。

**练习 3.2.** Bernoulli$(p)$ 随机变量只取 $0,1$，因此

$$
\mathbb E X=0\cdot(1-p)+1\cdot p=p,
\qquad
\mathbb E X^2=p.
$$

由方差恒等式，

$$
\operatorname{Var}(X)=\mathbb E X^2-(\mathbb E X)^2
=p-p^2=p(1-p).
$$

**练习 3.3.** 有限值随机变量满足

$$
\mathbb E X=\sum_i\lambda_i x_i,
\qquad
\mathbb E[\varphi(X)]=\sum_i\lambda_i\varphi(x_i).
$$

权重 $\lambda_i$ 非负且和为 $1$，直接代入定理 3.4，得到 $\varphi(\mathbb E X)\le\mathbb E[\varphi(X)]$。

**练习 3.4.** 对固定 $\omega\in(0,1)$，当 $n>1/\omega$ 时 $\omega\notin(0,1/n)$，故 $X_n(\omega)=0$。因此 $X_n\to0$ 逐点。另一方面，

$$
\mathbb E X_n=n\,\mathbb P((0,1/n))=1
$$

对所有 $n$ 成立，所以期望不趋于 $0$。不存在统一的可积控制函数：若 $Y\ge X_n$ 对所有 $n$，则对充分小的 $\omega$ 选择 $n$ 约为 $1/(2\omega)$，可得 $Y(\omega)\ge1/(2\omega)-1$，其在零点附近不可积。因此 EI-2c 的可积控制假设失败。

**练习 3.5.** 令 $\widetilde X=X-\mathbb E X$、$\widetilde Y=Y-\mathbb E Y$。则

$$
\begin{aligned}
\operatorname{Var}(X+Y)
&=\mathbb E[(\widetilde X+\widetilde Y)^2]\\
&=\mathbb E[\widetilde X^2]
 +\mathbb E[\widetilde Y^2]
 +2\mathbb E[\widetilde X\widetilde Y]\\
&=\operatorname{Var}(X)+\operatorname{Var}(Y)
 +2\operatorname{Cov}(X,Y).
\end{aligned}
$$

Cauchy--Schwarz 保证交叉项可积。

## 第 4 章

**练习 4.1.** 设 $X\sim\operatorname{Bernoulli}(p)$、$Y\sim\operatorname{Bernoulli}(q)$ 且独立。则

$$
\mathbb P(X+Y=0)=(1-p)(1-q),
$$

$$
\mathbb P(X+Y=1)=p(1-q)+(1-p)q,
$$

$$
\mathbb P(X+Y=2)=pq.
$$

三项之和为 $1$，因此这给出了完整分布。

**练习 4.2.** 可测性给出 $\sigma(f(X))\subseteq\sigma(X)$ 与 $\sigma(g(Y))\subseteq\sigma(Y)$。若 $A\in\sigma(f(X))$、$B\in\sigma(g(Y))$，独立性给出

$$
\mathbb P(A\cap B)=\mathbb P(A)\mathbb P(B).
$$

所以 $f(X)$ 与 $g(Y)$ 独立。

**练习 4.3.** 中间状态可以为 $0$ 或 $1$，因此

$$
\begin{aligned}
P^2(0,1)
&=P(0,0)P(0,1)+P(0,1)P(1,1)\\
&=(1-b)b+ba.
\end{aligned}
$$

## 第 5 章

**练习 5.1.** 对 $\mathbb P(Y=y)>0$ 的 $y$，令

$$
m_y=\frac{\mathbb E[X\mathbf 1_{\{Y=y\}}]}{\mathbb P(Y=y)}.
$$

在零概率点任取有限 $m_y$。则 $Z=\sum_y m_y\mathbf 1_{\{Y=y\}}$ 是 $\sigma(Y)$-可测且可积的。任意 $A\in\sigma(Y)$ 是若干纤维 $\{Y=y\}$ 的并；在正概率纤维上

$$
\int_{\{Y=y\}}Z\,d\mathbb P
=m_y\mathbb P(Y=y)
=\int_{\{Y=y\}}X\,d\mathbb P,
$$

零概率纤维两边都为零。逐纤维求和即验证定义。

**练习 5.2.** $M_n$ 是 $\mathcal F_n$-可测且可积。由于 $X_{n+1}$ 与 $\mathcal F_n$ 独立且均值为零，

$$
\mathbb E[X_{n+1}\mid\mathcal F_n]=0.
$$

因此

$$
\mathbb E[M_{n+1}\mid\mathcal F_n]
=\mathbb E[M_n+X_{n+1}\mid\mathcal F_n]
=M_n,
$$

所以 $(M_n)$ 是鞅。

**练习 5.3.** 事件 $\{\tau\le m\}$ 表示在前 $m$ 步已经到达 $1$，或者 $m=N$；它由前 $m$ 个增量决定，故属于自然滤过。又 $\tau\le N$，所以 $\tau$ 是有界停时。简单对称随机游走是从 $S_0=0$ 开始的鞅，定理 5.3 给出

$$
\mathbb E S_\tau=\mathbb E S_0=0.
$$

$S_\tau$ 不恒为 $1$：未在 $N$ 前击中的路径取值 $S_N$，它们补偿击中路径的正值。

**练习 5.4.** 令 $c=\mathbb E X$。常数 $c$ 是 $\mathcal G$-可测且可积的。对 $A\in\mathcal G$，独立性通过简单函数逼近给出

$$
\mathbb E[X\mathbf 1_A]
=\mathbb E X\,\mathbb P(A)
=\int_Ac\,d\mathbb P.
$$

所以 $c$ 满足条件期望定义；唯一性给出 $\mathbb E[X\mid\mathcal G]=\mathbb E X$ 几乎处处。

**练习 5.5.** 在定义中取 $A=\Omega$，得到

$$
\mathbb E[\mathbb E[X\mid\mathcal G]]=\mathbb E X.
$$

若 $A\in\mathcal H\subseteq\mathcal G$，则

$$
\int_A\mathbb E[X\mid\mathcal G]\,d\mathbb P
=\int_AX\,d\mathbb P.
$$

故 $\mathbb E[\mathbb E[X\mid\mathcal G]\mid\mathcal H]$ 与 $\mathbb E[X\mid\mathcal H]$ 满足同一组 $\mathcal H$-积分恒等式，唯一性给出塔性质。

## 第 6 章

**练习 6.1.** 对 $|X_n-X|^p$ 使用 Markov 不等式：

$$
\mathbb P(|X_n-X|>\varepsilon)
\le\frac{\mathbb E|X_n-X|^p}{\varepsilon^p}\to0.
$$

这正是依概率收敛。

**练习 6.2.** 固定 $m\ge1$。Markov 不等式给出

$$
\sum_n\mathbb P(|X_n|>1/m)
\le m\sum_n\mathbb E|X_n|<\infty.
$$

由第一 Borel--Cantelli 引理，$\{|X_n|>1/m\}$ 几乎处处只发生有限次。对所有 $m$ 取可列交。在所得概率一事件上，给定 $\varepsilon>0$，选 $m$ 使 $1/m<\varepsilon$，则充分大 $n$ 时 $|X_n|\le1/m<\varepsilon$。故 $X_n\to0$ 几乎处处。

**练习 6.3.** 弱大数律说

$$
\forall\varepsilon>0,\qquad
\mathbb P(|\overline X_n-p|>\varepsilon)\to0.
$$

强大数律说

$$
\mathbb P\{\omega:\overline X_n(\omega)\to p\}=1.
$$

前者允许坏样本集合随 $n$ 改变，只要求其概率趋零；后者给出一个固定概率一集合，其中每条路径都收敛。

**练习 6.4.** 例 6.1(1) 否定“依概率推出几乎处处”；例 6.1(2) 否定“依分布推出依概率”；例 6.1(3) 否定无控制条件下“几乎处处推出 $L^1$”。

**练习 6.5.** 有

$$
\frac1{n^2}\sum_{k=1}^n\operatorname{Var}(X_k)
\le\frac C{n^2}\sum_{k=1}^nk^\alpha.
$$

若 $\alpha>-1$，右端为 $O(n^{\alpha-1})$；若 $\alpha=-1$，为 $O((\log n)/n^2)$；若 $\alpha<-1$，为 $O(n^{-2})$。因此 $\alpha<1$ 时必趋零。对 $\alpha\ge1$，取独立中心随机变量使 $\operatorname{Var}(X_k)=k^\alpha$；在 $\alpha=1$ 时归一化方差和趋于正常数，在 $\alpha>1$ 时不趋零。所以给定上界保证条件成立恰当于 $\alpha<1$。

**练习 6.6.** 因为 $\sum_{k=1}^nX_k-n\mu=n(\overline X_n-\mu)$，中心极限定理等价于

$$
\frac{\sqrt n(\overline X_n-\mu)}{\sigma}
\Rightarrow N(0,1).
$$

## 第 7 章

**练习 7.1.** 对有限状态空间，

$$
P^{n+1}(i,j)=\sum_kP^n(i,k)P(k,j),
$$

这就是矩阵乘法；$P^0$ 是单位矩阵。归纳得到递归核幂与普通矩阵幂一致。

**练习 7.2.** 在 $0<a,b<1$ 下，

$$
\frac b{1-a+b}=\frac12
\iff 2b=1-a+b
\iff a+b=1.
$$

**练习 7.3.** 边缘概率为

$$
\mathbb P(X_n=1)=\mathbb E\Theta
=\frac12\left(\frac14+\frac34\right)=\frac12.
$$

给定 $\Theta$ 后过程独立同分布，所以任意有限块的混合分布不随平移改变，过程平稳。若是一阶 Markov 链，应有

$$
\mathbb P(X_3=1\mid X_1=1,X_2=1)
=\mathbb P(X_3=1\mid X_2=1).
$$

但

$$
\mathbb P(X_3=1\mid X_2=1)
=\frac{\mathbb E\Theta^2}{\mathbb E\Theta}
=\frac{5/16}{1/2}=\frac58,
$$

而

$$
\mathbb P(X_3=1\mid X_1=1,X_2=1)
=\frac{\mathbb E\Theta^3}{\mathbb E\Theta^2}
=\frac{7/32}{5/16}=\frac7{10}.
$$

两者不同，故过程不是一阶 Markov 链。

## 第 8 章

**练习 8.1.** 令 $U$ 为 $\mathcal X$ 上均匀分布。则

$$
D(P_X\|U)=\log|\mathcal X|-H(X).
$$

Gibbs 不等式给出熵上界。等号等价于散度为零，再由 Gibbs 等号条件得到 $P_X=U$；均匀分布反向直接取到上界。

**练习 8.2.** Bernoulli$(p)$ 的熵为

$$
h_2(p)=-p\log p-(1-p)\log(1-p).
$$

由 $0\log0=0$，$h_2(0)=h_2(1)=0$；代入 $p=1/2$ 得 $h_2(1/2)=1$。

**练习 8.3.** 独立时 $P_{XY}=P_XP_Y$，所以互信息为零。反之，若互信息为零，Gibbs 等号条件给出 $P_{XY}=P_XP_Y$，故 $X,Y$ 独立。

**练习 8.4.** 条件互信息恒等式为

$$
I(X;Z\mid Y)=H(X\mid Y)-H(X\mid Y,Z)\ge0.
$$

移项即得 $H(X\mid Y,Z)\le H(X\mid Y)$。

**练习 8.5.** 给定 $Y=y$ 后，$Z=f(y)$ 确定，故 $X$ 与 $Z$ 在给定 $Y$ 后条件独立，即 $X\to Y\to Z$。数据处理不等式给出 $I(X;Z)\le I(X;Y)$。

## 第 9 章

**练习 9.1.** 取码字 $0,10,11$。Kraft 和为

$$
2^{-1}+2^{-2}+2^{-2}=1,
$$

平均长度为

$$
\frac12\cdot1+\frac14\cdot2+\frac14\cdot2=\frac32.
$$

**练习 9.2.** 均匀消息满足 $H(J)=\log M$。因此

$$
\begin{aligned}
I(J;\widehat J)
&=H(J)-H(J\mid\widehat J)\\
&\ge\log M-h_2(P_e)-P_e\log(M-1).
\end{aligned}
$$

**练习 9.3.** 输出分布为

$$
\mathbb P(Y=0)=(1-\epsilon)(1-q),\quad
\mathbb P(Y=1)=(1-\epsilon)q,\quad
\mathbb P(Y=? )=\epsilon.
$$

展开得到

$$
H(Y)=h_2(\epsilon)+(1-\epsilon)h_2(q).
$$

对每个输入，输出只在该输入符号和擦除符号之间随机，故 $H(Y\mid X)=h_2(\epsilon)$。所以

$$
I(X;Y)=(1-\epsilon)h_2(q).
$$

**练习 9.4.** 若 $x^n,x'^n\in\mathcal C_n$ 且 $f_n(x^n)=f_n(x'^n)=j$，则

$$
x^n=g_n(j)=x'^n.
$$

故 $f_n|_{\mathcal C_n}$ 单射，$|\mathcal C_n|\le M$。零错误要求每个正概率序列都正确解码，而 DMS 支持为 $(\operatorname{supp}P)^n$，所以

$$
|\operatorname{supp}P|^n\le|\mathcal C_n|\le M.
$$

**练习 9.5.** 平均数不超过最大项：

$$
P_{e,n}^{\mathrm{av}}
=\frac1M\sum_j\lambda_j
\le\max_j\lambda_j=P_{e,n}^{\max}.
$$

严格例子取两个消息和恒定输出信道，解码器总输出消息 $1$。此时 $\lambda_1=0,\lambda_2=1$，平均错误为 $1/2$，最大错误为 $1$。

**练习 9.6.** 联合表为

$$
\begin{array}{c|cc}
 &Y=0&Y=1\\\hline
X=0&(1-q)(1-\varepsilon)&(1-q)\varepsilon\\
X=1&q\varepsilon&q(1-\varepsilon)
\end{array}.
$$

故 $\mathbb P(Y=1)=\varepsilon+(1-2\varepsilon)q$，而 $H(Y\mid X)=h_2(\varepsilon)$。所以

$$
I_q(X;Y)
=h_2(\varepsilon+(1-2\varepsilon)q)-h_2(\varepsilon).
$$

**练习 9.7.** 若所有行均为 $Q$，则 $p(x)W(y\mid x)=p(x)Q(y)$，输入输出独立，容量为零。反之，若两行 $W(\cdot\mid x_0)$、$W(\cdot\mid x_1)$ 不同，在两输入上各放概率 $1/2$。若互信息为零，则独立性迫使这两个正概率输入下的条件输出分布都等于同一输出边缘，矛盾。因此某输入分布产生正互信息，容量大于零。

**练习 9.8.** 由 $h_2(u)\le1$，

$$
(1-P_{e,n}^{\mathrm{av}})\log M\le nC(W)+1.
$$

除以 $n(1-P_{e,n}^{\mathrm{av}})$ 得

$$
R_n^{\mathrm{ch}}
\le\frac{C(W)+1/n}{1-P_{e,n}^{\mathrm{av}}}.
$$

若错误概率不趋零，分母不趋于 $1$，甚至可趋于零；该上界因而允许码率大于容量，只表明这种码不可靠。

## 第 10 章

**练习 10.1.** 由独立性与链式法则，

$$
H(X_1^n)
=\sum_{k=1}^nH(X_k\mid X_1^{k-1})
=\sum_{k=1}^nH(X_k)
=nH(X_1).
$$

故 $h(X)=H(X_1)$。

**练习 10.2.** 二状态链有

$$
h(X)=\frac{1-a}{1-a+b}h_2(b)
+\frac b{1-a+b}h_2(a).
$$

当 $a=b=p$ 时，两行转移分布都是 Bernoulli$(p)$，故 $h(X)=h_2(p)$。

**练习 10.3.** 长度 $n$ 的块只有 $0^n$ 与 $1^n$，各概率 $1/2$，所以

$$
H_n=1,\qquad h(X)=\lim_{n\to\infty}\frac{H_n}{n}
=\lim_{n\to\infty}\frac1n=0.
$$

对 $f(x)=\mathbf 1_{\{x_1=1\}}$，时间平均恒为 $\Theta$。不变 $\sigma$-代数包含 $\sigma(\Theta)$，故 Birkhoff 极限为 $\mathbb E[f\mid\mathcal I]=\Theta$，而非总体均值 $1/2$。

**练习 10.4.** 令 $N_{k-1}=\sum_{j<k}L_j$。奇数均匀块末端，均匀位置比例至少

$$
\frac{L_k}{N_{k-1}+L_k}\ge\frac{k}{k+1}.
$$

偶数确定块末端，均匀位置比例至多

$$
\frac{N_{k-1}}{N_{k-1}+L_k}\le\frac1{k+1}.
$$

两个子序列分别趋于 $1$ 与 $0$。块熵等于均匀位置数，所以 $H_n/n$ 不收敛。

**练习 10.5.** 取 $A_n=\mathcal T_{n,\delta}$。AEP 给出其概率趋于 $1$，所以充分大 $n$ 时至少为 $1-\eta$；典型集上界同时给出 $|A_n|\le2^{n(h+\delta)}$。

**练习 10.6.** 概率空间是路径空间

$$
(\mathcal X^{\mathbb N},
\mathcal B(\mathcal X)^{\otimes\mathbb N},\mu),
$$

变换为左移 $T$，函数为 $f(x)=\mathbf 1_{\{x_1=a\}}$，不变 $\sigma$-代数为

$$
\mathcal I=\{A:\mu(T^{-1}A\mathbin{\triangle}A)=0\}.
$$

Birkhoff 平均正是字母 $a$ 的经验频率。遍历时 $\mathcal I$ 模零平凡，极限为 $\int f\,d\mu=\mathbb P(X_1=a)$。

**练习 10.7.** 定理 10.6 只涉及固定 logits 经温度缩放所得概率向量的 Shannon 熵。要形成可检验的创造力命题，至少还需指定文本样本空间、抽样协议、创造力的可测评分函数或人工评价规则、任务与提示分布、比较的统计量、长度和质量等混杂变量，以及要检验的统计或因果关系。没有这些对象时，“创造力”不是定理中的数学量。
