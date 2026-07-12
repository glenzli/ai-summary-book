# 附录 AD：椭圆曲线约化、Neron 模型、Kodaira 符号和 Tate Algorithm

本附录补足第八、第十和第九十章中椭圆曲线局部理论的技术接口。目标不是重证 Neron 模型存在定理或 Tate algorithm，而是把最小模型、约化类型、导子、局部 L 因子和 Frey 曲线降层所需局部计算放在同一套可引用的定义和命题中。

**收口归一化回指。** 本附录涉及局部 Frobenius、Tate module、导子、局部 L 因子和 Frey 曲线约化；与模性和 Galois 表示比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 1、5、6、7、8 节。

## AD.1 局部 Weierstrass 模型和最小判别式

设 $F$ 为非 Archimedean 局部域，整数环为 $\mathcal O_F$，一致化元为 $\varpi$，剩余域为 $k$，且 $q=\#k$。

**定义 AD.1.** $E/F$ 的一个 integral Weierstrass model 是方程
$$
y^2+a_1xy+a_3y=x^3+a_2x^2+a_4x+a_6,\qquad a_i\in\mathcal O_F,
$$
其 generic fiber 与 $E$ 同构。其判别式记为 $\Delta$。

**命题 AD.2（变量变换与判别式）.** 若两个 Weierstrass 方程由允许变换
$$
x=u^2x'+r,\qquad y=u^3y'+u^2sx'+t,\qquad u\in F^\times
$$
相连，则判别式满足
$$
\Delta'=u^{-12}\Delta.
$$

**证明路线（外部输入）.** Weierstrass 不变量 $b_i,c_4,c_6,\Delta$ 对标准变量变换有权重。$x$ 的权重为 $2$，$y$ 的权重为 $3$，判别式为权重 $12$ 的相对不变量。把长 Weierstrass 不变量逐项代入变换公式可得 $\Delta=u^{12}\Delta'$，即所述公式。$\square$

**定义 AD.3.** integral Weierstrass model 称为 minimal，若 $v_F(\Delta)$ 在所有 integral Weierstrass models 中最小。最小值记为
$$
v_F(\Delta_E).
$$

**外部输入定理 AD.4（局部最小模型存在性）.** 每条椭圆曲线 $E/F$ 都存在局部最小 Weierstrass model。若 $F$ 为离散赋值域，则最小判别式指数 $v_F(\Delta_E)$ 与模型选择无关。

## AD.2 Neron 模型和约化

**定义 AD.5.** $E/F$ 的 Neron model 是光滑、分离、有限型 $\mathcal O_F$-群概形 $\mathcal E/\mathcal O_F$，generic fiber 为 $E$，并满足 Neron mapping property：对任意光滑 $\mathcal O_F$-scheme $S$ 和任意 $F$-morphism
$$
S_F\to E,
$$
存在唯一 $\mathcal O_F$-morphism
$$
S\to\mathcal E
$$
延拓它。

**外部输入定理 AD.6（Neron model existence）.** 每条 Abelian variety over $F$ 都存在 Neron model。特别地，每条椭圆曲线 $E/F$ 都存在 Neron model。

**定义 AD.7.** 记 Neron model 的特殊纤维为 $\mathcal E_k$，其 identity component 为 $\mathcal E_k^0$。Component group 定义为
$$
\Phi_E=\mathcal E_k/\mathcal E_k^0.
$$
Tamagawa number 为
$$
c_F(E)=\#\Phi_E(k)
$$
当该集合有限时。

**命题 AD.8.** $E/F$ 有好约化当且仅当 Neron model $\mathcal E/\mathcal O_F$ 是 Abelian scheme；在椭圆曲线情形等价于 $\mathcal E_k$ 为光滑 genus $1$ 曲线。

**证明路线（外部输入）.** 若 $E$ 有好约化，则存在 proper smooth integral model，其群结构由 $E$ 的 group law 延拓，并满足 Neron mapping property，因此是 Neron model。反之，若 Neron model proper，则它是 Abelian scheme，特殊纤维为 Abelian variety；维数一情形即光滑 genus $1$ 曲线。完整证明使用 Neron mapping property 和 Abelian schemes 的 valuative criterion。$\square$

## AD.3 Kodaira-Neron 约化类型

**定义 AD.9.** 椭圆曲线 $E/F$ 的 Kodaira symbol 是由最小正则模型特殊纤维的 irreducible components、交叉图和重数决定的符号：
$$
I_0,\ I_n,\ II,\ III,\ IV,\ I_n^*,\ IV^*,\ III^*,\ II^*.
$$
其中 $I_0$ 表示好约化，$I_n$ 表示乘法约化，其他符号表示加法约化。

**外部输入定理 AD.10（Kodaira-Neron classification）.** 若 residue characteristic 不是 $2,3$，Kodaira symbol、最小判别式指数、特殊纤维 components 数和导子指数满足下表：

| Type | reduction | $v(\Delta_E)$ | components $m$ | conductor exponent $f$ |
|---|---:|---:|---:|---:|
| $I_0$ | good | $0$ | $1$ | $0$ |
| $I_n$, $n\ge1$ | multiplicative | $n$ | $n$ | $1$ |
| $II$ | additive | $2$ | $1$ | $2$ |
| $III$ | additive | $3$ | $2$ | $2$ |
| $IV$ | additive | $4$ | $3$ | $2$ |
| $I_0^*$ | additive | $6$ | $5$ | $2$ |
| $I_n^*$, $n\ge1$ | additive | $n+6$ | $n+5$ | $2$ |
| $IV^*$ | additive | $8$ | $7$ | $2$ |
| $III^*$ | additive | $9$ | $8$ | $2$ |
| $II^*$ | additive | $10$ | $9$ | $2$ |

在 residue characteristic $2,3$，同一符号仍存在，但 conductor exponent 可能包含 wild contribution，需由 Tate algorithm 计算。

**命题 AD.11.** $E/F$ 半稳定当且仅当 Kodaira type 为 $I_0$ 或某个 $I_n$。

**证明.** 半稳定按定义表示特殊纤维只允许好约化或 nodal multiplicative reduction。Kodaira 分类中 $I_0$ 正是好约化，$I_n$ 正是乘法约化；其他类型均为 cuspidal 或更复杂的 additive reduction。因此等价。$\square$

## AD.4 Tate 曲线和乘法约化

**外部输入定理 AD.12（Tate uniformization）.** 若 $E/F$ 有 split multiplicative reduction，则存在 $q_E\in F^\times$，$v(q_E)>0$，使
$$
E(\overline F)\simeq \overline F^\times/q_E^\mathbb Z
$$
作为 $G_F$-modules 相容。并且
$$
v(\Delta_E)=v(q_E),\qquad f(E)=1.
$$

**命题 AD.13.** 若 $E/F$ 有乘法约化，则局部 L 因子为
$$
L(E/F,s)=
\begin{cases}
(1-q^{-s})^{-1},&\text{split multiplicative},\\
(1+q^{-s})^{-1},&\text{nonsplit multiplicative}.
\end{cases}
$$

**证明路线（外部输入）.** Split multiplicative 情形由 Tate uniformization 给出 exact sequence
$$
0\to\mathbb Q_\ell(1)\to V_\ell(E)\to\mathbb Q_\ell\to0.
$$
惯性在不变量商上给出一维不变量，Frobenius 在该不变量上以 $1$ 作用，故局部因子为 $(1-q^{-s})^{-1}$。Nonsplit multiplicative reduction 在一个二次非分歧扩张后 split，Frobenius 在相应不变量上多出非平凡 unramified quadratic character，故特征值为 $-1$。$\square$

## AD.5 导子公式

**定义 AD.14.** 对 $\ell\ne p$，令
$$
V_\ell(E)=T_\ell(E)\otimes_{\mathbb Z_\ell}\mathbb Q_\ell.
$$
局部导子指数定义为 Artin conductor
$$
f(E/F)=a(V_\ell(E)).
$$
该整数与 $\ell$ 的选择无关。

**外部输入定理 AD.15（Ogg conductor formula）.** 设 $\Delta_E$ 为最小判别式，$m$ 为最小正则模型特殊纤维的 irreducible components 数。则
$$
f(E/F)=v(\Delta_E)+1-m.
$$
在 residue characteristic $2,3$ 的 wild 情形中，该公式仍以最小正则模型的 components 数吸收 wild contribution；实际计算通常通过 Tate algorithm 完成。

**命题 AD.16.** 若 $E/F$ 半稳定，则
$$
f(E/F)=
\begin{cases}
0,&\text{good reduction},\\
1,&\text{multiplicative reduction}.
\end{cases}
$$

**证明.** 好约化类型为 $I_0$，表 AD.10 给出 $f=0$。乘法约化类型为 $I_n$，$n\ge1$，表 AD.10 给出 $f=1$。也可由 Ogg 公式得到：$v(\Delta_E)=n$，components 数 $m=n$，故 $f=n+1-n=1$。$\square$

## AD.6 Tate Algorithm 接口

**外部输入定理 AD.17（Tate algorithm）.** 给定局部域 $F$ 上的 integral Weierstrass equation，Tate algorithm 通过有限步代数检验输出：

1. 最小 Weierstrass model；
2. Kodaira symbol；
3. 最小判别式指数 $v(\Delta_E)$；
4. component group 和 Tamagawa number；
5. conductor exponent $f(E/F)$。

算法只使用 $a_i,b_i,c_4,c_6,\Delta$ 的 valuation、若干 residual polynomial 的分解情况以及允许变量变换。

**命题 AD.18.** 若 $p\ne2$，曲线
$$
E:\ y^2=(x-r_1)(x-r_2)(x-r_3)
$$
有 $r_i\in\mathcal O_F$，且模 $\mathfrak p_F$ 后恰有两个根相等、第三个根不同，则 $E$ 有乘法约化。

**证明路线（外部输入）.** 模 $\mathfrak p_F$ 的平面三次曲线有唯一奇点。两个根相等且第三个根不同意味着奇点为 node，而不是 cusp；node 的两条切线是否在 $k$ 中定义决定 split/nonsplit。按定义这就是乘法约化。$\square$

## AD.7 Frey 曲线局部导子接口

设
$$
a^p+b^p=c^p,\qquad p\ge5,
$$
为 primitive Fermat 反例，并取 Frey 曲线
$$
E:\ y^2=x(x-a^p)(x+b^p).
$$

**命题 AD.19.** Frey 曲线的判别式为
$$
\Delta=16a^{2p}b^{2p}c^{2p}.
$$

**证明.** 对方程 $y^2=(x-e_1)(x-e_2)(x-e_3)$，其中
$$
e_1=0,\qquad e_2=a^p,\qquad e_3=-b^p,
$$
短 Weierstrass 三次的判别式等于
$$
16(e_1-e_2)^2(e_1-e_3)^2(e_2-e_3)^2.
$$
这里
$$
e_1-e_2=-a^p,\quad e_1-e_3=b^p,\quad e_2-e_3=a^p+b^p=c^p.
$$
代入即得 $\Delta=16a^{2p}b^{2p}c^{2p}$。$\square$

**外部输入定理 AD.20（Frey 曲线局部类型）.** 在标准奇偶归一化的 primitive Fermat 反例下，Frey 曲线满足：

1. $E$ 半稳定；
2. 对奇素数 $q\mid abc$，$E$ 在 $q$ 处有乘法约化，且 $v_q(\Delta_E)$ 被 $p$ 整除；
3. $E$ 的导子具有形状
   $$
   N_E=2\prod_{q\mid abc,\ q\text{ odd}}q;
   $$
4. 模 $p$ 表示 $\overline\rho_{E,p}$ 的 prime-to-$p$ Serre conductor 等于 $2$。

**证明路线（外部输入）.** 第 2 项由命题 AD.18 和判别式公式 AD.19 的局部 valuation 给出；半稳定性还需检查素数 $2$ 的标准归一化情形。由于半稳定乘法约化的导子指数为 $1$，得到第 3 项。对 $q\mid abc$，$v_q(\Delta_E)$ 是 $p$ 的倍数；Tate curve 描述说明模 $p$ 惯性扩张类在这些 $q$ 处消失或降为非分歧，从而 residual conductor 删除这些奇素数。素数 $2$ 的剩余贡献由 Tate algorithm 的专门计算给出。完整计算是 Frey-Serre-Ribet 理论的局部输入。$\square$

**命题 AD.21.** 接受 AD.20，则第十章定理 10.11 中的 Frey 曲线局部导子计算成立。

**证明.** AD.20(1) 给出半稳定性。AD.20(2) 和 AD.16 说明奇坏素数在 $E$ 的导子中以一次出现。AD.20(4) 说明模 $p$ 残余表示的 prime-to-$p$ 导子只剩素数 $2$。这正是 Ribet 降层从 $N_E$ 删除所有奇素数并降到级 $2$ 所需的局部导子输入。$\square$

## 练习

**练习 AD.1.** 用变量变换权重解释为什么判别式按 $u^{-12}$ 缩放。

**练习 AD.2.** 用 Kodaira 表说明半稳定椭圆曲线的导子是坏乘法素数的一次乘积。

**练习 AD.3.** 说明 split multiplicative 与 nonsplit multiplicative 的局部 L 因子为什么只差一个符号。

**练习 AD.4.** 对 Frey 曲线 $y^2=x(x-a^p)(x+b^p)$ 计算三根差并推出判别式。

**练习 AD.5.** 解释为什么 $v_q(\Delta_E)$ 被 $p$ 整除会导致 residual conductor 可能小于 $E$ 的 conductor。
