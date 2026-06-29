# 第十五章：函子性原理

## 本章目标

本章系统表述 Langlands 函子性原理（functoriality principle）。第十一章定义了 L 同态
$$
\xi:{}^LH\to{}^LG,
$$
第十二章说明局部参数可由 $\xi$ 推前，第十三、十四章说明当目标为 `GL(n)` 时可借助强重数一和 converse theorem 检测全局自守性。本章把这些接口组织为全局转移问题：给定 $H$ 的自守表示 $\sigma$，是否存在 $G$ 的自守表示 $\Pi$，使得几乎所有局部 Satake 参数由 $\xi$ 推前得到？

## 依赖前置知识

需要第十一章的 L 群和 L 同态，第十二章的局部 L-packet，第十三章的全局自守表示和 L 函数，第十四章的 `GL(n)` converse theorem。需要知道 base change、automorphic induction、isobaric sum、Rankin-Selberg L 函数和 trace formula 的基本接口。本章把一般函子性、base change、automorphic induction、若干 symmetric power lifts、tensor product lifts 和 endoscopic transfer 的已知情形作为外部输入或猜想。

收口归一化回指：本章的函子性相容性以非分歧 Satake 参数和 L 群表示的局部因子为检测对象；相关 convention 见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、8 节。

## 15.1 从 L 同态到局部参数推前

本章固定整体域 $K$。设 $H/K$ 与 $G/K$ 为 connected reductive groups。由于数域情形没有无条件构造的全局 Langlands 群，本章把
$$
\xi:{}^LH\to{}^LG
$$
理解为一族可局部化的数据：对每个位置 $v$，有局部 L 同态
$$
\xi_v:{}^LH_v\to{}^LG_v,
$$
并且在几乎所有非分歧位置由同一个 pinned root datum 上的 L 群同态给出。

**定义 15.1.** 设 $v$ 为 $K$ 的位置，$\varphi_v:W_{K_v}'\to{}^LH_v$ 为 $H(K_v)$ 的局部参数。由 $\xi_v$ 得到的推前参数定义为
$$
\xi_{v,*}\varphi_v=\xi_v\circ\varphi_v:W_{K_v}'\to{}^LG_v.
$$

**命题 15.2.** 若 $\varphi_v$ 是覆盖 $W_{K_v}$ 的局部参数，则 $\xi_{v,*}\varphi_v$ 也是覆盖 $W_{K_v}$ 的局部参数。

**证明.** 由第十一章 L 同态定义，$\xi_v$ 与到 $W_{K_v}$ 的投影相容。于是复合
$$
W_{K_v}'\xrightarrow{\varphi_v}{}^LH_v\xrightarrow{\xi_v}{}^LG_v\to W_{K_v}
$$
等于 $\varphi_v$ 到 $W_{K_v}$ 的自然投影。代数性和半单性条件由 $\xi_v$ 在对偶群上的代数同态保持。$\square$

**定义 15.3.** 若 $\pi_v$ 是 $H(K_v)$ 的不可约可容许表示，且局部 LLC 给出参数 $\varphi_{\pi_v}$，则 $\xi$ 对 $\pi_v$ 的局部函子性转移是 $G(K_v)$ 的 L-packet
$$
\Pi_{\xi_{v,*}\varphi_{\pi_v}}(G).
$$
当 $G=\operatorname{GL}_N$ 时，该 packet 为单元素，记为
$$
\xi_{v,*}\pi_v.
$$

**注 15.4.** 对一般 $G$，局部转移不是单个表示，而是 packet 或稳定虚表示。若忽略 packet 内部结构，许多 endoscopic 符号和 multiplicity 公式会失真。

## 15.2 弱转移与强转移

设 $\sigma=\otimes_v'\sigma_v$ 为 $H(\mathbb A_K)$ 的自守表示，$\Pi=\otimes_v'\Pi_v$ 为 $G(\mathbb A_K)$ 的自守表示。

**定义 15.5.** 称 $\Pi$ 是 $\sigma$ 沿 $\xi$ 的弱函子性转移，若存在有限位置集合 $S$，使得对所有 $v\notin S$：

1. $H$ 与 $G$ 在 $K_v$ 上 unramified；
2. $\sigma_v$ 与 $\Pi_v$ spherical；
3. 两者 Satake 参数满足
   $$
   s(\Pi_v)=\xi_v(s(\sigma_v))
   $$
   作为 ${}^LG_v$ 中的半单共轭类。

**定义 15.6.** 称 $\Pi$ 是 $\sigma$ 沿 $\xi$ 的强函子性转移，若对每个位置 $v$，$\Pi_v$ 的局部 L-packet 参数等于
$$
\xi_{v,*}\varphi_{\sigma_v}.
$$
当目标为 `GL(N)` 时，这等价于
$$
\varphi_{\Pi_v}=\xi_v\circ\varphi_{\sigma_v}
$$
对所有 $v$ 成立。

**命题 15.7.** 强转移推出弱转移。

**证明.** 对几乎所有 $v$，$\sigma_v$ 与 $\Pi_v$ 非分歧。非分歧局部参数由几何 Frobenius 上的 Satake 参数决定。若每个局部参数满足
$$
\varphi_{\Pi_v}=\xi_v\circ\varphi_{\sigma_v},
$$
则在非分歧处取 $\operatorname{Fr}_v$ 得到
$$
s(\Pi_v)=\xi_v(s(\sigma_v)).
$$
这正是弱转移条件。$\square$

**命题 15.8.** 若 $G=\operatorname{GL}_N$，则弱函子性转移若存在则唯一。

**证明.** 设 $\Pi$ 与 $\Pi'$ 都是 $\sigma$ 沿 $\xi$ 的弱转移。则对几乎所有位置 $v$，$\Pi_v$ 与 $\Pi_v'$ 有相同 Satake 参数。对 `GL(N)`，Satake 参数决定非分歧局部表示。由第十四章强重数一，$\Pi\cong\Pi'$。$\square$

**注 15.9.** 对一般 $G$，弱转移不一定唯一；即使局部 packet 已知，也可能需要稳定 trace formula 和 multiplicity formula 来确定全局表示或稳定组合。

## 15.3 全局函子性猜想

**猜想 15.10（Langlands 函子性，弱形式）.** 设
$$
\xi:{}^LH\to{}^LG
$$
为 L 同态，设 $\sigma$ 为 $H(\mathbb A_K)$ 的 cuspidal automorphic representation，满足适当的代数性、中心特征和局部 relevance 条件。则存在 $G(\mathbb A_K)$ 的自守表示 $\Pi$，使 $\Pi$ 是 $\sigma$ 沿 $\xi$ 的弱函子性转移。

**猜想 15.11（Langlands 函子性，强形式）.** 在局部 LLC、packet 参数化和内形式数据均已固定的情形，$\Pi$ 可取为满足定义 15.6 的强转移；若转移目标不是单一表示而是 packet 或稳定分布，则应存在相应稳定转移，并满足局部字符恒等式。

**注 15.12.** 函子性不保证 cusp forms 转移为 cusp forms。若 $\xi\circ\varphi_\sigma$ 的像落入 ${}^LG$ 的 proper Levi subgroup，则转移通常属于 Eisenstein spectrum 或 `GL(N)` 的 isobaric sum，而不是 cuspidal representation。

## 15.4 L 函数的相容性

函子性的一个可检验后果是 L 函数恒等式。

**命题 15.13.** 设 $\Pi$ 是 $\sigma$ 沿 $\xi$ 的弱转移。设
$$
r:{}^LG\to\operatorname{GL}(V)
$$
为 L 群表示数据。则对足够大的有限集合 $S$，有
$$
L^S(s,\Pi,r)=L^S(s,\sigma,r\circ\xi).
$$

**证明.** 对 $v\notin S$，两侧均非分歧。由弱转移，
$$
s(\Pi_v)=\xi_v(s(\sigma_v)).
$$
因此
$$
r_v(s(\Pi_v))=(r_v\circ\xi_v)(s(\sigma_v)).
$$
两侧局部 L 因子是对应线性算子 characteristic polynomial 的倒数，故逐项相等。对所有 $v\notin S$ 相乘即得部分 L 函数相等。$\square$

**注 15.14.** 反向推理需要谨慎。若若干 L 函数相等，不一定自动得到函子性转移；通常还需要足够多的 twists、局部控制和 converse theorem。

## 15.5 目标为 `GL(N)` 的函子性与 converse theorem

设
$$
\xi:{}^LH\to{}^L\operatorname{GL}_N
$$
为 L 同态。若 $\sigma$ 是 $H(\mathbb A_K)$ 的自守表示，则局部 LLC for `GL(N)` 给出每个位置的候选局部表示
$$
\Pi_v=\xi_{v,*}\sigma_v.
$$
形式 restricted tensor product
$$
\Pi=\otimes_v'\Pi_v
$$
是 $\operatorname{GL}_N(\mathbb A_K)$ 的可容许表示候选对象。

**命题 15.15.** 若上述 $\Pi$ 是 automorphic，则它是 $\sigma$ 沿 $\xi$ 的强转移。

**证明.** 构造中每个局部分量 $\Pi_v$ 的局部参数定义为
$$
\varphi_{\Pi_v}=\xi_v\circ\varphi_{\sigma_v}.
$$
若 $\Pi$ 是自守表示，则它满足定义 15.6 的每个局部条件。$\square$

**外部输入定理 15.16（converse theorem 的函子性用途）.** 在许多 `GL(N)` 目标问题中，若候选对象 $\Pi=\otimes_v'\Pi_v$ 满足适当的中心特征、局部可容许性、单位性，并且对足够多的 cuspidal twists $\tau$，Rankin-Selberg L 函数
$$
L(s,\Pi\times\tau)
$$
具有解析延拓、函数方程和有界性条件，则 $\Pi$ 为 automorphic。因此 $\Pi$ 给出沿 $\xi$ 的 functorial lift。

**注 15.17.** 这解释了为什么第十三章的 L 函数解析性质和第十四章的 converse theorem 是函子性的技术核心。许多已知 lift 的证明不是直接构造自守形式，而是构造足够多 L 函数的函数方程，再应用 converse theorem。

**注 15.17.1.** Langlands-Shahidi 方法提供这些函数方程的一类重要来源：由 maximal parabolic 的 adjoint action 得到局部 $\gamma$ 因子和全局 L 函数解析性质，再把它们输入 converse theorem 或低阶 lift 的证明。附录 M 固定这一方法的局部系数语言。

## 15.6 Base change 与 automorphic induction

设 $E/K$ 为有限扩张。

**定义 15.18.** 对 $\operatorname{GL}_n$，base change 从 $K$ 到 $E$ 的局部参数描述为限制：
$$
\varphi_{\pi_v}:W_{K_v}'\to\operatorname{GL}_n(\mathbb C)
\quad\mapsto\quad
\varphi_{\pi_v}|_{W_{E_w}'}:W_{E_w}'\to\operatorname{GL}_n(\mathbb C)
$$
其中 $w\mid v$。若全局表示 $\operatorname{BC}_{E/K}(\pi)$ 存在，则其 $w$ 处局部参数应为上述限制。

**定义 15.19.** Automorphic induction 的局部参数描述为诱导：
$$
\varphi_{\sigma_w}:W_{E_w}'\to\operatorname{GL}_n(\mathbb C)
\quad\mapsto\quad
\operatorname{Ind}_{W_{E_w}'}^{W_{K_v}'}\varphi_{\sigma_w},
$$
其目标是 $\operatorname{GL}_{n[E_w:K_v]}(\mathbb C)$。在 $K_v$ 处，完整局部参数为
$$
\bigoplus_{w\mid v}\operatorname{Ind}_{W_{E_w}'}^{W_{K_v}'}\varphi_{\sigma_w},
$$
维数为 $n[E:K]$。若全局表示 $\operatorname{AI}_{E/K}(\sigma)$ 存在，则它应具有这些局部参数。

**外部输入定理 15.20（Arthur-Clozel，solvable base change 与 automorphic induction）.** 对 cyclic 扩张以及更一般的 solvable 扩张，`GL(n)` 的 base change 和 automorphic induction 在适当假设下存在，并满足几乎所有位置的局部参数限制或诱导相容性。

**注 15.21.** Base change 与 automorphic induction 是函子性的基本例子。它们在类域论中已经出现：$n=1$ 时，base change 是 Hecke 特征的 norm pullback，automorphic induction 对应 Weil 群表示的诱导。

## 15.7 对称幂、外方幂和张量积转移

对 $G=\operatorname{GL}_n$，对偶群也是 $\operatorname{GL}_n(\mathbb C)$。任意有限维代数表示
$$
r:\operatorname{GL}_n(\mathbb C)\to\operatorname{GL}_N(\mathbb C)
$$
给出 L 同态
$$
{}^L\operatorname{GL}_n\to{}^L\operatorname{GL}_N.
$$
函子性预期给出从 $\operatorname{GL}_n$ 到 $\operatorname{GL}_N$ 的 lift。

**定义 15.22.** 常见表示包括：

1. 对称幂表示
   $$
   \operatorname{Sym}^m:\operatorname{GL}_2(\mathbb C)\to\operatorname{GL}_{m+1}(\mathbb C).
   $$
2. 外方幂表示
   $$
   \wedge^k:\operatorname{GL}_n(\mathbb C)\to\operatorname{GL}_{\binom nk}(\mathbb C).
   $$
3. 张量积表示
   $$
   \operatorname{Std}_m\boxtimes\operatorname{Std}_n:
   \operatorname{GL}_m(\mathbb C)\times\operatorname{GL}_n(\mathbb C)
   \to
   \operatorname{GL}_{mn}(\mathbb C).
   $$
4. Adjoint 表示
   $$
   \operatorname{Ad}:\operatorname{GL}_n(\mathbb C)\to\operatorname{GL}(\mathfrak{sl}_n(\mathbb C)).
   $$

**外部输入定理 15.23（若干低阶 functorial lifts）.** 下列函子性转移在相应技术假设下已知：

1. Gelbart-Jacquet 的 symmetric square lift：
   $$
   \operatorname{GL}_2\to\operatorname{GL}_3.
   $$
2. Kim-Shahidi 和 Kim 的若干低阶 symmetric power lifts，例如 $\operatorname{Sym}^3$ 和 $\operatorname{Sym}^4$ for `GL(2)`。
3. 若干 tensor product lifts，例如 `GL(2)\times GL(2)` 到 `GL(4)`，以及若干 `GL(2)\times GL(3)` 类型情形。
4. 若干 exterior square lift 情形，例如 `GL(4)` 到 `GL(6)` 的外方幂转移。

**注 15.24.** 定理 15.23 不是任意 $m,n,k$ 的完整 functoriality。一般 symmetric power、exterior power 和 tensor product functoriality 仍属于广义 Langlands 函子性猜想。

**命题 15.25.** 若 $\operatorname{Sym}^m\pi$ 作为 `GL(m+1)` 的自守表示存在，则对足够大的 $S$，
$$
L^S(s,\operatorname{Sym}^m\pi,\operatorname{Std})
=
L^S(s,\pi,\operatorname{Sym}^m).
$$

**证明.** 这是命题 15.13 在
$$
\xi=\operatorname{Sym}^m:{}^L\operatorname{GL}_2\to{}^L\operatorname{GL}_{m+1},
\qquad
r=\operatorname{Std}
$$
下的特例。$\square$

## 15.8 Endoscopy 作为函子性的特殊形态

Endoscopy 涉及某个 endoscopic group $H$ 与目标群 $G$ 的 L 群关系。它不是单纯的表示到表示转移，而通常是稳定分布或 packet 之间的转移。

**定义 15.26.** 一个 endoscopic transfer 的接口数据包括：

1. endoscopic group $H$；
2. L 群同态或嵌入
   $$
   {}^LH\to{}^LG;
   $$
3. matching of conjugacy classes；
4. transfer factor；
5. 稳定 orbital integrals 与 characters 的恒等式。

**注 15.27.** Endoscopy 将在第十六章专门展开。本章只记录它与函子性的关系：endoscopic classification 通常通过 trace formula 证明某些从 $H$ 到 $G$ 或从 classical groups 到 `GL(N)` 的转移。

**外部输入定理 15.28（Arthur-Mok 型分类的函子性接口）.** 对若干 quasi-split classical groups 和 unitary groups，Arthur、Mok 及相关工作构造了到适当 `GL(N)` 的稳定转移，并用其描述离散自守谱。该转移在非分歧位置与标准 L 群嵌入给出的 Satake 参数推前相容。

## 15.9 Galois 表示侧的函子性

若某个自守表示 $\pi$ 已知对应 Galois 表示
$$
\rho_{\pi,\ell}:G_K\to\widehat G(\overline{\mathbb Q}_\ell)
$$
或对应到 L 群的 $\ell$-adic 形式。若 L 同态 $\xi:{}^LG\to{}^LH$ 的对偶群部分有 $\ell$-adic 实现 $\widehat\xi_\ell$，则 Galois 侧给出复合
$$
\widehat\xi_\ell\circ\rho_{\pi,\ell}.
$$

**命题 15.29.** 假设 $\pi$ 对应 $\rho_{\pi,\ell}$，且 $\Pi$ 是 $\pi$ 沿 $\xi:{}^LG\to{}^L\operatorname{GL}_N$ 的 functorial transfer。若 $\xi$ 的对偶群部分给出 $\widehat\xi_\ell:\widehat G\to\operatorname{GL}_N$，并且两侧在几乎所有非分歧位置满足 Frobenius-Satake 相容，则 $\Pi$ 对应的 Galois 表示应为
$$
\rho_{\Pi,\ell}\cong\widehat\xi_\ell\circ\rho_{\pi,\ell}
$$
的半单化。

**证明草图.** 对几乎所有 $v$，$\pi_v$ 与 $\Pi_v$ 非分歧。自守侧函子性给出 Satake 参数关系
$$
s(\Pi_v)=\xi(s(\pi_v)).
$$
Galois-自守相容把 $s(\pi_v)$ 与 $\rho_{\pi,\ell}(\operatorname{Frob}_v)$ 的 characteristic polynomial 对应，把 $s(\Pi_v)$ 与 $\rho_{\Pi,\ell}(\operatorname{Frob}_v)$ 对应。于是两侧 Frobenius characteristic polynomials 对几乎所有 $v$ 相同。由 Chebotarev density theorem，连续半单 Galois 表示由这些多项式确定。$\square$

**注 15.30.** 命题 15.29 是条件命题。数域上并非每个自守表示都已知有 Galois 表示；也并非每个 L 同态都已知保持代数性、纯性和局部 Hodge-Tate 条件。

## 15.10 本章小结

函子性原理说：自守表示之间的自然转移应由 L 群同态控制，而不是由原群之间的同态控制。弱转移由几乎所有 Satake 参数决定；强转移要求每个局部参数相容。目标为 `GL(N)` 时，强重数一保证弱转移唯一，converse theorem 可把候选局部数据提升为全局自守表示。Base change、automorphic induction、symmetric powers、tensor products、exterior powers 和 endoscopy 都是函子性的具体面向。一般函子性仍是 Langlands 纲领的核心开放问题之一。

## 练习

**练习 15.1.** 证明强转移推出弱转移。

**练习 15.2.** 用强重数一证明目标为 `GL(N)` 时弱转移唯一。

**练习 15.3.** 设 $\Pi$ 是 $\sigma$ 沿 $\xi$ 的弱转移。证明 $L^S(s,\Pi,r)=L^S(s,\sigma,r\circ\xi)$。

**练习 15.4.** 对二次扩张 $E/K$，描述 `GL(1)` 的 base change 与 automorphic induction 在 Hecke 特征上的作用。

**练习 15.5.** 对 $\operatorname{Sym}^2:\operatorname{GL}_2(\mathbb C)\to\operatorname{GL}_3(\mathbb C)$，写出非分歧 Satake 参数 $(\alpha,\beta)$ 的推前。

**练习 15.6.** 解释为什么 endoscopic transfer 不能只用单个表示之间的映射描述。

**练习 15.7.** 在已知 Galois 表示存在的情形，说明函子性如何对应 Galois 表示的复合。
