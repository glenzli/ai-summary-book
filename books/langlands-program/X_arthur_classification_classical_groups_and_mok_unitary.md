# 附录 X：Arthur 分类、Classical Groups 和 Mok 的 Unitary Groups 接口

收口归一化回指：本附录的 Arthur 参数、standard transfer、multiplicity formula 和内形式修正依赖 Satake 与 transfer convention；见 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 4、8 节。

## X.1 Classical groups 的范围

本附录补充第十七章。目标不是重证 Arthur 或 Mok 的分类，而是固定哪些群、哪些参数和哪些 multiplicity formula 在本书中作为外部输入使用。

**定义 X.1.** 本附录中的 quasi-split classical groups 指如下类型：

1. symplectic groups $\operatorname{Sp}_{2n}$；
2. split 或 quasi-split special orthogonal groups $\operatorname{SO}_m$；
3. quasi-split unitary groups $U_n$ attached to a quadratic extension $E/F$。

其 L 群的标准表示记为
$$
\operatorname{Std}:{}^LG\to\operatorname{GL}_N(\mathbb C).
$$

**定义 X.2.** 一个 global Arthur parameter 的接口形式为形式和
$$
\psi=\boxplus_i(\pi_i,b_i),
$$
其中 $\pi_i$ 是某个 $\operatorname{GL}_{n_i}(\mathbb A_K)$ 的 cuspidal automorphic representation，$b_i\ge1$，并满足 self-duality sign、central character 和 dimension condition，使得
$$
\sum_i n_i b_i=N.
$$

**注 X.3.** 精确定义需要 Arthur 的 $L^2$-parameter、global Arthur group 或等价的形式化数据。本书使用 $\boxplus_i(\pi_i,b_i)$ 作为接口符号。

## X.2 Self-duality sign 和 L 函数判别

**定义 X.4.** Cuspidal representation $\pi$ of $\operatorname{GL}_n(\mathbb A_K)$ 称为 self-dual，若 $\pi^\vee\simeq\pi$。其 orthogonal 或 symplectic type 由
$$
L(s,\pi,\operatorname{Sym}^2)
\quad\text{和}\quad
L(s,\pi,\wedge^2)
$$
在 $s=1$ 的极点判别。

**外部输入定理 X.5（self-duality sign criterion）.** 若 $\pi$ 是 self-dual cuspidal automorphic representation of $\operatorname{GL}_n(\mathbb A_K)$，则恰有一个标准情形可发生：symmetric square 或 exterior square L 函数在 $s=1$ 有极点。该极点决定 $\pi$ 的 orthogonal 或 symplectic type，并控制 $\pi$ 能进入哪类 classical group 的 Arthur parameter。

**命题 X.6.** Arthur parameter 的 dimension condition 与标准转移的 degree 相容。

**证明.** 参数块 $(\pi_i,b_i)$ 在标准转移到 $\operatorname{GL}_N$ 时贡献 $\pi_i$ 与 $\operatorname{SL}_2(\mathbb C)$ 的 $b_i$ 维不可约表示组合。其 degree 为 $n_i b_i$。所有块相加得到 degree
$$
\sum_i n_i b_i.
$$
定义 X.2 要求该和为标准表示维数 $N$，因此与到 $\operatorname{GL}_N$ 的标准转移相容。$\square$

## X.3 Arthur packets 和 component groups

**定义 X.7.** 对局部 Arthur parameter
$$
\psi_v:W_{K_v}'\times\operatorname{SL}_2(\mathbb C)\to{}^LG,
$$
定义 component group 的接口形式为
$$
\mathcal S_{\psi_v}=\pi_0(Z_{\widehat G}(\operatorname{im}\psi_v)/Z(\widehat G)^{W_{K_v}}).
$$
局部 Arthur packet 记为
$$
\Pi_{\psi_v}(G/K_v).
$$

**外部输入定理 X.8（local Arthur packets）.** 对 quasi-split symplectic 和 special orthogonal groups，Arthur 构造局部 packets $\Pi_{\psi_v}$，并给出由 $\mathcal S_{\psi_v}$ 的 characters 控制的内部参数化。对 unitary groups，Mok 及后续工作给出相应版本。

**命题 X.9.** 若 $\psi_v$ 的 Arthur $\operatorname{SL}_2$ 因子平凡，则 $\psi_v$ 给出普通 Langlands parameter。

**证明.** 若 Arthur $\operatorname{SL}_2$ 因子平凡，则 $\psi_v$ 只依赖 $W_{K_v}'$。于是它就是一个 homomorphism
$$
W_{K_v}'\to{}^LG
$$
满足局部 Langlands parameter 的形式条件。Arthur packet 在这种情形应与对应的 tempered L-packet 相容。$\square$

## X.4 Multiplicity formula

**外部输入定理 X.10（Arthur multiplicity formula）.** 对 quasi-split symplectic 和 special orthogonal groups，离散谱按 global Arthur parameters 分解：
$$
L^2_{\operatorname{disc}}(G(K)\backslash G(\mathbb A_K))
=
\widehat\bigoplus_{\psi}
\widehat\bigoplus_{\pi\in\Pi_\psi}
m_\psi(\pi)\,\pi,
$$
其中 $m_\psi(\pi)$ 由 global component group 的 character pairing 和 Arthur 的 sign character 决定。Unitary groups 有 Mok 版本。

**命题 X.11.** Multiplicity formula 迫使 packet 内部参数化进入全局谱分解。

**证明.** 若只知道每个位置的 coarse packet 集合，而不知道 packet 成员对应的 component group character，则无法计算
$$
m_\psi(\pi).
$$
定理 X.10 的 multiplicity 依赖局部 characters 的乘积与 global sign character 的比较。因此 enhanced packet data 不是局部 LLC 的附加装饰，而是全局离散谱 multiplicity 的输入。$\square$

## X.5 Standard transfer to `GL(N)`

**外部输入定理 X.12（standard endoscopic transfer）.** 对上述 classical groups，Arthur-Mok 分类给出到 $\operatorname{GL}_N$ 的标准转移。若 $\pi\in\Pi_\psi$，其转移的 isobaric 形状由
$$
\psi=\boxplus_i(\pi_i,b_i)
$$
决定。

**命题 X.13.** Standard transfer 在非分歧位置等于 L 群标准表示推前。

**证明.** 在非分歧位置，packet 含有 spherical 成员，Satake parameter 为
$$
s_v\rtimes\operatorname{Fr}_v\in{}^LG.
$$
标准 L 同态 $\operatorname{Std}$ 把它送到 $\operatorname{GL}_N(\mathbb C)$ 中的半单共轭类。Arthur-Mok 的标准转移要求 $\operatorname{GL}_N$ 侧 spherical representation 的 Satake parameter 正是该像。因此非分歧局部 L 因子相等。$\square$

## X.6 Inner forms 和 Kaletha-Arthur 接口

**外部输入定理 X.14（inner form refinements）.** 对内形式或 pure inner forms，Arthur packet 和 multiplicity formula 需要加入 Kottwitz 符号、rigid inner twists、transfer factor normalization 和 refined endoscopic data。Kaletha 及相关工作给出现代 refined local Langlands 接口。

**命题 X.15.** 内形式修正不能从 quasi-split packet 事后忽略。

**证明.** 内形式 $G'$ 的局部表示可能与 quasi-split $G$ 共享同一个 L parameter，但在不同 inner form 上出现。若不记录 Kottwitz 或 rigid inner twist 数据，就不能判断某个 enhanced parameter 属于哪一个 $G'(F)$。全局 multiplicity formula 也要在所有相关局部内形式上取 restricted tensor product。因此内形式修正是 packet 参数化的一部分。$\square$

## X.7 与 beyond endoscopy 的关系

**命题 X.16.** Arthur 分类说明 endoscopy 是离散谱组织机制，而 beyond endoscopy 试图反向用 L 函数极点识别来源。

**证明.** Arthur 分类从 stable trace formula 和 endoscopy 出发，把 classical groups 的离散谱分解为参数 $\psi=\boxplus_i(\pi_i,b_i)$。这些参数中的 $\pi_i$ 来自 `GL(n)`，并由 self-dual L 函数极点判别其类型。Beyond endoscopy 的思想是通过 trace formula 中加入 L 函数权重或极点检测，直接识别这些 functorial 来源。两者方向不同，但识别的结构对象相同。$\square$

## 练习

**练习 X.1.** 对 $\psi=(\pi,1)$，说明 Arthur 参数何时是 tempered。

**练习 X.2.** 解释 component group character 为什么进入 multiplicity formula。

**练习 X.3.** 写出 standard transfer 在非分歧 Satake 参数上的公式。

**练习 X.4.** 说明 unitary groups 的 Mok 分类和 Arthur 正交/辛分类的关系。

**练习 X.5.** 解释内形式为何需要 rigid inner twist 数据。
