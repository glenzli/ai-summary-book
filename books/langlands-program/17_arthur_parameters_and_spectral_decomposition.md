# 第十七章：Arthur 参数和谱分解

## 本章目标

本章解释 Arthur 参数（Arthur parameters）在自守谱分解中的作用。局部 Langlands 参数描述单个局部表示的 L-packet；全局 Langlands 参数预期描述 tempered 或更一般的自守表示。但对 classical groups 的离散谱，非 tempered 表示系统性出现。Arthur 的思想是在 Langlands 参数之外加入一个额外的 $\operatorname{SL}_2(\mathbb C)$，用它编码非 tempered 性和残余谱来源，并通过稳定 trace formula 给出 multiplicity formula。

## 依赖前置知识

需要第十二章的 L-packet，第十三章的自守谱，第十五章的函子性，第十六章的 stable trace formula 和 endoscopy。需要知道 tempered representation、discrete automorphic spectrum、isobaric sum、component group 和 stable character。Arthur 对 classical groups 的分类、Mok 对 unitary groups 的分类、局部 Arthur packets 和 multiplicity formula 均作为外部输入。

## 17.1 离散谱与残余谱

设 $K$ 为整体域，$G/K$ 为 connected reductive group。自守商为
$$
[G]=G(K)\backslash G(\mathbb A_K).
$$

**定义 17.1.** 固定 unitary central character $\omega$。离散自守谱 $L^2_{\operatorname{disc}}(G,\omega)$ 是 $L^2([G],\omega)$ 中分解为不可约表示直和的闭子空间。它包含 cuspidal spectrum 和 residual spectrum。

**定义 17.2.** Cuspidal spectrum 是由尖点自守表示生成的部分。Residual spectrum 是由 Eisenstein series 的极点产生的离散谱部分。

**注 17.3.** 对 `GL(n)`，Mœglin-Waldspurger 描述了 residual spectrum；cuspidal representations 是构造 Rankin-Selberg L 函数的核心输入。对 classical groups，离散谱的组织更自然地由 Arthur 参数而不是单纯 cuspidal data 给出。

## 17.2 Tempered 猜想与 Arthur 的修正

**猜想 17.4（Ramanujan-tempered 预期，粗略形式）.** 对许多合适归一化的 cuspidal automorphic representations $\pi$，若 $v$ 非分歧，则 Satake 参数应在 tempered compact real form 中有有界像；等价地，局部分量 $\pi_v$ 应为 tempered。

**注 17.5.** 对一般群和一般数域，该猜想未完全证明。更重要的是，离散谱并不全由 tempered cuspidal representations 组成；residual spectrum 和 endoscopic lifts 产生非 tempered 离散表示。Arthur 参数正是为了系统描述这些非 tempered 离散谱。

## 17.3 局部 Arthur 参数

设 $F$ 为局部域，$G/F$ 为 connected reductive group。

**定义 17.6.** 一个局部 Arthur 参数是同态
$$
\psi:W_F'\times\operatorname{SL}_2(\mathbb C)\to{}^LG
$$
满足：

1. 其限制到 $W_F'$ 和额外 $\operatorname{SL}_2(\mathbb C)$ 满足与 L 参数类似的连续性和代数性条件。
2. $W_F$ 的像在 $\widehat G$ 中 bounded modulo center。
3. 第二个 $\operatorname{SL}_2(\mathbb C)$ 是 Arthur $\operatorname{SL}_2$，用于编码非 tempered 性。

**定义 17.7.** 由 Arthur 参数 $\psi$ 关联的 Langlands 参数 $\varphi_\psi$ 定义为
$$
\varphi_\psi(w,x)
=
\psi\left(w,x,
\begin{pmatrix}
|w|^{1/2}&0\\
0&|w|^{-1/2}
\end{pmatrix}
\right),
$$
其中 $w\in W_F$，$x\in\operatorname{SL}_2(\mathbb C)$，$|\cdot|$ 为第五章的 Weil norm character。

**注 17.8.** 若 Arthur $\operatorname{SL}_2$ 平凡，则 $\varphi_\psi$ tempered。若它非平凡，则 $\varphi_\psi$ 通常非 tempered。这一公式解释了 Arthur 参数如何产生非 tempered Langlands 参数。

**定义 17.9.** 局部 A-packet 是与 Arthur 参数 $\psi$ 关联的一组局部表示，记为
$$
\Pi_\psi(G/F).
$$
它通常是 Langlands L-packets 的并集或带权组合，而不必等于单个 L-packet。

## 17.4 全局 Arthur 参数

全局 Arthur 参数在实际分类中通常通过 `GL(n)` cuspidal data 表示。对 classical groups，标准嵌入
$$
{}^LG\to{}^L\operatorname{GL}_N
$$
把 Arthur 参数推到 `GL(N)` 的 isobaric 自守表示。

**定义 17.10（形式全局 Arthur 参数）.** 对 classical group 的一个全局 Arthur 参数可写为形式和
$$
\psi=\boxplus_i(\pi_i,b_i),
$$
其中：

1. $\pi_i$ 是 $\operatorname{GL}_{n_i}(\mathbb A_K)$ 的 cuspidal automorphic representation。
2. $b_i$ 是正整数，对应 Arthur $\operatorname{SL}_2$ 的 $b_i$ 维不可约表示。
3. 满足维数关系
   $$
   \sum_i n_ib_i=N
   $$
   其中 $N$ 由 $G$ 的标准 L 嵌入决定。
4. $\pi_i$ 满足适当 self-duality、正交或辛型符号条件。

**注 17.11.** 这是 Arthur 对 classical groups 的全局参数接口形式。完整定义需要区分 symplectic、special orthogonal、orthogonal、unitary groups，以及中心、outer automorphisms 和 pure inner forms。

**定义 17.12.** 若所有 $b_i=1$，则称 $\psi$ 为 tempered Arthur 参数。若存在 $b_i>1$，则 $\psi$ 对应非 tempered 贡献。

## 17.5 Component Group 与 Multiplicity Formula

设 $\psi$ 为全局 Arthur 参数。其 centralizer 给出有限 component group。

**定义 17.13.** 令
$$
S_\psi=\operatorname{Cent}_{\widehat G}(\operatorname{im}\psi),
$$
并定义 Arthur component group
$$
\mathcal S_\psi=\pi_0(S_\psi/Z(\widehat G)^{\Gamma_K}).
$$

**定义 17.14.** 全局 A-packet 形式上为 restricted product
$$
\Pi_\psi=\prod_v'\Pi_{\psi_v}(G/K_v),
$$
其中 $\psi_v$ 是 $\psi$ 在 $v$ 处的局部化。

**外部输入定理 17.15（Arthur multiplicity formula，接口形式）.** 对若干 quasi-split classical groups，离散谱分解可写为
$$
L^2_{\operatorname{disc}}(G)
\cong
\bigoplus_{\psi}
\bigoplus_{\pi\in\Pi_\psi}
m(\pi)\,\pi,
$$
其中 $\psi$ 遍历合适的全局 Arthur 参数，multiplicity $m(\pi)$ 由 $\mathcal S_\psi$ 的字符、局部 packet 参数化和全局 epsilon character 决定。

**注 17.16.** 定理 17.15 的完整版本包含稳定 trace formula、endoscopic transfer、局部字符恒等式和内形式修正。本书当前只把它作为后续谱分解和 functoriality 的接口。

## 17.6 Classical Groups 到 `GL(N)` 的标准转移

设 $G$ 为 classical group，其 L 群有标准表示
$$
\operatorname{Std}:{}^LG\to{}^L\operatorname{GL}_N.
$$

**外部输入定理 17.17（Arthur 标准转移，接口形式）.** 对 Arthur packet 中的离散自守表示 $\pi$，其到 `GL(N)` 的标准转移是 isobaric automorphic representation
$$
\boxplus_i\left(\pi_i\boxtimes [b_i]\right),
$$
其中 $\psi=\boxplus_i(\pi_i,b_i)$，而 $[b_i]$ 表示 $\operatorname{SL}_2(\mathbb C)$ 的 $b_i$ 维不可约表示在 `GL` 侧产生的 Speh 型或相应 isobaric 数据。

**注 17.18.** 该转移通常不是 cuspidal。Arthur $\operatorname{SL}_2$ 非平凡时，目标 `GL(N)` 表示反映 residual 或非 tempered 行为。

## 17.7 局部-全局相容性

**条件 17.19（Arthur 参数的局部化）.** 全局 Arthur 参数 $\psi$ 应在每个位置 $v$ 给出局部 Arthur 参数
$$
\psi_v:W_{K_v}'\times\operatorname{SL}_2(\mathbb C)\to{}^LG_v.
$$
全局 packet 中的表示
$$
\pi=\otimes_v'\pi_v
$$
应满足
$$
\pi_v\in\Pi_{\psi_v}(G/K_v)
$$
对所有 $v$ 成立。

**命题 17.20.** 若 $\psi$ tempered，则其关联的 Langlands 参数 $\varphi_{\psi_v}$ 在每个 $v$ 处 tempered。

**证明.** Tempered 意味着 Arthur $\operatorname{SL}_2$ 在定义 17.7 中平凡或所有 $b_i=1$。于是公式中额外的
$$
\begin{pmatrix}|w|^{1/2}&0\\0&|w|^{-1/2}\end{pmatrix}
$$
不产生非平凡权。参数在 $W_{K_v}$ 上的像 bounded modulo center，因此局部 Langlands 参数 tempered。$\square$

## 17.8 与 Ramanujan 猜想的关系

Arthur 参数把“非 tempered 离散谱”从一般 tempered 预期中分离出来。

**命题 17.21.** 若离散谱中的表示 $\pi$ 属于参数 $\psi=\boxplus_i(\pi_i,b_i)$，且某个 $b_i>1$，则 $\pi$ 的某些局部分量预期非 tempered。

**证明草图.** 当 $b_i>1$ 时，Arthur $\operatorname{SL}_2$ 的非平凡不可约表示在定义 17.7 中引入 $|w|$ 的非零幂。该幂改变 Langlands 参数在 Weil 群上的 boundedness，因此对应局部分量不再 tempered。完整陈述需要局部 A-packet 与 Langlands quotient 的关系。$\square$

**注 17.22.** 这并不反驳 cuspidal Ramanujan 猜想；它说明离散谱中 residual 和 endoscopic 贡献需要 Arthur 参数而不是单纯 tempered 参数来描述。

## 17.9 Arthur 参数与 L 函数极点

Arthur 参数中的块 $(\pi_i,b_i)$ 与 L 函数极点相关。

**外部输入定理 17.23（L 函数判别符号，接口形式）.** 对 classical groups，cuspidal representation $\pi_i$ 的 self-duality 类型可通过 symmetric square 或 exterior square L 函数在 $s=1$ 的极点判别。该符号决定 $\pi_i$ 是否能出现在给定 $G$ 的 Arthur 参数中。

**注 17.24.** 这解释了为什么第十五章中的 symmetric/exterior square L 函数不仅是函子性例子，也进入 classical groups 谱分解的参数条件。

## 17.10 本章小结

Arthur 参数是 Langlands 参数的谱分解增强版。额外的 $\operatorname{SL}_2(\mathbb C)$ 记录非 tempered 和 residual 现象；全局参数 $\boxplus_i(\pi_i,b_i)$ 把 classical groups 的离散谱与 `GL(n)` cuspidal data 连接起来。Arthur multiplicity formula 用 component group 字符决定 packet 中哪些表示以何重数出现。该理论依赖稳定 trace formula 和 endoscopic transfer，是现代 classical groups Langlands 理论的核心。

## 练习

**练习 17.1.** 解释 Arthur 参数与 Langlands 参数的区别。

**练习 17.2.** 写出由局部 Arthur 参数 $\psi$ 产生 Langlands 参数 $\varphi_\psi$ 的公式。

**练习 17.3.** 对形式参数 $\psi=\boxplus_i(\pi_i,b_i)$，说明维数关系 $\sum_i n_ib_i=N$ 的来源。

**练习 17.4.** 解释为什么 $b_i>1$ 通常产生非 tempered 局部分量。

**练习 17.5.** 说明 component group 在 multiplicity formula 中的作用。

**练习 17.6.** 解释 classical groups 到 `GL(N)` 的标准转移为什么可能不是 cuspidal。
