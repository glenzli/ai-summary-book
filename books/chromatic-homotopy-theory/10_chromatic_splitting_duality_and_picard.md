# 第十章：Chromatic splitting、Gross-Hopkins duality 与 Picard groups

## 本章目标

本章讨论 $K(n)$-local category 的三个结构性问题：chromatic splitting、Gross-Hopkins duality 和 Picard group。它们都位于基础 chromatic tower 之后，依赖 Morava stabilizer descent、local duality 和连续群上同调。

## 依赖前置知识

需要第五章的 fracture square、第六章的 Morava descent 和第九章的前沿分层。Brown-Comenetz dual、local cohomology 和 Picard spectrum 作为外部背景。

## 10.1 Chromatic splitting problem

**定义 10.1.** Chromatic splitting problem 研究 fracture square 中的低高度重叠项
$$
L_{n-1}L_{K(n)}X
$$
以及映射 $L_{n-1}X\to L_{n-1}L_{K(n)}X$ 的分裂行为。最核心对象是 $X=\mathbb S$。

**警告 10.2.** Chromatic splitting 与 telescope conjecture 是不同问题。前者研究 chromatic fracture square 的重叠和分裂；后者比较 finite/telescopic localization 与 chromatic localization。

**外部输入 10.3 (rational splitting).** Barthel-Schlank-Stapleton-Weinstein 计算了所有高度和素数下 $K(n)$-local sphere 的有理同伦群，并在有理层面验证了 Hopkins chromatic splitting conjecture 的预测。整数层面的完整 splitting 仍需分开处理。

**命题 10.4.** 若对某个 $n$ 和 $X$ 有 $L_{n-1}L_{K(n)}X\simeq0$，则 fracture square 退化为乘积
$$
L_nX\simeq L_{n-1}X\times L_{K(n)}X.
$$

**证明.** 这是命题 5.13 的直接应用。关键是这里的假设通常不成立，因此命题只是边界计算工具。证毕。

## 10.2 Brown-Comenetz dual 和 Gross-Hopkins duality

**定义 10.5.** Brown-Comenetz dual $I$ 是表示函子
$$
X\longmapsto \operatorname{Hom}(\pi_0X,\mathbb Q/\mathbb Z)
$$
的谱。对谱 $X$，其 Brown-Comenetz dual 记为
$$
IX=F(X,I).
$$

**定义 10.6.** 第 $n$ 个 monochromatic Brown-Comenetz dualizing object 在本章记为 $I_n^{GH}$，定义为
$$
I_n^{GH}=I M_n\mathbb S
$$
或按文献采用等价的 $K(n)$-local dualizing object convention。不同 convention 会产生悬挂和 determinant twist 的差异。

**外部输入定理 10.7 (Gross-Hopkins duality).** 在 $K(n)$-local category 中，Brown-Comenetz/monochromatic duality 可由 Morava module 的 determinant twist 和悬挂描述。精确公式依赖素数、高度、$E_n$ 的 convention 和 dualizing object 定义。

**使用限制 10.8.** 本书当前不写未定位的完整公式。任何类似
$$
I_n^{GH}\simeq \Sigma^a S\langle \det\rangle
$$
的陈述都必须指定 $a$、determinant sphere、prime/height 范围和 convention 来源。

## 10.3 Picard group

**定义 10.9.** $K(n)$-local Picard group 为
$$
\operatorname{Pic}_{K(n)}=\pi_0\operatorname{Pic}(\mathbf{Sp}_{K(n)}),
$$
即 $K(n)$-local spectra 中可逆对象的等价类。乘法由 smash product 给出。

**定义 10.10.** 若 $X\in\operatorname{Pic}_{K(n)}$，其 Morava module 为
$$
(E_n)_*X,
$$
这是带连续 $\mathbb G_n$-作用的 invertible $(E_n)_*$-module。由此得到比较映射
$$
\operatorname{Pic}_{K(n)}\longrightarrow \operatorname{Pic}_{\mathbb G_n}((E_n)_*).
$$

**定义 10.11.** 比较映射的 kernel 中元素常称为 exotic Picard elements：它们在 Morava module 层面看起来像球谱悬挂，但在 $K(n)$-local category 中不等价于球谱悬挂。

**外部输入 10.12.** Hopkins-Mahowald-Sadofsky、Hovey-Sadofsky、Goerss-Henn-Mahowald-Rezk、Mor、Devalapurkar 等工作给出多个高度和素数下 Picard group、exotic Picard group 和 descent spectral sequence 的计算与解释。

**警告 10.13.** Picard group 的“代数部分”和“exotic 部分”依赖高度、素数和 comparison theorem。不能从 Morava module 为 rank-one 直接推出谱可逆对象分类完成。

## 10.4 Descent spectral sequence for Picard spectra

**外部输入 10.14.** 存在 Picard spectrum 的 descent spectral sequence，用于从 $E_n$-local/profinite Galois descent 计算 $\operatorname{Pic}_{K(n)}$。它与 $E_n$-Adams spectral sequence 有紧密关系，并可比较某些 differential。

**检查表 10.15.** 使用 Picard descent 前必须记录：

1. 使用的是 Picard group 还是 Picard spectrum；
2. 是否在 $K(n)$-local category；
3. $\mathbb G_n$ action 的模型；
4. $E_2$ 页是连续群上同调还是其他 descent cohomology；
5. exotic part 的定义和检测方式。

## 10.5 Picard group 的代数近似

**定义 10.16.** Picard group 的代数近似是目标群
$$
\operatorname{Pic}_{\mathbb G_n}((E_n)_*)
$$
中的可逆 Morava modules。它包含 degree shift、rank-one module 和 stabilizer character 数据。

**命题 10.17.** 若 $X$ 是 $K(n)$-local invertible spectrum，则 $(E_n)_*X$ 是 invertible Morava module。

**证明.** 若 $X$ 可逆，则存在 $Y$ 使
$$
X\otimes Y\simeq \mathbb S_{K(n)}.
$$
张量 $E_n$ 并取 homotopy，得到
$$
(E_n)_*X\otimes_{(E_n)_*}(E_n)_*Y\cong (E_n)_*
$$
这里调用本章登记的 completed Kunneth 同构；其收敛和完备性条件是本命题
的外部输入。该同构表明 $(E_n)_*Y$ 是 $(E_n)_*X$ 的张量逆，故
$(E_n)_*X$ 为可逆 Morava module。$E_n$ 上的 $\mathbb G_n$-作用与
$E_n\otimes X$ 的函子性给出 semilinear action，并与张量乘法相容。
因此它确实是定义 10.16 中的可逆对象。证毕。

**警告 10.18.** 命题 10.17 给出比较映射，不给出满射或单射。kernel 正是 exotic Picard phenomena 的来源之一。

## 10.6 Gross-Hopkins duality 的使用格式

**规则 10.19.** 引用 Gross-Hopkins duality 时必须写成如下形式：

$$
I_n^{GH}\simeq \Sigma^a S\langle\det\rangle\otimes P
$$

并逐项说明 $a$、determinant sphere、exotic factor $P$、素数范围和文献 convention。若某项未知或不需要，应保留符号而不是删除。

**例 10.20.** “$I_n^{GH}$ 是球谱的悬挂”不是可接受陈述，除非已经证明 determinant twist 和 exotic factor 在当前 $(n,p)$ 下都平凡。

## 本章小结

Chromatic splitting 研究 fracture square 的重叠项，Gross-Hopkins duality 描述 $K(n)$-local dualizing object，Picard group 分类 $K(n)$-local 可逆谱。这三者都不是基础形式推论，而是 Morava stabilizer descent、连续群上同调和 local duality 的深层应用。

## 练习

**练习 10.1.** 解释为什么 $\operatorname{Pic}_{K(n)}$ 中的悬挂球谱给出一个 $\mathbb Z$ 或其周期商的自然子群。

**练习 10.2.** 说明 exotic Picard element 为什么不能只通过 $(E_n)_*$ 作为普通模来检测。

**练习 10.3.** 对高度 $1$，查阅 $K(1)$-local Picard group 的基本形状，并标明哪些部分来自悬挂，哪些来自 $p$-adic character。

**练习 10.4.** 写出 chromatic fracture square 中可能导致 splitting problem 的右下角项。
