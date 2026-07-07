# 第三章：Prismatic cohomology 与基础比较定理

## 本章目标

本章把第二章定义的 $R\Gamma_\Delta(X/A)$ 与 classical cohomology theories 联系起来。所有深层比较定理均标为外部输入；本章的内部工作是固定对象、态射、specialization、滤过和 Frobenius 结构的精确口径。

## 依赖前置知识

依赖第二章的 bounded prism、relative prismatic site、$\mathcal O_\Delta$、$\overline{\mathcal O}_\Delta$ 和 Frobenius。需要熟悉 de Rham complex、crystalline cohomology、etale cohomology 和 derived tensor product 的基本语言。

## 3.1 Affine 记号

**约定 3.1.** 若 $X=\operatorname{Spf}(R)$ 是 affine $p$-adic formal scheme over $A/I$，则记
$$
\Delta_{R/A}=R\Gamma_\Delta(X/A).
$$
若 $R$ 是 $p$-completely smooth over $A/I$，则 $\mathbb L_{R/(A/I)}$ 可由 $\Omega^1_{R/(A/I)}$ 表示。

**定义 3.2.** Hodge-Tate specialization complex 定义为
$$
\overline\Delta_{R/A}=\Delta_{R/A}\otimes_A^L A/I.
$$
De Rham specialization complex 定义为
$$
\Delta^{\mathrm{dR}}_{R/A}=\phi_A^\ast\Delta_{R/A}\otimes_A^L A/I,
$$
其中 $\phi_A^\ast\Delta_{R/A}=A\otimes_{A,\phi_A}^L\Delta_{R/A}$。

**警告 3.3.** Hodge-Tate specialization 与 de Rham specialization 的差异在于是否先沿 $\phi_A$ pullback。后续任何公式若省略此差异，都必须视为未校验。

## 3.2 Hodge-Tate comparison

**外部输入定理 3.4（Hodge-Tate comparison）.** 令 $(A,I)$ 为 bounded prism，令 $X$ 为 smooth $p$-adic formal scheme over $A/I$。则 $\overline\Delta_{X/A}$ 带自然 conjugate filtration，其 associated graded 在标准 twist convention 下由 cotangent complex 给出：
$$
\operatorname{gr}^{i}_{\mathrm{conj}}\overline\Delta_{X/A}
\simeq
R\Gamma\left(X,\wedge^i\mathbb L_{X/(A/I)}\right)[-i]\{-i\}.
$$
若 $X$ smooth，则可写成
$$
\operatorname{gr}^{i}_{\mathrm{conj}}\overline\Delta_{X/A}
\simeq
R\Gamma\left(X,\Omega^i_{X/(A/I)}\right)[-i]\{-i\}.
$$

**说明 3.5.** 这里 $M\{-i\}$ 的符号依赖 $I/I^2$ 的 dual 或 inverse convention。本书当前 convention 采用 `NOTATION.md` 中的 $M\{i\}=M\otimes(I/I^2)^{\otimes i}$；最终版本必须通过 locator 与 Bhatt-Scholze 原文逐项核对符号方向。

**形式推论 3.6.** 若 $X$ proper smooth over $A/I$，且各 $R\Gamma(X,\Omega^i)$ 是 perfect complex，则 $\overline\Delta_{X/A}$ 是 $A/I$ 上的 perfect complex。

**证明.** Conjugate filtration 的 associated graded 是有限个 perfect complex。Perfect complexes 在有限扩张和有限滤过下稳定。故结论成立。证毕。

## 3.3 De Rham comparison

**外部输入定理 3.7（de Rham comparison）.** 在定理 3.4 的假设下，存在自然拟同构
$$
\phi_A^\ast R\Gamma_\Delta(X/A)\otimes_A^L A/I
\simeq
R\Gamma_{\mathrm{dR}}(X/(A/I)).
$$
该同构与乘法结构相容，并可在适当模型中提升为 commutative differential graded algebra 层面的同构。

**警告 3.8.** 若去掉左侧的 $\phi_A^\ast$，通常得到的是 Hodge-Tate specialization 而不是 de Rham cohomology。这个差异是 prismatic theory 中最常见的公式错误之一。

## 3.4 Crystalline comparison

**外部输入定理 3.9（crystalline comparison）.** 令 $(A,I)$ 为 bounded prism，且 $I=(p)$。若 $X$ smooth over $A/p$，则 prismatic cohomology 与 crystalline cohomology 存在自然 $\phi$-相容比较同构。精确地说，在 Bhatt-Scholze 的 convention 下，$R\Gamma_\Delta(X/A)$ 给出 crystalline cohomology 的 Frobenius descent。

**说明 3.10.** 本书当前版本暂不把该定理写成唯一公式，因为 crystalline cohomology 的底环和 divided power envelope convention 会影响表述。最终版本需要在 locator 阶段固定 Berthelot crystalline site 的口径。

## 3.5 Etale comparison

**外部输入定理 3.11（etale comparison）.** 令 $(A,I)$ 为 perfect bounded prism，$X$ smooth proper over $A/I$，并令 $X_\eta$ 表示相应 generic fibre。则 prismatic cohomology 在 invert $I$、取 Frobenius fixed points 并 modulo $p^n$ 后，与 $X_\eta$ 的 $p$-adic etale cohomology 比较。该比较为自然拟同构，并与 cup product 相容。

**警告 3.12.** 定理 3.11 不能简写为
$$
R\Gamma_\Delta(X/A)\cong R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z_p).
$$
左侧是 $A$-complex 带 Frobenius，右侧是 $\mathbf Z_p$-complex 带 Galois/pro-etale 信息；二者通过 invert $I$ 和 Frobenius fixed construction 才比较。

## 3.6 Base change

**外部输入定理 3.13（prismatic base change）.** 令 $(A,I)\to(B,J)$ 为 bounded prisms 的态射，令 $X$ 为 smooth $p$-adic formal scheme over $A/I$，并设
$$
Y=X\times_{\operatorname{Spf}(A/I)}\operatorname{Spf}(B/J).
$$
则存在自然 base-change comparison
$$
R\Gamma_\Delta(X/A)\widehat\otimes_A^L B
\simeq
R\Gamma_\Delta(Y/B),
$$
其中左侧完备化按 derived $(p,J)$-adic completion 解释。

**形式推论 3.14.** 若 $R\Gamma_\Delta(X/A)$ 为 perfect $A$-complex，则 base change 后得到的 $R\Gamma_\Delta(Y/B)$ 为 perfect $B$-complex。

**证明.** Perfect complexes 在 derived base change 下保持 perfect；derived completion 在 bounded prism 的假设下与目标完备范畴相容。关键相容性属于定理 3.13 的内容。证毕。

## 3.7 统一图式

**说明 3.15.** 对 smooth proper $X$，prismatic cohomology 的作用可以概括为如下图式，其中箭头均需按相应外部输入定理解释：
$$
\begin{array}{ccc}
R\Gamma_\Delta(X/A) & \xrightarrow{\;\mathrm{HT}\;} & \operatorname{gr}\text{-pieces from }\Omega^\bullet_X \\
\downarrow\mathrm{dR} & & \\
R\Gamma_{\mathrm{dR}}(X/(A/I)) & & \\
\downarrow\mathrm{crys}\text{ in }I=(p)\text{ case} & & \\
R\Gamma_{\mathrm{crys}}(X/A) & & \\
\downarrow\mathrm{etale}\text{ in perfect case} & & \\
R\Gamma_{\mathrm{et}}(X_\eta,\mathbf Z_p) & &
\end{array}
$$
该图不是一个单一交换图，而是一组在不同 base prism 和 specialization 下成立的比较。

## 3.8 比较态射的类型检查

**定义 3.16.** 本书把 comparison statement 分为四类：

1. **specialization isomorphism**：由 $R\Gamma_\Delta(X/A)$ 经 base change 得到目标 cohomology；
2. **filtered comparison**：比较同时保留 filtration；
3. **Frobenius comparison**：比较保留 Frobenius 或 Frobenius-semilinear structure；
4. **fixed-point comparison**：目标由 Frobenius fixed 或 fibre construction 得到。

**命题 3.17.** Etale comparison 属于 fixed-point comparison，而 de Rham comparison 属于 specialization isomorphism with filtration。

**证明.** De Rham comparison 的形式是
$$
\phi_A^\ast R\Gamma_\Delta(X/A)\otimes_A^L A/I\simeq R\Gamma_{\mathrm{dR}}(X/(A/I)),
$$
并保留 Hodge filtration。Etale comparison 需要先 invert $I$，再通过 $\varphi=1$ 的 derived fibre construction 提取 $\mathbf Z_p$-信息。两者使用不同的操作，故分类不同。证毕。

**警告 3.18.** 若一个证明把 fixed-point comparison 当作普通 base change comparison 使用，则它通常会丢失 derived fixed points 中的 cokernel 项。

## 本章小结

本章定义了 Hodge-Tate 和 de Rham specialization，列出 prismatic cohomology 的基础比较定理，并明确这些定理都是外部输入。书内可证明的是滤过、perfectness 和 base change 的形式推论；核心比较同构本身依赖 Bhatt-Scholze。

## 练习

**练习 3.1.** 解释为什么 $\Delta_{R/A}\otimes_A^L A/I$ 与 $\phi_A^\ast\Delta_{R/A}\otimes_A^L A/I$ 不是同一个 construction。

**练习 3.2.** 在 $X$ proper smooth 且 $R\Gamma(X,\Omega^i)$ perfect 的假设下，补全形式推论 3.6 中“有限滤过保持 perfect”的证明。

**练习 3.3.** 写出定理 3.11 的错误简写版本，并指出每个对象的系数环和附加结构为什么不匹配。
