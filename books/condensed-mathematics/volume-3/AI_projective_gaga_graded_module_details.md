# 附录 AI：Projective GAGA 的 graded module 细节

## AI.0 目标

附录 Y 给出 projective GAGA 的证明结构。本附录补充 projective space 上的 graded module 模型，说明 coherent sheaf 如何由有限生成 graded module 控制，并说明解析化 full faithfulness 与 essential surjectivity 的代数骨架。

完整 GAGA 仍以 Serre theorem 和解析有限生成定理为输入。

## AI.1 代数侧 Serre correspondence

设

$$
S=\mathbb C[X_0,\ldots,X_n],
\qquad
\mathbb P^n=\operatorname{Proj}S.
$$

**输入定理 AI.1（Serre correspondence）.** coherent algebraic sheaves on \(\mathbb P^n\) 等价于有限生成 graded \(S\)-modules modulo finite length torsion 的范畴。更具体地：

1. 有限生成 graded module \(M\) 给出 coherent sheaf \(\widetilde M\)；
2. 对任意 coherent sheaf \(\mathcal F\)，graded module
   $$
   \Gamma_\ast(\mathcal F)=\bigoplus_{d\in\mathbb Z}H^0(\mathbb P^n,\mathcal F(d))
   $$
   在足够高次数有限生成；
3. 自然映射
   $$
   \widetilde{\Gamma_\ast(\mathcal F)}\to\mathcal F
   $$
   为同构。

## AI.2 解析侧输入

**输入定理 AI.2（解析 Serre correspondence）.** 对 \((\mathbb P^n)^{an}\) 上相干解析层 \(\mathcal G\)，graded vector space

$$
\Gamma_\ast^{an}(\mathcal G)
=
\bigoplus_d H^0((\mathbb P^n)^{an},\mathcal G(d))
$$

带有自然 \(S\)-module 结构，并在足够高次数有限生成。且

$$
\widetilde{\Gamma_\ast^{an}(\mathcal G)}^{an}\to\mathcal G
$$

为同构。

AI.2 是 GAGA 的分析核心之一；其证明使用 Cartan A/B、有限生成和 twisting。

## AI.3 上同调比较到 graded module 比较

**输入定理 AI.3（twist cohomology comparison）.** 对 algebraic coherent sheaf \(\mathcal F\) 和所有 \(d,q\)，自然映射

$$
H^q(\mathbb P^n,\mathcal F(d))
\to
H^q((\mathbb P^n)^{an},\mathcal F(d)^{an})
$$

为同构。

**命题 AI.4.** 在 AI.3 下，

$$
\Gamma_\ast(\mathcal F)\cong
\Gamma_\ast^{an}(\mathcal F^{an})
$$

作为 graded \(S\)-modules。

**证明.** 逐次数取 \(q=0\) 的 AI.3，并检查乘以 \(X_i\) 的结构映射来自同一全局截面 \(\mathcal O(1)\)。证毕。

## AI.4 Full faithfulness

**定理 AI.5.** 在 AI.1-AI.4 下，解析化函子

$$
\operatorname{Coh}(\mathbb P^n)
\to
\operatorname{Coh}((\mathbb P^n)^{an})
$$

full faithful。

**证明.** 对 \(\mathcal F,\mathcal G\)，选择足够大的 \(m\)，使 \(\mathcal F(m)\) 由全局截面生成，并有有限 presentation

$$
\mathcal O(-b)^{\oplus s}\to\mathcal O(-a)^{\oplus r}\to\mathcal F\to0.
$$

对 \(\mathcal Hom(-,\mathcal G)\) 应用得到 Hom 的 kernel 表示。解析化保持该 presentation，AI.3 对 \(\mathcal G(a)\)、\(\mathcal G(b)\) 的 \(H^0\) 比较给两个有限维 Hom 空间比较同构。kernel 在同构下对应，故 Hom 比较同构。证毕。

## AI.5 Essential surjectivity

**定理 AI.6.** 在 AI.1-AI.2 下，每个相干解析层 \(\mathcal G\) 同构于某个 algebraic coherent sheaf 的解析化。

**证明.** 由 AI.2，取有限生成 graded \(S\)-module

$$
M=\Gamma_\ast^{an}(\mathcal G)_{\ge N}
$$

的高次截断。由 AI.1，\(M\) 给出 algebraic coherent sheaf \(\widetilde M\)。解析化后由 AI.2 的重构同构得到

$$
(\widetilde M)^{an}\cong\mathcal G.
$$

证毕。

## AI.6 Projective variety 情形

若 \(X\subset\mathbb P^n\) 是 projective variety，则 coherent sheaves on \(X\) 可看作 \(\mathbb P^n\) 上由理想 sheaf 支撑控制的 coherent sheaves。GAGA for \(X\) 可由 \(\mathbb P^n\) 情形和 closed immersion 的 exactness 推出。

## 练习

1. 对 \(\mathcal O(d)\)，写出 \(\Gamma_\ast(\mathcal O(d))\)。
2. 解释 finite length torsion 为什么 sheafification 后消失。
3. 在 AI.5 中写出 Hom kernel 的具体两项公式。
4. 说明 AI.6 为什么需要高次截断。
