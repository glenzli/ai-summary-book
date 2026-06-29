# 附录 F：伴随函子与投影公式的形式骨架

## F.0 目标

第二卷第七章使用 $f_!$、$f^!$ 和投影公式。本附录证明其中不依赖 Scholze 具体构造的范畴论部分：

1. 在可展示稳定范畴中，余极限保持函子有右伴随。
2. 在紧生成三角范畴中，保持小直和的精确函子在 Brown representability 假设下有右伴随。
3. 投影公式可被组织为“张量作用与左伴随相容”的自然变换。
4. 一旦投影公式成立，右伴随 $f^!$ 满足相应的内部 Hom 公式。

Scholze 理论真正输入的是：具体几何情形下 $f_!$ 的构造、它保持所需余极限/紧对象、以及投影公式自然变换是等价。本附录只证明这些输入之后的形式推论。

## F.1 可展示稳定范畴中的右伴随

**定义 F.1.** 稳定 $\infty$-category $\mathcal C$ 称为可展示（presentable），如果它可由小集合对象在小余极限和可访问 colimit 下生成，并且有所有小极限和小余极限。

**定理 F.2（presentable adjoint functor theorem）.** 设

$$
F:\mathcal C\to\mathcal D
$$

是可展示 $\infty$-category 之间的函子。若 $F$ 保持所有小余极限并且可访问，则 $F$ 有右伴随

$$
G:\mathcal D\to\mathcal C.
$$

**证明边界.** 这是可展示范畴的伴随函子定理。本书作为一般范畴论输入使用；其证明依赖可访问范畴理论。第二卷使用的是其推论：在 presentable stable setting 中，几何构造出的 colimit-preserving $f_!$ 自动是左伴随。

**推论 F.3.** 若 $f_!:\mathcal C\to\mathcal D$ 是可展示稳定范畴之间的可访问余极限保持函子，则存在 $f^!:\mathcal D\to\mathcal C$ 和自然等价

$$
\operatorname{Map}_{\mathcal D}(f_!X,Y)
\simeq
\operatorname{Map}_{\mathcal C}(X,f^!Y).
$$

**证明.** 由定理 F.2 直接得到右伴随 $f^!$；显示的映射空间等价是伴随的定义。证毕。

## F.2 紧生成三角范畴版本

若读者只使用三角范畴，可采用以下版本。

**定义 F.4.** 带小直和的三角范畴 $\mathcal T$ 称为紧生成，如果存在一组紧对象 $\mathcal G$，使得对象 $X$ 为零当且仅当

$$
\operatorname{Hom}_{\mathcal T}(G,X[n])=0
$$

对所有 $G\in\mathcal G$ 和 $n\in\mathbb Z$ 成立。

**定理 F.5（Brown representability 形式）.** 设 $\mathcal T,\mathcal U$ 是紧生成三角范畴，且

$$
F:\mathcal T\to\mathcal U
$$

是精确函子并保持所有小直和。若 $\mathcal U$ 满足 Brown representability，则 $F$ 有右伴随。

**证明边界.** 对固定 $Y\in\mathcal U$，函子

$$
X\mapsto\operatorname{Hom}_{\mathcal U}(FX,Y)
$$

是 $\mathcal T^{op}\to\mathbf{Ab}$ 的 cohomological functor，并把小直和变为乘积。Brown representability 断言它由某个对象 $G(Y)\in\mathcal T$ 表示。表示对象随 $Y$ 函子化，给出右伴随。完整证明需要 Brown representability 的标准构造。

## F.3 张量作用和投影公式

设 $\mathcal C,\mathcal D$ 是稳定闭对称幺半范畴。设 $\mathcal D$ 通过一个对称幺半函子

$$
f^*:\mathcal C\to\mathcal D
$$

成为 $\mathcal C$-模范畴，即 $M\in\mathcal C$ 作用在 $N\in\mathcal D$ 上为

$$
M\star N=f^*M\otimes_{\mathcal D}N.
$$

设

$$
f_!:\mathcal D\to\mathcal C
$$

是余极限保持函子。

**定义 F.6.** $f_!$ 的投影公式自然变换是

$$
\pi_{M,N}:
M\otimes_{\mathcal C} f_!N
\to
f_!(f^*M\otimes_{\mathcal D}N),
$$

或反向约定

$$
f_!(f^*M\otimes_{\mathcal D}N)
\to
M\otimes_{\mathcal C} f_!N.
$$

本书采用第二卷正文中的反向约定：

$$
f_!(f^*M\otimes_{\mathcal D}N)
\simeq
M\otimes_{\mathcal C}f_!N.
$$

**命题 F.7（由生成元检验投影公式）.** 假设两边关于 $M$ 和 $N$ 都保持小余极限，并且 $\mathcal C,\mathcal D$ 分别由紧生成集 $\mathcal G_\mathcal C,\mathcal G_\mathcal D$ 在小余极限下生成。若 $\pi_{M,N}$ 对

$$
M\in\mathcal G_\mathcal C,\qquad N\in\mathcal G_\mathcal D
$$

为等价，则 $\pi_{M,N}$ 对所有 $M,N$ 为等价。

**证明.** 固定 $N$，令 $\mathcal A_N$ 为使 $\pi_{M,N}$ 为等价的 $M$ 的全子范畴。由于两边关于 $M$ 保持小余极限，$\mathcal A_N$ 对小余极限封闭；又包含生成集 $\mathcal G_\mathcal C$，故 $\mathcal A_N=\mathcal C$。再令 $\mathcal B$ 为使所有 $M$ 下 $\pi_{M,N}$ 为等价的 $N$ 的全子范畴。同理 $\mathcal B$ 对小余极限封闭并包含 $\mathcal G_\mathcal D$，故 $\mathcal B=\mathcal D$。证毕。

**注 F.8.** Scholze 情形中的困难正是构造合适的生成元并验证 $\pi$ 在生成元上为等价；一旦完成，命题 F.7 给出形式推广。

附录 H 给出命题 F.7 背后的紧生成和自然变换生成元检验形式。

## F.4 投影公式推出右伴随 Hom 公式

假设 $f_!\dashv f^!$，且 $\mathcal C,\mathcal D$ 是闭对称幺半范畴。记内部 Hom 为

$$
\mathcal Hom_\mathcal C(-,-),
\qquad
\mathcal Hom_\mathcal D(-,-).
$$

**定理 F.9.** 若投影公式成立，则对 $M\in\mathcal C$、$Y\in\mathcal C$ 有自然等价

$$
f^!\mathcal Hom_\mathcal C(M,Y)
\simeq
\mathcal Hom_\mathcal D(f^*M,f^!Y),
$$

在两边均具有上述内部 Hom 且投影公式成立的闭幺半环境中成立。

**证明.** 对任意 $N\in\mathcal D$，连续使用伴随、投影公式、闭幺半伴随和 $f_!\dashv f^!$：

$$
\begin{aligned}
\operatorname{Map}_{\mathcal D}
(N,f^!\mathcal Hom_\mathcal C(M,Y))
&\simeq
\operatorname{Map}_{\mathcal C}
(f_!N,\mathcal Hom_\mathcal C(M,Y))\\
&\simeq
\operatorname{Map}_{\mathcal C}
(M\otimes f_!N,Y)\\
&\simeq
\operatorname{Map}_{\mathcal C}
(f_!(f^*M\otimes N),Y)\\
&\simeq
\operatorname{Map}_{\mathcal D}
(f^*M\otimes N,f^!Y)\\
&\simeq
\operatorname{Map}_{\mathcal D}
(N,\mathcal Hom_\mathcal D(f^*M,f^!Y)).
\end{aligned}
$$

由 Yoneda 引理得到结论。证毕。

**推论 F.10（dualizing object 形式）.** 令

$$
\omega_f=f^!\mathbf 1_\mathcal C.
$$

若 $M=\mathbf 1_\mathcal C$，则定理 F.9 退化为恒等式。若 $f^*$ 保持双对偶对象，并且 $M$ dualizable，则

$$
f^!M\simeq f^*M\otimes\omega_f.
$$

**证明.** dualizable 对象满足

$$
\mathcal Hom_\mathcal D(f^*M,\omega_f)
\simeq
(f^*M)^\vee\otimes\omega_f.
$$

若 $f^*(M^\vee)\simeq(f^*M)^\vee$，则由定理 F.9 代入 $Y=\mathbf 1$ 并对偶化，得到常见的相对 dualizing object 公式。证毕。

## F.5 对第二卷第七章的回填

1. 第七章命题 7.2 可替换为推论 F.3 或定理 F.5，取决于使用 $\infty$-范畴还是三角范畴语言。
2. 投影公式的全体对象版本可由命题 F.7 从生成元检验推出。
3. $f^!$ 与内部 Hom 的相容性由定理 F.9 推出。
4. 这些形式结论不构造 $f_!$ 本身；$f_!$ 的存在、边界项和生成元验证仍是 Scholze 输入。

## 练习

**练习 F.1.** 在推论 F.3 中，把伴随单位和余单位写成映射空间自然等价对应的态射。

**练习 F.2.** 证明命题 F.7 中“包含生成集且对小余极限封闭”的全子范畴等于全范畴。

**练习 F.3.** 在定理 F.9 的证明中逐行标出使用的是哪一个伴随。

**练习 F.4.** 设 $f$ proper 且 $f_!=Rf_*$。说明定理 F.9 如何退化为 Grothendieck duality 中的内部 Hom 公式。
