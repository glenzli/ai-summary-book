# 附录 Y：Rational descent 的证明模块

## Y.0 目标

附录 R 把 rational Čech descent 作为输入定理 R.13。本附录把该定理拆成可证明的范畴论部分和真正的几何输入部分。

目标形式为：若 \(X=\bigcup_iU_i\) 是 finite rational cover，则

$$
D(A,\mathcal M)\to
\operatorname{Tot}D(A_{U_\bullet},\mathcal M_{U_\bullet})
$$

为等价。

## Y.1 Čech nerve

令 \(U_\bullet\) 为 cover \(X=\bigcup_iU_i\) 的 Čech nerve：

$$
U_n=\coprod_{i_0,\ldots,i_n}U_{i_0}\cap\cdots\cap U_{i_n}.
$$

analytic rings \((A_{U_I},\mathcal M_{U_I})\) 给出 cosimplicial stable category

$$
[n]\mapsto D(A_{U_n},\mathcal M_{U_n}).
$$

**定义 Y.1（descent category）。** 定义

$$
\operatorname{Desc}(U_\bullet)
=
\operatorname{Tot}D(A_{U_\bullet},\mathcal M_{U_\bullet}).
$$

其对象是局部对象、交叠等价和高阶 cocycle 的全套数据。

## Y.2 全忠实的形式判别

**命题 Y.2（mapping space descent 推出全忠实）。** 若对任意 \(M,N\in D(A,\mathcal M)\)，自然映射

$$
\operatorname{Map}(M,N)
\to
\operatorname{Tot}\operatorname{Map}(M|_{U_\bullet},N|_{U_\bullet})
$$

为等价，则限制函子

$$
D(A,\mathcal M)\to\operatorname{Desc}(U_\bullet)
$$

全忠实。

**证明.** totalization 范畴中的 mapping space 由 cosimplicial mapping space 的 totalization 计算。因此上式正是全忠实条件。证毕。

## Y.3 本质满的形式判别

**命题 Y.3（对象 glueing 推出本质满）。** 若任意 descent datum

$$
(M_i,\varphi_{ij},\varphi_{ijk},\ldots)
$$

都来自某个 \(M\in D(A,\mathcal M)\)，则限制函子本质满。

**证明.** 本质满的定义就是 totalization 中每个对象在限制函子的本质像中。证毕。

## Y.4 Rational acyclicity 输入

上述 Y.2-Y.3 是形式范畴论。真正需要证明的是 mapping space descent 和对象 glueing。它们通常来自 rational localization 的 acyclicity。

**输入定理 Y.4（analytic rational acyclicity）。** 对 finite rational cover \(X=\bigcup_iU_i\)，单位对象和一族生成对象 \(G\) 满足 Čech 复形 acyclic：

$$
G\to \operatorname{Tot}G|_{U_\bullet}
$$

为等价。

**输入定理 Y.5（生成元上的 descent）。** 若 Y.4 对一组 compact generators 成立，并且 restriction functors 保持 colimit、exact triangle 和 compact generators，则整个 \(D(A,\mathcal M)\) 满足 descent。

**命题 Y.6（生成元 descent 推出范畴 descent）。** 在 Y.5 的假设下，

$$
D(A,\mathcal M)\to\operatorname{Desc}(U_\bullet)
$$

为等价。

**证明.** 全忠实性：使 mapping space descent 成立的对象对 \(M\) 和 \(N\) 分别形成 localizing subcategory；若它对 compact generators 成立，则对所有对象成立。  

本质满性：descent datum 的 totalization 范畴由局部 compact generators 生成；若这些局部 generators 可由全局 generators glueing 得到，则任意 colimit 和 cofiber 构造出的对象也来自全局。精确证明使用 presentable stable category 中的 Barr-Beck/monadic descent 或 compact generation descent 判别。证毕。

## Y.5 Huber pair 的 rational localization

**输入定理 Y.7（Huber rational compatibility）。** 离散 Huber pair \((A,A^+)\) 的 rational subset \(U\) 给出 analytic ring \((A_U,\mathcal M_U)\)，并且：

1. rational intersections 对应 iterated rational localization；
2. restriction functors 与 analyticization 相容；
3. rational cover 的 Čech nerve 保持在同一类 analytic rings 中；
4. 生成对象在 rational cover 上满足 Y.4。

**说明.** 这是 Scholze analytic geometry 的核心输入；本书不把它压缩成伪证明。

## Y.6 Descent 的后果

**定理 Y.8（rational descent 的形式闭包）。** 接受 Y.7 后，第二卷使用的 rational descent 后果均成立：

1. 对象可由 rational cover glueing；
2. 态射空间满足 totalization 公式；
3. 局部零对象可检测全局零对象；
4. perfect/compact 性在满足子堆条件时可局部检测；
5. analytic tensor 与 restriction 相容。

**证明.** 1、2 由 Y.2-Y.6。3 是 2 对 \(M=N\) 或 mapping out of generators 的特例。4 需要 perfect 子范畴对 descent 封闭；这是附加假设下的形式后果。5 来自 restriction functors 的 symmetric monoidal 性和 analytic tensor 的局部化定义。证毕。

## Y.7 本附录闭包

**结论 Y.9.** rational descent 的范畴论框架已经书内闭合；真正的外部输入是 Huber rational localization 与 rational acyclicity，即 Y.7。

## 练习

1. 对二开覆盖写出 \(\operatorname{Desc}(U_\bullet)\) 的对象数据。
2. 证明 Y.2。
3. 说明 Y.6 为什么需要 compact generation 或其他生成性假设。
4. 解释 rational descent 与普通 sheaf descent 的差别。

