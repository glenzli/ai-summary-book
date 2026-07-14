# 第十章：Crystalline、de Rham-Witt 与 $q$-de Rham specialization

“Prismatic cohomology 统一多种理论”只有在 base prism 改变时仍能逐项识别输出，才是可验证的命题。Crystalline prism、perfectoid/$A_{\inf}$ prism 与 $q$-crystalline prism 分别导向 crystalline、de Rham--Witt 和 $q$-de Rham 结构，但所需完备化、Frobenius 与坐标依赖并不相同。本章以第二章的基本 prism 和第三章的比较接口为共同起点，结合第五章的 BMS 对象与附录 B 的局部模型，分别推导三种 specialization 的输入、输出和边界，并用低阶 $q$-二项式计算展示 $q\to1$ 时怎样回到通常微分。

## 10.1 Crystalline prism

**定义 10.1.** 若 $(A,I)=(A,(p))$，称其为 crystalline prism。对 $X$ smooth over $A/p$，其 prismatic cohomology 记为
$$
R\Gamma_\Delta(X/A).
$$

**外部输入定理 10.2（crystalline specialization）.** 在定义 10.1 的假设下，$R\Gamma_\Delta(X/A)$ 与 crystalline cohomology of $X/A$ 存在自然 $\varphi$-equivariant comparison。若 $A$ 带 Frobenius lift，该 comparison 可解释为 crystalline cohomology 的 Frobenius descent。

**警告 10.3.** 定理 10.2 中的 crystalline cohomology 需要指定 divided power structure。若 $A$、$p$-adic divided powers 或 crystalline site 的 convention 改变，公式形式也会改变。

**命题 10.4.** 在 crystalline prism 情形，boundedness 自动成立。

**证明.** 因为 $A/I=A/p$，其中 $p$ 作用为零。故
$$
(A/p)[p^\infty]=A/p=(A/p)[p].
$$
因此 $p^\infty$-torsion 由 $p$ 一步杀死。证毕。

## 10.2 de Rham-Witt interface

**外部输入定义 10.5.** 令 $R$ 为 characteristic $p$ 的合适光滑环。de Rham-Witt complex $W\Omega^\bullet_R$ 是带 Frobenius、Verschiebung 和 differential 的 Witt vector differential complex，用于计算 crystalline cohomology。

**外部输入定理 10.6.** 在适当 smoothness 和 finiteness 假设下，crystalline cohomology 可由 de Rham-Witt complex 计算：
$$
R\Gamma_{\mathrm{crys}}(R/W(k))
\simeq
R\Gamma(W\Omega^\bullet_R).
$$

**说明 10.7.** Prismatic cohomology 在 crystalline prism 上回收 crystalline cohomology，因此通过定理 10.2 和 10.6 与 de Rham-Witt complex 相接。该相接是定理组合，不是 de Rham-Witt complex 的重新定义。

## 10.3 $A_{\inf}$ 与 de Rham-Witt 的积分联系

**外部输入定理 10.8.** BMS 的 $A_{\inf}$-cohomology 在合适 affine 情形与 relative de Rham-Witt complexes 有关系，并可在某些坐标情形中表现为 $q$-deformation of de Rham cohomology。

**警告 10.9.** 这个定理不能简写成“$A_{\inf}$-cohomology 等于 de Rham-Witt complex”。二者处于不同底环、不同完备性和不同 Frobenius convention 中；只有在特定 hypotheses 和 comparison maps 下相接。

## 10.4 $q$-de Rham specialization

**定义 10.10.** 设
$$
A=\mathbf Z_p[[q-1]],\qquad I=([p]_q),\qquad [p]_q=\frac{q^p-1}{q-1}.
$$
称 $(A,I)$ 为 $q$-crystalline prism。

**外部输入定理 10.11.** 在合适坐标和 smoothness 假设下，$q$-crystalline prism 上的 prismatic cohomology 与 $q$-de Rham complex 相比较。该 comparison 给出 $q$-de Rham cohomology 的坐标无关解释。

**例 10.12（形式一变量模型）.** 对 $R=A/I\langle T\rangle$，$q$-de Rham complex 的一变量模型使用 $q$-difference operator
$$
\nabla_q(f)=\frac{f(qT)-f(T)}{qT-T}.
$$
当 $q\to1$ 时，该算子形式上趋向普通导数 $\frac{df}{dT}$。

**警告 10.13.** 例 10.12 是坐标模型，不是全局定义。坐标无关性正是 prismatic comparison 的内容之一。

## 10.5 三种 specialization 的边界

**命题 10.14（输入 prism 判别）.** 若一个 comparison statement 没有说明基 prism 是 crystalline、perfect/BMS、Breuil-Kisin 还是 $q$-crystalline，则该 statement 不足以判定目标 cohomology theory。

**证明.** 不同 base prism 的 quotient $A/I$、Frobenius、Cartier divisor 和 topology 不同。Crystalline prism 的 quotient 为 characteristic $p$ 对象，perfect prism 的 quotient 是 perfectoid ring，Breuil-Kisin prism 的 quotient 是 $\mathcal O_K$，$q$-crystalline prism 的 quotient 带 $q$-deformation 参数。目标 comparison 依赖这些输入。因此未说明基 prism 的陈述不具备确定数学含义。证毕。

## 10.6 $q$-de Rham 的局部二项式计算

**命题 10.15.** 对 $f(T)=T^n$，$q$-difference operator
$$
\nabla_q(f)=\frac{f(qT)-f(T)}{qT-T}
$$
满足
$$
\nabla_q(T^n)=[n]_qT^{n-1},\qquad [n]_q=1+q+\cdots+q^{n-1}.
$$

**证明.** 直接计算：
$$
\nabla_q(T^n)=\frac{q^nT^n-T^n}{(q-1)T}
=\frac{q^n-1}{q-1}T^{n-1}
=[n]_qT^{n-1}.
$$
证毕。

**推论 10.16.** 当 $q\to1$ 时，$\nabla_q(T^n)$ 形式上趋向 ordinary derivative $nT^{n-1}$。

**证明.** 多项式 $[n]_q$ 在 $q=1$ 的值为 $n$。代入命题 10.15 即得。证毕。

## 10.7 Base prism 决定的三个出口

Crystalline、de Rham-Witt 和 $q$-de Rham 都是 prismatic cohomology 的 specialization 或 comparison 后果，但它们对应不同 prism 和不同外部输入定理。正式教材写法必须避免把“统一”误写成“相同”。

## 练习

**练习 10.1.** 对 crystalline prism $(A,(p))$，说明 $A/p$ 的 $p^\infty$-torsion 为什么由 $p$ 一步杀死。

**练习 10.2.** 写出 de Rham-Witt complex 中 Frobenius、Verschiebung 和 differential 之间至少一个标准相容关系，并查明其来源。

**练习 10.3.** 对 $f(T)=T^n$ 计算例 10.12 中的 $\nabla_q(f)$。
