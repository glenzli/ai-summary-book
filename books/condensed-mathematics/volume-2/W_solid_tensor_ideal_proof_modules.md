# 附录 W：Solid 核张量理想性的证明模块

## W.0 目标

附录 Q 把 solid kernel 的张量理想性作为输入定理 Q.5。本附录把该输入拆成若干可检查模块，说明真正困难集中在哪里。

目标结论是：

> 若 \(N\) 属于由 \(K_S=\operatorname{cofib}(\mathbb Z[\underline S]\to\mathbb Z^\square[S])\) 生成的 localizing subcategory \(\mathcal N_\square\)，则对任意 \(X\in D(\mathbf{CondAb})\)，有 \(N\otimes^LX\in\mathcal N_\square\)。

## W.1 张量理想性的生成元判别

**命题 W.1（生成元判别）。** 设 \(\mathcal C\) 是 presentable stable symmetric monoidal category，张量积分别保持小余极限。令 \(\mathcal N\) 为由一组对象 \(\mathcal K\) 生成的 localizing subcategory。若对所有 \(K\in\mathcal K\) 和所有 \(X\in\mathcal C\)，有

$$
K\otimes X\in\mathcal N,
$$

则 \(\mathcal N\) 是张量理想。

**证明.** 固定 \(X\)。对象类

$$
\mathcal A_X=\{M\in\mathcal C\mid M\otimes X\in\mathcal N\}
$$

对 shift、cofiber 和小余极限封闭，因为 \(-\otimes X\) exact 且保持小余极限，\(\mathcal N\) localizing。假设给出 \(\mathcal K\subset\mathcal A_X\)，故 \(\mathcal N\subset\mathcal A_X\)。这对所有 \(X\) 成立，因此 \(\mathcal N\) 是张量理想。证毕。

应用到 solid 时，只需证明：

$$
K_S\otimes^LX\in\mathcal N_\square
$$

对所有 profinite \(S\) 和所有 \(X\) 成立。

## W.2 再降到生成对象

若 \(D(\mathbf{CondAb})\) 由一族对象 \(\mathcal G\) 在 colimit 和 cofiber 下生成，并且 \(K_S\otimes G\in\mathcal N_\square\) 对所有 \(G\in\mathcal G\) 成立，则可推出对所有 \(X\) 成立。

**命题 W.2（第二生成元判别）。** 假设张量积 \(K_S\otimes -\) 保持小余极限和 cofiber。若 \(\mathcal G\) 生成 \(\mathcal C\)，且 \(K_S\otimes G\in\mathcal N\) 对所有 \(G\in\mathcal G\) 成立，则 \(K_S\otimes X\in\mathcal N\) 对所有 \(X\in\mathcal C\) 成立。

**证明.** 固定 \(S\)。对象类

$$
\mathcal B_S=\{X\mid K_S\otimes X\in\mathcal N\}
$$

是 localizing subcategory，且按假设包含 \(\mathcal G\)。若 \(\mathcal G\) 生成 \(\mathcal C\)，则 \(\mathcal B_S=\mathcal C\)。证毕。

因此 solid 张量理想性的核心计算可压缩为：

> 对 profinite \(S\) 和一组生成对象 \(G\)，证明 \(K_S\otimes G\) 被 solidification 杀掉。

## W.3 凝聚阿贝尔群的生成对象

在凝聚阿贝尔群中，基本生成对象来自紧 Hausdorff 或 profinite 测试对象：

$$
\mathbb Z[\underline T].
$$

**输入定理 W.3（凝聚生成族）。** \(D(\mathbf{CondAb})\) 由 \(\mathbb Z[\underline T]\) 生成，其中 \(T\) 遍历一个小骨架中的 profinite 空间。

**书内使用.** 第一卷已证明可表对象和自由凝聚阿贝尔群的基本泛性质；生成族定理的可展示范畴细节作为形式化基础输入使用。

由 W.2，核心计算变为：

$$
K_S\otimes^L\mathbb Z[\underline T]\in\mathcal N_\square.
$$

## W.4 与乘积公式的关系

由定义，

$$
K_S=\operatorname{cofib}(\mathbb Z[\underline S]\to\mathbb Z^\square[S]).
$$

张量 \(\mathbb Z[\underline T]\) 后得到 cofiber

$$
\mathbb Z[\underline S]\otimes^L\mathbb Z[\underline T]
\to
\mathbb Z^\square[S]\otimes^L\mathbb Z[\underline T]
\to
K_S\otimes^L\mathbb Z[\underline T].
$$

第一项有自然同构

$$
\mathbb Z[\underline S]\otimes^L\mathbb Z[\underline T]
\simeq
\mathbb Z[\underline{S\times T}]
$$

因为自由对象张量满足双线性泛性质。

真正困难是比较

$$
\mathbb Z^\square[S]\otimes^L\mathbb Z[\underline T]
$$

与 solid 测度对象。solid localization 后应有

$$
L^\square(\mathbb Z^\square[S]\otimes^L\mathbb Z[\underline T])
\simeq
\mathbb Z^\square[S\times T].
$$

这正是 profinite 测度张量公式的一部分。

## W.5 核心计算输入

**输入定理 W.4（profinite 测度张量计算）。** 对 profinite \(S,T\)，自然态射

$$
L^\square(\mathbb Z^\square[S]\otimes^L\mathbb Z[\underline T])
\to
\mathbb Z^\square[S\times T]
$$

为等价，并且它与

$$
\mathbb Z[\underline{S\times T}]\to\mathbb Z^\square[S\times T]
$$

相容。

**命题 W.5.** 在 W.4 下，

$$
K_S\otimes^L\mathbb Z[\underline T]\in\mathcal N_\square.
$$

**证明.** 对 cofiber sequence 局部化。第一项局部化为

$$
L^\square\mathbb Z[\underline{S\times T}]
\simeq
\mathbb Z^\square[S\times T].
$$

第二项由 W.4 局部化后也等价于同一对象，且映射与 Dirac-to-measure map 相容。因此两项之间的映射在 solid localization 后为等价，其 cofiber 即 \(K_S\otimes^L\mathbb Z[\underline T]\) 在局部化后为零。故该 cofiber 属于核 \(\mathcal N_\square\)。证毕。

## W.6 Solid 核张量理想性

**定理 W.6（solid 核张量理想性，归约形式）。** 假设：

1. \(D(\mathbf{CondAb})\) 由 \(\mathbb Z[\underline T]\) 生成；
2. W.4 的 profinite 测度张量计算成立；
3. \(\ker L^\square=\mathcal N_\square\)。

则 \(\mathcal N_\square\) 是张量理想。

**证明.** 由 W.5，\(K_S\otimes\mathbb Z[\underline T]\in\mathcal N_\square\)。由 W.2，\(K_S\otimes X\in\mathcal N_\square\) 对所有 \(X\) 成立。由 W.1，\(\mathcal N_\square\) 是张量理想。证毕。

## W.7 Nöbeling 定理的位置

W.4 的证明依赖 profinite 空间上整数值连续函数群的结构。Nöbeling 定理给出：

$$
C(S,\mathbb Z)
$$

是自由阿贝尔群。该自由性用于把 profinite 测度对象分解成可控的乘积型对象，从而证明测度张量公式。

**边界 W.7.** Nöbeling 自由性本身不足以自动推出 W.4。还需要：

1. 自由基选择与 profinite functoriality 的相容控制；
2. 派生张量与无限乘积的 solid 修正；
3. Dirac-to-measure map 在乘积 \(S\times T\) 下的自然性；
4. 生成核 \(\mathcal N_\square\) 与这些比较图的相容性。

这些是 Scholze solid tensor 证明的核心计算。

## W.8 本附录闭包

**结论 W.8.** solid kernel 张量理想性的证明已经归约为 profinite 测度张量计算 W.4。W.1、W.2、W.5、W.6 是书内证明；W.3、W.4 和 \(\ker L^\square=\mathcal N_\square\) 是仍需外部证明或后续展开的输入。

## 练习

1. 证明 W.1 中 \(\mathcal A_X\) 对 cofiber 封闭。
2. 证明 \(\mathbb Z[\underline S]\otimes\mathbb Z[\underline T]\simeq\mathbb Z[\underline{S\times T}]\) 的泛性质。
3. 说明为什么普通张量积不保持无限乘积会阻碍 W.4。
4. 用 W.6 推出附录 Q 的 Q.5。

