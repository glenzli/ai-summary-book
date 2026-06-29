# 附录 AJ：Grothendieck duality 的构造义务

## AJ.0 目标

附录 AD 把 Grothendieck-Serre duality 作为输入。本附录拆解一般 \(f^!\) 和 dualizing complex 的构造义务：

1. smooth morphism 的 \(f^!\)；
2. closed immersion 的 \(f^!\)；
3. proper morphism 的因子分解；
4. trace map；
5. base change 与 projection formula。

本附录给出形式构造链条，不证明完整六函子定理。

## AJ.1 Smooth morphism

**输入/定义 AJ.1.** 若 \(f:X\to Y\) 是相对维数 \(d\) 的 smooth morphism，则

$$
f^!(-)=Lf^\ast(-)\otimes\omega_{X/Y}[d].
$$

这里

$$
\omega_{X/Y}=\det\Omega^1_{X/Y}.
$$

**命题 AJ.2.** 若 \(Y=*\)，\(X\) 是光滑 \(d\)-维空间，则

$$
f^!\mathbb C\simeq\omega_X[d].
$$

**证明.** 代入 AJ.1，且 \(\omega_{X/*}=\omega_X\)。证毕。

## AJ.2 Closed immersion

设 \(i:Z\hookrightarrow X\) 是 regular closed immersion，codimension 为 \(c\)，法丛为 \(N_{Z/X}\)。

**输入定理 AJ.3（regular immersion duality）.** 有

$$
i^!(-)\simeq
Li^\ast(-)\otimes\det(N_{Z/X})[-c].
$$

等价地，对 \(\mathcal O_Z\)，

$$
R\mathcal Hom_X(i_\ast\mathcal O_Z,\mathcal O_X)
\simeq
i_\ast\det(N_{Z/X})[-c].
$$

**证明骨架.** regular immersion 局部由 regular sequence 定义；Koszul complex 解析 \(i_\ast\mathcal O_Z\)。对 Koszul complex 取 dual 得 determinant 和 shift。完整证明需检查局部 regular sequences 的 glueing。

## AJ.3 Proper morphism 的因子分解

**输入定理 AJ.4（嵌入分解）.** 对本书考虑的可嵌入复解析空间或有限型代数几何对象，proper morphism \(f:X\to Y\) 可局部分解为

$$
X\xrightarrow{i}P\xrightarrow{p}Y
$$

其中 \(i\) 是 closed immersion，\(p\) 是 smooth/projective bundle 型 morphism。

**构造 AJ.5.** 定义

$$
f^!=i^!p^!.
$$

**输入定理 AJ.6（分解无关性）.** 不同嵌入分解给出的 \(f^!\) 自然同构，并满足 composition law：

$$
(g\circ f)^!\simeq f^!g^!.
$$

## AJ.4 Trace

**输入定理 AJ.7（trace/counit）.** 对 proper \(f:X\to Y\)，存在 counit

$$
Rf_\ast f^!(-)\to \operatorname{id}_{D^b_{\operatorname{coh}}(Y)}.
$$

对 \(Y=*\)，这给出

$$
R\Gamma(X,\omega_X^\bullet)\to\mathbb C.
$$

**命题 AJ.8.** 若 \(X\) 光滑 proper，则 AJ.7 的 trace 与积分

$$
\int_X:\ H^n(X,\omega_X)\to\mathbb C
$$

一致。

**证明.** 该相容是 trace 构造的归一化条件。对 \(X=\mathbb P^n\) 可由 Čech residue 验证；一般情形由局部坐标、partition of unity 或代数 residue 的 functoriality 粘合。证毕。

## AJ.5 Projection formula 与 duality

**输入定理 AJ.9（Grothendieck duality）.** 对 proper \(f\) 和 \(F,G\in D^b_{\operatorname{coh}}\)，有自然同构

$$
R\mathcal Hom_Y(Rf_\ast F,G)
\simeq
Rf_\ast R\mathcal Hom_X(F,f^!G).
$$

**推论 AJ.10.** 取 \(Y=*\)、\(G=\mathbb C\)，得附录 AD 的 global duality。

**证明.** 代入 \(Y=*\)，左侧为 \(R\operatorname{Hom}_{\mathbb C}(R\Gamma(X,F),\mathbb C)\)，右侧为 \(R\Gamma(X,R\mathcal Hom(F,f^!\mathbb C))\)。证毕。

## 练习

1. 对 smooth curve \(X\to *\)，写出 \(f^!\mathbb C\)。
2. 用 Koszul complex 计算 divisor \(D\hookrightarrow X\) 的 \(i^!\mathcal O_X\)。
3. 解释 AJ.6 为什么是构造 \(f^!\) 的核心难点。
4. 从 AJ.9 推出 AD.3。
