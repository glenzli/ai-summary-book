# 第一卷练习答案与教师手册补充

作者：Dr. Stochastic Parrot

## 使用说明

全书统一答案见 [../SOLUTIONS.md](../SOLUTIONS.md)。本文件补第一卷中最容易造成证明跳步的题目，作为教师手册入口。

## 1. Sheaf 条件与 separated 性

**命题。** 若预层 \(F\) 满足 sheaf 条件，则 \(F\) separated。

**详解。** 设 \(\{U_i\to U\}\) 为覆盖，且 \(s,t\in F(U)\) 满足

$$
s|_{U_i}=t|_{U_i}
$$

对所有 \(i\) 成立。则 \((s|_{U_i})_i\) 与 \((t|_{U_i})_i\) 是同一个匹配族。sheaf 条件给该匹配族的粘合唯一。由于 \(s\) 与 \(t\) 都是它的粘合，故 \(s=t\)。

## 2. 可表预层是 sheaf

**命题。** 在 compact Hausdorff 站点上，可表预层

$$
h_T(S)=\operatorname{Hom}_{\mathbf{CHaus}}(S,T)
$$

是 sheaf。

**详解。** 对有限联合满射覆盖 \(\{S_i\to S\}\)，令

$$
q:\coprod_iS_i\to S.
$$

因 \(\coprod_iS_i\) 紧且 \(S\) Hausdorff，\(q\) 是闭满射，故为 quotient map。给定相容族 \(f_i:S_i\to T\)，它们给出连续映射

$$
f':\coprod_iS_i\to T.
$$

相容性等价于 \(f'\) 在 \(q\) 的纤维上常值。于是存在唯一集合映射 \(f:S\to T\) 使 \(fq=f'\)。由 quotient 性，\(f\) 连续。唯一性由 \(q\) 满射得出。

## 3. ED 自由对象投射

**命题。** 若 \(E\) 极不连通，则 \(\mathbb Z[\underline E]\) 在 \(\mathbf{CondAb}\) 中投射。

**详解。** 对 sheaf 满射 \(A\to B\)，需证明

$$
\operatorname{Hom}(\mathbb Z[\underline E],A)
\to
\operatorname{Hom}(\mathbb Z[\underline E],B)
$$

满射。由自由对象泛性质，这等价于

$$
A(E)\to B(E)
$$

满射。取 \(b\in B(E)\)。sheaf 满射定义给覆盖 \(p:E'\to E\)，使 \(b|_{E'}\) 可提升到 \(A(E')\)。Gleason lifting 给截面 \(s:E\to E'\)。沿 \(s\) 拉回该提升，即得 \(A(E)\) 中提升。

## 4. Ext 定义独立性

**命题。** Ext 群不依赖投射分解选择。

**详解。** 设 \(P_\bullet\to M\) 与 \(Q_\bullet\to M\) 为两个投射分解。比较定理给链映射

$$
P_\bullet\to Q_\bullet,\qquad Q_\bullet\to P_\bullet
$$

提升 \(\operatorname{id}_M\)。复合链映射与恒等链映射链同伦。对目标 \(A\) 取 Hom 复形，链同伦等价诱导 cohomology 同构。因此

$$
H^n\operatorname{Hom}(P_\bullet,A)
\cong
H^n\operatorname{Hom}(Q_\bullet,A).
$$

## 5. Tor 长正合列

**命题。** 短正合列

$$
0\to N'\to N\to N''\to0
$$

诱导 Tor 长正合列。

**详解。** 取 \(M\) 的 K-flat 或投射分解 \(P_\bullet\to M\)。逐项张量给短正合列

$$
0\to P_\bullet\otimes N'
\to P_\bullet\otimes N
\to P_\bullet\otimes N''
\to0,
$$

因为 \(P_i\) 平坦。短正合复形给同调长正合列，定义同调为 Tor 即得结论。
