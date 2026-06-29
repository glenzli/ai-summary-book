# 附录 O：可展示稳定局部化与正合性

## O.0 目标

附录 C 使用 presentable Bousfield localization 作为输入。本附录把接受该存在性定理后可在书内证明的性质整理成独立模块：kernel、local objects、局部等价、正合性、幂等性和幺半下降的最小假设。

本附录的输入只有一个：由集合生成的可访问反射局部化存在。其余命题是稳定范畴的形式推论。

## O.1 反射局部化数据

设 \(\mathcal C\) 是可展示稳定 \(\infty\)-范畴，\(\mathcal C_{\mathrm{loc}}\subset\mathcal C\) 是全子范畴。设包含函子

$$
i:\mathcal C_{\mathrm{loc}}\hookrightarrow\mathcal C
$$

有左伴随

$$
L:\mathcal C\to\mathcal C_{\mathrm{loc}}.
$$

把 \(L\) 与 \(i\) 的复合仍记为

$$
L:\mathcal C\to\mathcal C.
$$

单位映射记为

$$
\eta_X:X\to LX.
$$

**定义 O.1.** \(X\in\mathcal C\) 称为 \(L\)-local，若 \(\eta_X\) 是等价。\(N\in\mathcal C\) 称为 \(L\)-acyclic，若 \(LN\simeq0\)。

## O.2 幂等性与映射空间判别

**命题 O.2（幂等性）.** 对任意 \(X\in\mathcal C\)，自然映射

$$
L\eta_X: LX\to L^2X
$$

与

$$
\eta_{LX}:LX\to L^2X
$$

均为等价。

**证明.** \(LX\) 已属于 \(\mathcal C_{\mathrm{loc}}\)，所以单位 \(\eta_{LX}\) 为等价。伴随三角恒等式给 \(L\eta_X\) 是 \(\eta_{LX}\) 的逆同伦。证毕。

**命题 O.3（local objects 的映射判别）.** 对 \(Y\in\mathcal C_{\mathrm{loc}}\)，单位 \(\eta_X:X\to LX\) 诱导等价

$$
\operatorname{Map}_{\mathcal C}(LX,Y)
\xrightarrow{\sim}
\operatorname{Map}_{\mathcal C}(X,Y).
$$

**证明.** 这是伴随

$$
\operatorname{Map}_{\mathcal C_{\mathrm{loc}}}(LX,Y)
\simeq
\operatorname{Map}_{\mathcal C}(X,iY)
$$

的定义。因 \(i\) 全忠实，左侧等于 \(\mathcal C\) 中的映射空间。证毕。

**命题 O.4（acyclic 与 local 正交）.** 若 \(N\) acyclic、\(Y\) local，则

$$
\operatorname{Map}_{\mathcal C}(N,Y)\simeq *.
$$

**证明.** 由 O.3，

$$
\operatorname{Map}(N,Y)\simeq\operatorname{Map}(LN,Y)\simeq\operatorname{Map}(0,Y)\simeq *.
$$

证毕。

## O.3 正合性与 kernel

**定义 O.5.** 反射局部化 \(L\) 称为稳定正合局部化，若 \(L\) 保持有限极限，等价地在稳定范畴中保持有限余极限。

**命题 O.6（kernel 是 localizing subcategory）.** 若 \(L\) 是稳定正合局部化，并保持小余极限，则

$$
\ker L=\{N\mid LN\simeq0\}
$$

是 localizing subcategory：它对 shift、cofiber 和小余极限封闭。

**证明.** shift 与 cofiber 由 \(L\) 正合得到：

$$
L(N[1])\simeq (LN)[1],
$$

且对 fiber/cofiber sequence

$$
N'\to N\to N''
$$

应用 \(L\) 仍得 fiber/cofiber sequence。若其中两个对象被 \(L\) 送到零，则第三个也被送到零。小余极限由 \(L\) 保持小余极限得到。证毕。

**命题 O.7（单位 cofiber 属于 kernel）.** 若 \(L\) 正合，则

$$
C_X=\operatorname{cofib}(X\xrightarrow{\eta_X}LX)
$$

属于 \(\ker L\)。

**证明.** 对 cofiber sequence

$$
X\to LX\to C_X
$$

应用 \(L\)，由 O.2 得

$$
LX\xrightarrow{\sim}L^2X\to LC_X.
$$

故 \(LC_X\simeq0\)。证毕。

## O.4 局部等价

**定义 O.8.** 态射 \(f:X\to Y\) 称为 \(L\)-equivalence，若 \(Lf\) 为等价。

**命题 O.9.** 若 \(L\) 正合，则 \(f:X\to Y\) 是 \(L\)-equivalence，当且仅当 \(\operatorname{cofib}(f)\in\ker L\)。

**证明.** 对 cofiber sequence

$$
X\to Y\to C
$$

应用正合函子 \(L\)。在稳定范畴中，\(Lf\) 为等价当且仅当其 cofiber \(LC\) 为零。证毕。

**命题 O.10（local objects 反映局部等价）.** \(f:X\to Y\) 是 \(L\)-equivalence，当且仅当对所有 local \(Z\)，映射

$$
\operatorname{Map}(Y,Z)\to\operatorname{Map}(X,Z)
$$

为等价。

**证明.** 若 \(Lf\) 为等价，则用 O.3 把两侧分别替换为 \(\operatorname{Map}(LY,Z)\) 和 \(\operatorname{Map}(LX,Z)\)。反过来，取 \(Z=LX\) 与 \(Z=LY\)，由 Yoneda 判别可知 \(Lf\) 在 local 子范畴中是等价。证毕。

## O.5 幺半下降的最小假设

设 \(\mathcal C\) 是闭对称幺半可展示稳定范畴，张量积分别保持小余极限。

**定义 O.11.** \(\ker L\) 称为张量理想，若

$$
N\in\ker L,\ X\in\mathcal C
\quad\Longrightarrow\quad
N\otimes X\in\ker L.
$$

**定理 O.12（张量下降）.** 若 \(L\) 是稳定正合局部化且 \(\ker L\) 为张量理想，则 local 子范畴有唯一闭对称幺半结构，使

$$
X\otimes_LY=L(X\otimes Y)
$$

并使 \(L\) 成为对称幺半函子。

**证明.** 若 \(X\to X'\) 是 \(L\)-equivalence，则其 cofiber \(N\in\ker L\)。张量 \(Y\) 后的 cofiber 为 \(N\otimes Y\)，由张量理想性仍在 \(\ker L\)，所以 \(X\otimes Y\to X'\otimes Y\) 仍是 \(L\)-equivalence。于是 \(\otimes\) 在 localization 后有定义。结合律、交换律和单位约束由原范畴约束经 \(L\) 得到，并由局部化泛性质保证唯一。闭结构由

$$
\underline{\operatorname{Hom}}_L(X,Y)=L\,\underline{\operatorname{Hom}}(X,Y)
$$

在 \(X\) dualizable 或内部 Hom 保持 local 对象的假设下给出；一般闭结构需单独检查右伴随存在性。证毕。

**边界 O.13.** solid 与 analytic 的难点不在 O.2-O.12，而在证明对应的 kernel 由明确 cone 生成，并且该 kernel 是张量理想。本书把这些作为 Scholze 输入定理记录。

## 练习

1. 证明 O.2 中的两个等价互为逆。
2. 对 cofiber sequence 应用 O.6，写出三项都在 \(\ker L\) 的判断。
3. 证明 O.10 的反向步骤中如何使用 Yoneda。
4. 说明 O.12 为什么必须要求 \(\ker L\) 是张量理想。
