# 附录 E：局部化技术引理

## E.0 目标

附录 C 给出了 Bousfield localization 的形式骨架。本附录继续把第二卷反复使用的局部化论证写成可直接引用的引理。重点不是重证 Scholze 的 solid/analytic 深层定理，而是在接受“局部化存在”和“核为张量理想”后，把后续推论做成完整证明。

本附录固定一个可展示稳定范畴 $\mathcal C$。映射空间记为 $\operatorname{Map}_{\mathcal C}(-,-)$。若读者采用三角范畴语言，可把 fiber/cofiber sequence 分别读作 distinguished triangle 的旋转。

## E.1 由态射族定义的局部对象

设 $\Sigma$ 是 $\mathcal C$ 中的一组态射。

**定义 E.1.** 对象 $X$ 称为 $\Sigma$-局部，如果对每个 $f:A\to B$ 属于 $\Sigma$，诱导映射

$$
\operatorname{Map}_{\mathcal C}(B,X)
\to
\operatorname{Map}_{\mathcal C}(A,X)
$$

是等价。态射 $g:M\to N$ 称为 $\Sigma$-等价，如果对每个 $\Sigma$-局部对象 $X$，映射

$$
\operatorname{Map}_{\mathcal C}(N,X)
\to
\operatorname{Map}_{\mathcal C}(M,X)
$$

是等价。

若 $\Sigma$ 中每个 $f$ 的 cofiber 组成对象族 $\mathcal K$，则 $\Sigma$-局部等价于 $\mathcal K$-局部：

$$
\operatorname{Map}(\operatorname{cofib}(f),X)\simeq *
$$

对所有 $f\in\Sigma$ 成立。

**命题 E.2.** $\Sigma$-等价满足二出三性质，并对 retract 封闭。

**证明.** 对任意局部对象 $X$，函子 $\operatorname{Map}(-,X)$ 把复合态射反向变为复合映射。空间中的等价满足二出三，因此 $\Sigma$-等价也满足二出三。retract 情形同理：等价的 retract 仍为等价。证毕。

**命题 E.3.** 若 $L:\mathcal C\to\mathcal C_{\Sigma\text{-loc}}$ 是反射局部化，则态射 $g$ 是 $\Sigma$-等价，当且仅当 $Lg$ 是等价。

**证明.** 若 $Lg$ 是等价，则对局部对象 $X$，

$$
\operatorname{Map}(N,X)
\simeq
\operatorname{Map}(LN,X)
\to
\operatorname{Map}(LM,X)
\simeq
\operatorname{Map}(M,X)
$$

是等价。反过来，若 $g$ 对所有局部对象诱导等价，则在局部子范畴中 $Lg$ 对所有对象诱导映射空间等价。由 Yoneda 引理，$Lg$ 是等价。证毕。

## E.2 核、像和局部化三角

记

$$
\mathcal N=\{M\in\mathcal C\mid LM\simeq0\}.
$$

**命题 E.4.** $\mathcal N$ 是 localizing subcategory：它对 shift、cofiber、任意小余极限封闭。

**证明.** $L$ 是稳定左伴随，因此保持 shift、cofiber 和小余极限。若 $M_i\in\mathcal N$，则

$$
L(\operatorname{colim}_iM_i)
\simeq
\operatorname{colim}_iLM_i
\simeq0.
$$

对 cofiber sequence $M'\to M\to M''$ 应用 $L$ 后仍为 cofiber sequence；若其中两个为零，则第三个为零。shift 同理。证毕。

**命题 E.5.** 对任意 $M\in\mathcal C$，单位映射

$$
\eta_M:M\to LM
$$

的 cofiber 属于 $\mathcal N$。并且 $M\to N$ 是局部等价，当且仅当 $\operatorname{cofib}(M\to N)\in\mathcal N$。

**证明.** 对 $\eta_M$ 应用 $L$ 得到

$$
LM\to LLM,
$$

这是等价，因为 $L$ 在局部对象上同构于恒等函子。因此 $L\operatorname{cofib}(\eta_M)=0$。

对一般 $f:M\to N$，应用 $L$ 得 cofiber sequence

$$
LM\to LN\to L\operatorname{cofib}(f).
$$

由命题 E.3，$f$ 是局部等价当且仅当 $Lf$ 是等价；这等价于 $L\operatorname{cofib}(f)=0$。证毕。

## E.3 张量理想与幺半下降

设 $\mathcal C$ 是可展示稳定对称幺半范畴，且 $\otimes$ 分别保持小余极限。

**定义 E.6.** 局部化 $L$ 称为幺半局部化，如果对所有 $M,N$，自然态射

$$
L(M\otimes N)\to L(LM\otimes LN)
$$

是等价，并且局部对象的张量积经 $L$ 后给出局部范畴的张量积。

**定理 E.7.** 若核 $\mathcal N=\ker L$ 是张量理想，即

$$
N\in\mathcal N,\ X\in\mathcal C
\quad\Rightarrow\quad
N\otimes X\in\mathcal N,
$$

则 $L$ 是幺半局部化。局部范畴中的张量积为

$$
X\otimes_LY=L(iX\otimes iY).
$$

**证明.** 若 $f:M\to M'$ 是局部等价，则由命题 E.5，$\operatorname{cofib}(f)\in\mathcal N$。对任意 $X$，因为张量保持 cofiber，

$$
\operatorname{cofib}(f\otimes X)
\simeq
\operatorname{cofib}(f)\otimes X
\in\mathcal N.
$$

故 $f\otimes X$ 仍是局部等价。于是张量积对两个变量都能沿 localization 下降，定义出 $\otimes_L$。结合律、交换律和单位约束来自原范畴，经 $L$ 后仍满足 coherence 图，因为这些图在 $\mathcal C$ 中交换，$L$ 是函子。证毕。

**推论 E.8.** 若 $A$ 是 $\mathcal C$ 中的交换代数对象，则 $LA$ 是局部范畴中的交换代数对象。若 $M$ 是 $A$-模，则 $LM$ 是 $LA$-模。

**证明.** 交换代数对象由乘法、单位和有限 coherence 图给出。幺半函子 $L$ 把这些结构图送到局部范畴。模对象同理。证毕。

## E.4 相对张量积的局部化

设 $A$ 是交换代数对象，$\operatorname{Mod}_A(\mathcal C)$ 为 $A$-模范畴。

**命题 E.9.** 在定理 E.7 的假设下，局部范畴中 $LA$-模的相对张量积满足

$$
LM\otimes_{LA}^{L}LN
\simeq
L(M\otimes_A^LN)
$$

对 $A$-模 $M,N$ 自然成立。

**证明.** 相对张量积是双边 bar construction 的几何实现：

$$
M\otimes_A^LN
\simeq
|\,M\otimes A^{\otimes \bullet}\otimes N\,|.
$$

由于 $L$ 保持小余极限，并且按定理 E.7 与张量积相容，

$$
L|M\otimes A^{\otimes\bullet}\otimes N|
\simeq
|LM\otimes (LA)^{\otimes_L\bullet}\otimes_L LN|.
$$

右侧正是局部范畴中 $LM\otimes_{LA}^LLN$ 的 bar construction。证毕。

这条公式是第二卷中所有

$$
L^\square(M\otimes_A^LN)
$$

型定义的范畴论理由。

## E.5 Solid 与 analytic 的使用模板

**模板 E.10（solid）.** 取

$$
\mathcal C=D(\mathbf{CondAb}),
\qquad
\Sigma=\{\mathbb Z[\underline S]\to\mathbb Z^\square[S]\}_{S\in\mathbf{ProFin}}.
$$

若接受 Scholze 输入定理：由 $\Sigma$ 生成的核是张量理想，则：

1. $D_\square(\mathbb Z)$ 是幺半局部化。
2. solid 张量积为
   $$
   M\otimes_{\mathbb Z}^{L,\square}N
   =
   L^\square(M\otimes_{\mathbb Z}^LN).
   $$
3. solid 环和 solid 模分别是局部幺半范畴中的代数对象和模对象。
4. 相对 solid 张量积由命题 E.9 给出。

**模板 E.11（analytic）.** 取凝聚环 $A$ 和测度对象赋值 $\mathcal M$，令

$$
\Sigma_{\mathcal M}
=
\{A[\underline S]\to\mathcal M[S]\}_{S}.
$$

若 $(A,\mathcal M)$ 满足 analytic ring 条件，使得 $\Sigma_{\mathcal M}$ 的局部对象形成幺半反射子范畴，则：

1. 解析化 $L_{(A,\mathcal M)}$ 是左伴随。
2. 解析张量积为
   $$
   M\otimes_{(A,\mathcal M)}^LN
   =
   L_{(A,\mathcal M)}(M\otimes_A^LN).
   $$
3. 对解析对象 $C$，
   $$
   R\operatorname{Hom}(L_{(A,\mathcal M)}M,C)
   \simeq
   R\operatorname{Hom}(M,C).
   $$

## E.6 错误类比的排除

**命题 E.12.** 若只知道 $L$ 是反射局部化，不能推出 $L$ 与张量积相容。

**证明.** 张量相容需要证明核 $\ker L$ 是张量理想。若存在 $N\in\ker L$ 与 $X\in\mathcal C$ 使 $N\otimes X\notin\ker L$，则 $0\simeq LN$，但

$$
L(N\otimes X)\not\simeq0\simeq LN\otimes_L LX.
$$

因此 $L$ 不能是幺半局部化。证毕。

**推论 E.13.** 普通张量积公式不能直接搬到 solid 或 analytic 范畴。每次使用无限乘积、profinite 测度对象或解析测度对象时，必须说明张量积是否已经过相应局部化。

**证明.** solid/analytic 张量积的定义包含局部化。若省略局部化，相当于在 $\mathcal C$ 中计算；若核不是已知张量理想，则结果可能不落在局部对象中。证毕。

## E.7 本附录小结

第二卷中“形式上正确”的推理链是：

1. 给出态射族 $\Sigma$。
2. 引用输入定理确认局部化 $L$ 存在。
3. 引用输入定理确认 $\ker L$ 是张量理想。
4. 用定理 E.7 下降幺半结构。
5. 用命题 E.9 处理相对张量积。
6. 用模板 E.10 或 E.11 回到 solid/analytic 具体对象。

其中 4、5 是本书证明的范畴论步骤；2、3 是 Scholze 理论的核心输入。

## 练习

**练习 E.1.** 证明命题 E.2 中的 retract 封闭性。

**练习 E.2.** 在命题 E.5 中写出 cofiber sequence 经 $L$ 后仍为 cofiber sequence 的理由。

**练习 E.3.** 用命题 E.9 推导 solid 交换环 $A$ 上的公式

$$
M\otimes_A^{L,\square}N
\simeq
L^\square(M\otimes_A^LN).
$$

**练习 E.4.** 给出一个形式例子说明“反射子范畴”不自动给出“幺半反射子范畴”。
