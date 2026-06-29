# 符号约定

本文件记录《同伦类型论与单值基础》的全书符号。后续章节新增核心符号时必须同步更新。

## 判断与语境

- $\Gamma\ \mathsf{ctx}$：$\Gamma$ 是合法语境。
- $\Gamma\vdash A:\mathcal U_i$：在语境 $\Gamma$ 中，$A$ 是第 $i$ 层宇宙中的类型。
- $\Gamma\vdash a:A$：在语境 $\Gamma$ 中，$a$ 是类型 $A$ 的项。
- $\Gamma\vdash a\equiv b:A$：$a$ 与 $b$ 在类型 $A$ 中 judgmentally equal，也称 definitional equality。
- $\Gamma,x:A$：语境扩张。若 $A$ 依赖于 $\Gamma$，则 $x$ 可在后续类型和项中出现。
- $B[a/x]$：把 $a$ 替换到 $B$ 中的自由变量 $x$。

## 基础类型构造

- $\prod_{x:A}B(x)$ 或 $\Pi_{x:A}B(x)$：依赖函数类型。
- $\sum_{x:A}B(x)$ 或 $\Sigma_{x:A}B(x)$：依赖对类型。
- $A\to B$：非依赖函数类型，即 $\prod_{x:A}B$，其中 $B$ 不依赖于 $x$。
- $A\times B$：非依赖积类型，即 $\sum_{x:A}B$，其中 $B$ 不依赖于 $x$。
- $\mathbf 0$：空类型。
- $\mathbf 1$：单位类型，其规范元素记为 $\star$。
- $A+B$：和类型。
- $\mathbb N$：自然数类型。
- $\mathbb Z$：整数类型；第十一章默认指附录 M 的归纳整数 $\mathbb Z_{\mathsf{ind}}$。
- $0_{\mathbb Z}$、$\mathsf{succ}_{\mathbb Z}$、$\mathsf{pred}_{\mathbb Z}$：整数的零、后继和前驱。

## 恒等类型与路径

- $\mathsf{Id}_A(a,b)$ 或 $a=_A b$：$a$ 与 $b$ 的恒等类型，也称路径类型。
- $\mathsf{refl}_a:a=_A a$：反身路径。
- 若 $p:a=b$，则 $p^{-1}:b=a$ 表示逆路径。
- 若 $p:a=b$ 且 $q:b=c$，则 $p\cdot q:a=c$ 表示路径复合。
- 若 $f:A\to B$ 且 $p:x=y$，则 $\mathsf{ap}_f(p):f(x)=f(y)$。
- 若 $f:\prod_{x:A}P(x)$ 且 $p:x=y$，则 $\mathsf{apd}_f(p):\mathsf{transport}^{P}(p,f(x))=f(y)$。
- 若 $P:A\to\mathcal U$ 且 $p:x=y$，则 $\mathsf{transport}^P(p):P(x)\to P(y)$。

## 等价与同伦层级

- $\mathsf{fib}_f(y)\coloneqq \sum_{x:A}(f(x)=y)$：函数 $f:A\to B$ 在 $y:B$ 处的 fiber。
- $\mathsf{isContr}(A)\coloneqq \sum_{c:A}\prod_{x:A}(c=x)$：$A$ 可收缩。
- $\mathsf{isProp}(A)\coloneqq \prod_{x,y:A}(x=y)$：$A$ 是命题。
- $\mathsf{isSet}(A)\coloneqq \prod_{x,y:A}\mathsf{isProp}(x=y)$：$A$ 是集合。
- $A\simeq B$：$A$ 与 $B$ 等价；具体定义以后续等价章节为准。
- $\mathsf{isEquiv}(f)$：函数 $f$ 是等价；本书以 fiber 可收缩定义为基准。
- $\mathsf{idtoequiv}_{A,B}:(A=B)\to(A\simeq B)$：从类型相等得到等价的映射。
- $\mathsf{ua}$：单值性给出的 $(A\simeq B)\to(A=B)$ 的方向或等价的逆方向，具体符号随章节说明。
- $\|A\|_n$：$n$-截断；$\|A\|$ 表示命题截断。
- $\mathsf{isOfHLevel}_n(A)$：$A$ 具有同伦层级 $n$。本书采用 HoTT Book 常见编号：$0$ 表示可收缩，$1$ 表示命题，$2$ 表示集合。

## 高阶归纳类型与合成同伦

- $\mathbb S^1$：圆类型，点构造子为 $\mathsf{base}$，路径构造子为 $\mathsf{loop}:\mathsf{base}=\mathsf{base}$。
- $\mathsf{susp}(A)$：$A$ 的悬挂。
- $\mathsf{pushout}(f,g)$：两个映射 $f:A\to B$、$g:A\to C$ 的 pushout 高阶归纳类型。
- $\pi_1(X,x_0)$：基点类型 $(X,x_0)$ 的基本群；严格定义见第十一章。
- $K(G,n)$：Eilenberg-Mac Lane 型；本书只在研究边界章节作为外部输入和形式化目标使用。
- $H^n(X;G)$：以 $K(G,n)$ 表示的第 $n$ 阶合成上同调群。
- $\widetilde H^n(X;G)$：带基点类型的第 $n$ 阶约化上同调群。
- $\smile$：上同调 cup product。

## 范畴论

- $\mathcal C$：预范畴或单值范畴。
- $\mathcal C^{\mathsf{op}}$：$\mathcal C$ 的反范畴。
- $\mathcal C(x,y)$：对象 $x,y$ 之间的 Hom 类型。
- $x\cong y$：范畴中的同构。
- $\mathsf{idtoiso}_{x,y}:(x=y)\to(x\cong y)$：对象相等诱导同构。
- $\mathsf{isUnivalentCat}(\mathcal C)$：范畴 $\mathcal C$ 是单值范畴，即 $\mathsf{idtoiso}$ 是等价。
- $F:\mathcal C\to\mathcal D$：预范畴之间的函子。
- $F\Rightarrow G$：函子 $F,G:\mathcal C\to\mathcal D$ 之间的自然变换类型。
- $F\cong_{\mathsf{nat}}G$：函子之间的自然同构类型。
- $[\mathcal C,\mathcal D]$：从 $\mathcal C$ 到 $\mathcal D$ 的函子范畴。
- $y(c)$：Yoneda 嵌入中的可表预层 $\mathcal C(-,c)$。
- $\mathsf{Nat}(P,Q)$：预层或函子之间的自然变换类型。
- $\widehat{\mathcal C}$：预范畴 $\mathcal C$ 的 Rezk 完备化。

## 宇宙约定

- 本书默认使用层级宇宙 $\mathcal U_0,\mathcal U_1,\ldots$。
- 除非章节明确声明，不假设 resizing。
- 若需要 cumulativity，将在相关章节中明确说明；默认规则只使用已声明的宇宙提升。
