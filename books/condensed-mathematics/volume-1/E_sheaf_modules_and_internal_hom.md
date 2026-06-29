# 附录 E：sheaf 模、内部 Hom 与派生张量

## E.0 目标

正文第九至第十一章使用了 sheaf of modules 的标准代数：张量积、内部 Hom、右正合性以及派生张量。本附录把这些事实集中列出，避免正文中每次都重新建立一般理论。

本附录的环境是一个 Grothendieck topos $\mathcal E$。在本书中主要取

$$
\mathcal E=\mathbf{CondSet}
$$

或某个站点上的 sheaf topos。

## E.1 环对象与模对象

**定义 E.1.** $\mathcal E$ 中的环对象 $R$ 是一个阿贝尔群对象，配备乘法和单位

$$
R\times R\to R,\qquad 1\to R
$$

满足结合律、分配律和单位律。若乘法交换，则称为交换环对象。

**定义 E.2.** $R$-模对象是阿贝尔群对象 $M$，配备作用

$$
R\times M\to M
$$

满足通常的模公理。所有 $R$-模构成范畴

$$
R\text{-}\mathbf{Mod}(\mathcal E).
$$

当 $\mathcal E=\mathbf{CondSet}$ 且 $R$ 是凝聚环时，这就是正文中的凝聚 $R$-模范畴。

## E.2 阿贝尔性与局部正合性

**定理 E.3.** $R\text{-}\mathbf{Mod}(\mathcal E)$ 是 Grothendieck 阿贝尔范畴。特别地，它有所有小极限和小余极限、filtered colimits 正合，并有生成元。

**证明.** 先在预层 $R$-模范畴中逐对象构造 kernel、cokernel、极限和余极限。预层模范畴是函子范畴，因此是 Grothendieck 阿贝尔范畴。

令 $a$ 表示 sheafification。对 sheaf 模态射 $f:M\to N$，kernel 已经是 sheaf，且逐对象计算：

$$
(\ker f)(U)=\ker(M(U)\to N(U)).
$$

cokernel 则是预层 cokernel 的 sheafification：

$$
\operatorname{coker}_{\operatorname{Sh}}(f)=a(\operatorname{coker}_{\operatorname{PSh}}(f)).
$$

sheafification 是左伴随，所以保持余极限；同时在阿贝尔群值 sheaf 情形中它是正合函子。于是 cokernel、image、coimage 的通常比较同构从预层模范畴传到 sheaf 模范畴，得到阿贝尔性。

filtered colimits 在预层中逐对象计算且正合；sheafification 保持 colimit 并正合，因此 sheaf 模范畴中 filtered colimits 正合。生成元可取站点对象 $U$ 对应的自由模

$$
R[h_U]
$$

的集合族：若 $M\ne0$，则存在 $U$ 和 $s\in M(U)$ 非零，$s$ 对应一个非零态射 $R[h_U]\to M$。故 $R\text{-}\mathbf{Mod}(\mathcal E)$ 是 Grothendieck 阿贝尔范畴。证毕。

**命题 E.4.** 一个 $R$-模态射

$$
M\to N
$$

是单射、满射或同构，当且仅当在站点的覆盖意义下局部满足相应性质。若站点有足够多的投射测试对象，例如极不连通对象，则可在这些对象上检测。

**证明.** 单射由 kernel 为零检测，满射由 cokernel 为零检测。同构等价于 kernel 与 cokernel 同时为零。若 sheaf $K$ 为零，则任意对象上截面为零；反过来，若每个截面在某个覆盖上局部为零，则由 sheaf 的 separated 性可知该截面为零。因此零对象、kernel 和 cokernel 都可局部检测。若站点有覆盖基，只需在覆盖基对象上检测。证毕。

## E.3 张量积

设 $R$ 是交换环对象。

**定义 E.5.** 对 $R$-模 $M,N$，其张量积

$$
M\otimes_R N
$$

定义为表示如下双线性映射函子的对象：

$$
\operatorname{Hom}_{R}(M\otimes_R N,P)
\cong
\operatorname{Bilin}_R(M,N;P).
$$

等价地，它是预层

$$
U\mapsto M(U)\otimes_{R(U)}N(U)
$$

的适当 sheaf 化，并取满足 sheaf 意义双线性泛性质的对象。

**命题 E.6.** 函子

$$
M\otimes_R-:R\text{-}\mathbf{Mod}(\mathcal E)\to R\text{-}\mathbf{Mod}(\mathcal E)
$$

是左伴随，因此保持所有余极限。特别地，它右正合。

**证明.** 由定义，$M\otimes_R-$ 左伴随于内部 Hom 函子 $\mathcal Hom_R(M,-)$。左伴随保持余极限；在阿贝尔范畴中保持 cokernel 等价于右正合。证毕。

## E.4 内部 Hom

**定义 E.7.** 对 $R$-模 $M,N$，内部 Hom

$$
\mathcal Hom_R(M,N)
$$

是满足

$$
\operatorname{Hom}_R(P,\mathcal Hom_R(M,N))
\cong
\operatorname{Hom}_R(P\otimes_R M,N)
$$

的 $R$-模。

在站点语言中，它可描述为

$$
U\mapsto
\operatorname{Hom}_{R|_U}(M|_U,N|_U),
$$

并带有自然 sheaf 结构。

**命题 E.8.** 对任意 $M,N,P$，有自然伴随同构

$$
\operatorname{Hom}_R(P\otimes_R M,N)
\cong
\operatorname{Hom}_R(P,\mathcal Hom_R(M,N)).
$$

因此 $R\text{-}\mathbf{Mod}(\mathcal E)$ 是闭对称幺半范畴。

**证明.** 由定义，

$$
\mathcal Hom_R(M,N)(U)
=
\operatorname{Hom}_{R|_U}(M|_U,N|_U).
$$

给定 $\alpha:P\otimes_R M\to N$，对每个 $U$ 和 $s\in P(U)$，得到 $R|_U$-线性映射

$$
M|_U\to N|_U,\qquad m\mapsto \alpha(s\otimes m).
$$

这定义了 $P\to\mathcal Hom_R(M,N)$。反向地，给定 $\beta:P\to\mathcal Hom_R(M,N)$，用 evaluation

$$
\mathcal Hom_R(M,N)\otimes_R M\to N
$$

复合 $\beta\otimes\operatorname{id}_M$ 得到 $P\otimes_RM\to N$。两种构造互逆，且与限制映射相容。证毕。

## E.5 平坦对象与左导出张量

**定义 E.9.** $R$-模 $F$ 称为平坦，如果

$$
F\otimes_R- 
$$

是正合函子。

**命题 E.10.** 若 $F$ 是形如 $R[\underline S]$ 的自由 $R$-模，且 $\underline S$ 是投射测试对象生成的自由凝聚集合，则 $F$ 是平坦的。

**证明.** 自由模上的张量积满足

$$
R[\underline S]\otimes_R M\simeq M[\underline S],
$$

即按 $\underline S$ 取自由扩张。若 $\underline S$ 来自投射测试对象，则 $M[\underline S]$ 可由在该测试对象上取截面描述；取值函子正合，故该函子保持 kernel 和 cokernel。于是 $R[\underline S]\otimes_R-$ 正合，$R[\underline S]$ 平坦。证毕。

**定理 E.11.** 导出范畴

$$
D(R)
$$

上存在左导出张量积

$$
-\otimes_R^L-.
$$

它可通过 K-flat 分解计算，并满足

$$
H^0(M\otimes_R^L N)=M\otimes_R N
$$

当 $M,N$ 置于次数 $0$ 且没有高 Tor 贡献时。

**证明.** 附录 H 命题 H.11 说明 $R\text{-}\mathbf{Mod}(\mathcal E)$ 是 Grothendieck 阿贝尔范畴。由附录 H 输入定理 H.13，每个复形有 K-flat 替换。对 $M^\bullet,N^\bullet$，取 K-flat 替换 $P^\bullet\to M^\bullet$，定义

$$
M^\bullet\otimes_R^LN^\bullet=P^\bullet\otimes_RN^\bullet.
$$

附录 H 命题 H.15 证明该定义与替换选择无关。因此得到导出范畴上的双函子 $-\otimes_R^L-$。若 $M,N$ 集中在次数 $0$，则第零同调是右正合函子 $-\otimes_RN$ 的第零左导出，等于普通张量积；高同调正是相应 Tor 项。证毕。

## E.6 Tor 群

**定义 E.12.** 对 $R$-模 $M,N$，定义

$$
\operatorname{Tor}_i^R(M,N)=H^{-i}(M\otimes_R^L N).
$$

若取同调次数约定，也可写为 $H_i$。本书采用上同调次数，因此负次数对应正的 Tor 指标。

**命题 E.13.** 若 $F$ 平坦，则

$$
\operatorname{Tor}_i^R(F,N)=0,\qquad i>0.
$$

**证明.** 平坦性说明 $F\otimes_R-$ 已经正合，因此无需分解 $F$ 即可计算导出张量。高同调为零。证毕。

## E.7 对正文的应用

1. 第九章的凝聚环和张量积使用定义 E.5。
2. 第十章的凝聚模范畴使用定理 E.3。
3. 第十一章的派生张量积使用定理 E.11。
4. 第十二章以后的 solid 和 analytic 张量积是在上述张量积之后再做 solidification 或 analytic localization。

## 练习

**练习 E.1.** 在预层模范畴中证明张量积的双线性泛性质。

**练习 E.2.** 证明 sheaf 化后的张量积仍满足 sheaf 范畴中的双线性泛性质。

**练习 E.3.** 设 $R$ 是凝聚环。证明 $R\otimes_R M\simeq M$。

**练习 E.4.** 证明若 $F$ 平坦，则短正合列

$$
0\to M'\to M\to M''\to0
$$

经 $F\otimes_R-$ 后仍正合。
