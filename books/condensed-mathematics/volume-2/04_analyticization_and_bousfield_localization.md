# 第四章：解析化与 Bousfield localization

解析模把普通自由 Dirac 组合和允许测度视作同一种测试数据。要把任意 $A$-模送到最接近
它的解析对象，就应强制所有映射

$$
A[\underline S]\to\mathcal M[S]
$$

在局部化后成为同构。等价地，它们的 cofiber $K_S^{\mathcal M}$ 必须被杀掉，而局部
对象恰是对全部 $K_S^{\mathcal M}$ 右正交的复形。这把第三章的 Hom 公理转化成一个
可以用反射泛性质操作的 Bousfield localization。

以下固定第三章的 analytic ring $(A,\mathcal M)$，区分局部对象、零化对象与反射函子，
并证明接受反射存在性后得到的基本形式结论。幺半相容性不是任意 Bousfield localization
自动具有的性质；核为张量理想的判别和相对张量下降在附录 K 展开，正文只调用精确的
Scholze 输入定理来定义 analytic tensor。

## 4.1 局部对象与零化对象

固定解析环 $(A,\mathcal M)$。设

$$
K_S^{\mathcal M}
=
\operatorname{cofib}(A[\underline S]\to\mathcal M[S]).
$$

**定义 4.1.** 对象 $C\in D(A)$ 称为 $\mathcal M$-局部对象，如果

$$
R\underline{\operatorname{Hom}}_A(K_S^{\mathcal M},C)\simeq0
$$

对所有极不连通 $S$ 成立。

**定义 4.2.** 对象 $N\in D(A)$ 称为 $\mathcal M$-零化对象，如果对所有 $\mathcal M$-局部对象 $C$，

$$
R\operatorname{Hom}_A(N,C)\simeq0.
$$

这里刻意使用普通导出 Hom：反射伴随直接控制 mapping complexes。局部对象定义 4.1
中的 internal Hom 消失更强，并来自 analytic ring 的本质像判别；不能仅由抽象伴随把
本定义中的 Hom 擅自加上下划线。

## 4.2 解析化函子

**外部输入定理 4.3（Scholze）.** 对满足第三章定义 3.3 的解析环
$(A,\mathcal M)$，包含函子

$$
D(A,\mathcal M)\hookrightarrow D(A)
$$

有左伴随

$$
L_{(A,\mathcal M)}:D(A)\to D(A,\mathcal M).
$$

该函子称为解析化。

**来源与边界.** 这是 S26 Proposition 7.5(ii)。反射性是 analytic 公理的结构后果，
不是任意一族 cones 自动具有的性质。

**泛性质 4.4.** 对任意 $C\in D(A)$ 和任意解析对象 $N\in D(A,\mathcal M)$，有自然同构

$$
R\operatorname{Hom}_A(L_{(A,\mathcal M)}C,N)
\simeq
R\operatorname{Hom}_A(C,N).
$$

**证明.** 这是稳定增强中左伴随的 mapping-spectrum 等价，转成普通导出 Hom 复形后的
陈述。Internal Hom 版本需要额外的 enriched/closed compatibility，不是左伴随定义本身。
证毕。

## 4.3 Bousfield localization 表述

**命题 4.5.** 若解析化函子存在，则对任意 $C$，cofiber

$$
\operatorname{cofib}(C\to L_{(A,\mathcal M)}C)
$$

是 $\mathcal M$-零化对象。

**证明.** 设 $N$ 是解析对象。由泛性质 4.4，

$$
R\operatorname{Hom}_A(L_{(A,\mathcal M)}C,N)
\to
R\operatorname{Hom}_A(C,N)
$$

是同构。把它放入 $R\operatorname{Hom}(-,N)$ 作用于三角

$$
C\to L_{(A,\mathcal M)}C\to Q\to
$$

所得的 fiber sequence，得到

$$
R\operatorname{Hom}_A(Q,N)\simeq0.
$$

证毕。

**命题 4.6.** 对所有极不连通 $S$，

$$
L_{(A,\mathcal M)}A[\underline S]\simeq L_{(A,\mathcal M)}\mathcal M[S].
$$

**证明.** 先证 $L_{(A,\mathcal M)}K_S^{\mathcal M}\simeq0$。对任意局部对象
\(N\)，命题 3.7 给出 internal Hom 消失；取全局截面后得到
\(R\operatorname{Hom}_A(K_S^{\mathcal M},N)=0\)。泛性质 4.4 将它识别为
\(R\operatorname{Hom}_A(LK_S^{\mathcal M},N)=0\)。取局部对象
\(N=LK_S^{\mathcal M}\)，则恒等态射为零，故 \(LK_S^{\mathcal M}=0\)。

再由 cofiber sequence

$$
A[\underline S]\to\mathcal M[S]\to K_S^{\mathcal M}\to
$$

应用 $L_{(A,\mathcal M)}$，得到 $L_{(A,\mathcal M)}K_S^{\mathcal M}\simeq0$，因此前两项同构。证毕。

## 4.4 解析张量积

若 $A$ 交换，则 $D(A)$ 有派生张量积 $\otimes_A^L$。

**外部输入定理 4.7（Scholze）.** 若 \(A\) 交换，则 $D(A,\mathcal M)$ 在

$$
C\otimes_{(A,\mathcal M)}^L D
=
L_{(A,\mathcal M)}(C\otimes_A^L D)
$$

下成为闭对称幺半范畴。

这一定理要求 localization 与张量积相容，来源为 S26 Proposition 7.5(ii)。它只给出
导出 analytic tensor 的存在；把它识别为心脏层 tensor 的总左导出还需第三章警告 3.6
中的 degree-zero 集中性。

## 4.5 与 solidification 的比较

对 $(A,\mathcal M)=(\mathbb Z,\mathbb Z^\square)$，

$$
L_{(A,\mathcal M)}=L^\square.
$$

于是解析化就是派生 solidification，解析张量积就是派生 solid 张量积。

## 4.6 杀掉 Dirac 与测度的差

解析化反射恰好杀掉

$$
\operatorname{cofib}(A[\underline S]\to\mathcal M[S])
$$

生成的局部化核，因而其局部对象正是解析复形。核的张量理想性允许普通派生张量在反射
后下降为 analytic tensor；在 $(\mathbb Z,\mathbb Z^\square)$ 情形，这一构造就是
solidification。第五章将更换底环与测度为
$(\underline{\mathbb R},\mathcal M_{<p})$，并检验经典函数空间何时真正落入相应局部
对象，而不是只在名称上被称作 liquid。

## 练习

**练习 4.1.** 证明命题 4.5。

**练习 4.2.** 在 solid 例子中写出 $K_S^{\mathcal M}$。

**练习 4.3.** 说明为什么解析张量积必须在普通张量积后再解析化。

**练习 4.4.** 比较 Bousfield localization 与阿贝尔范畴中的反射子范畴。
