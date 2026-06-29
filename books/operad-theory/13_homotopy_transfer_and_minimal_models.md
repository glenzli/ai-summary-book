# 第十三章：同伦转移定理与最小模型

## 本章目标

本章说明 operad 同伦代数如何在 quasi-isomorphism 下转移。核心目标是：

1. 定义 contraction 和 strong deformation retract。
2. 陈述同伦转移定理。
3. 给出 $A_\infty$ 和 $L_\infty$ 转移的树公式。
4. 定义 minimal model。
5. 说明 higher operations 与 Massey products、formality obstruction 的关系。

本章的完整符号证明依赖 homological perturbation theory。为保持正文可检查，本章给出精确公式框架，并把全符号版本标为外部输入。

## 依赖前置知识

需要第九章的 bar-cobar 构造、第十章的 $A_\infty/L_\infty$-代数，以及链复形的 homotopy 和 quasi-isomorphism。

## 13.1 Contraction

**定义 13.1.** 设 $(A,d_A)$ 和 $(H,d_H)$ 是链复形。一个 contraction
$$
\left(
H
\xrightarrow{i}
A
\xrightarrow{p}
H,
h:A\to A[1]
\right)
$$
由链映射 $i,p$ 和次数 $+1$ 的线性映射 $h$ 组成，满足
$$
p i=\operatorname{id}_H,
$$
以及
$$
i p-\operatorname{id}_A=d_A h+h d_A.
$$
若还满足 side conditions
$$
h i=0,\qquad p h=0,\qquad h^2=0,
$$
则称为 strong deformation retract data。

**解释 13.2.** $H$ 是 $A$ 的一个同伦 retract。等式
$$
i p-\operatorname{id}_A=d_A h+h d_A
$$
说明 $ip$ 与 $\operatorname{id}_A$ 链同伦。若 $H=H_\*(A)$ 且 $d_H=0$，这样的 contraction 是把链复形压缩到其同调上的具体选择。

**命题 13.3.** 若存在 contraction $H\rightleftarrows A$，则 $i:H\to A$ 和 $p:A\to H$ 都是 quasi-isomorphism。

**证明.** 因为 $pi=\operatorname{id}_H$，所以 $H_\*(p)H_\*(i)=\operatorname{id}_{H_\*(H)}$。又因为 $ip$ 与 $\operatorname{id}_A$ 链同伦，二者诱导同一个同调映射，所以
$$
H_\*(i)H_\*(p)=\operatorname{id}_{H_\*(A)}.
$$
因此 $H_\*(i)$ 和 $H_\*(p)$ 互为逆。$\square$

## 13.2 Homological perturbation lemma

**外部输入定理 13.4.** 设
$$
(H,d_H)\xrightarrow{i}(A,d_A)\xrightarrow{p}(H,d_H),h
$$
是 contraction。若 $\delta:A\to A[-1]$ 是微分扰动，使得
$$
(d_A+\delta)^2=0
$$
且 $h\delta$ 局部幂零，则存在转移后的微分
$$
d_H'=d_H+p\delta(1-h\delta)^{-1}i
$$
以及修正后的 contraction
$$
(H,d_H')\rightleftarrows(A,d_A+\delta).
$$
这里
$$
(1-h\delta)^{-1}=\sum_{r\ge0}(h\delta)^r
$$
在局部幂零条件下对每个元素是有限和。

**说明 13.5.** Homological perturbation lemma 是同伦转移公式的技术核心。Operad 转移定理可视为把代数结构编码成 bar/cobar 微分后，对该微分应用扰动引理。

## 13.3 同伦转移定理

**外部输入定理 13.6.** 设 $\mathcal P$ 是 Koszul operad，$\mathcal P_\infty=\Omega\mathcal P^¡$。若 $A$ 是 $\mathcal P_\infty$-代数，并且有 contraction
$$
H\rightleftarrows A,
$$
则：

1. $H$ 自然获得 $\mathcal P_\infty$-代数结构；
2. $i:H\to A$ 延拓为 $\mathcal P_\infty$-algebra 的 $\infty$-morphism；
3. $p:A\to H$ 也可延拓为反向 $\infty$-morphism；
4. 这些 $\infty$-morphism 在底层链复形上是 quasi-isomorphism。

若 $H=H_\*(A)$ 且 $d_H=0$，所得 $\mathcal P_\infty$-代数称为 $A$ 的 minimal model 候选。

**证明思想.** 把 $\mathcal P_\infty$-代数结构视为 cofree coalgebra 或 bar construction 上的 coderivation。Contraction 给出底层链复形的同伦等价。对 coderivation 的非线性部分应用 homological perturbation lemma，得到 $H$ 上的新 coderivation。Coderivation 平方为零等价于 $\mathcal P_\infty$-relations，因此 $H$ 得到转移结构。$\square$

## 13.4 $A_\infty$ 转移公式

**设定 13.7.** 设 $(A,d,\mu)$ 是 dg associative algebra，并有 contraction
$$
H\xrightarrow{i}A\xrightarrow{p}H,\qquad h:A\to A[1].
$$
假设 $H$ 的微分为 $d_H$。转移后的 $A_\infty$-结构记为
$$
m_n^H:H^{\otimes n}\to H.
$$

**公式 13.8.** 低阶运算为
$$
m_1^H=d_H,
$$
$$
m_2^H(x,y)=p\mu(i x,i y),
$$
而 $m_3^H$ 是两棵平面二叉树的和：
$$
m_3^H(x,y,z)
=
p\mu(h\mu(i x,i y),i z)
\pm
p\mu(i x,h\mu(i y,i z)).
$$
符号由第九章的悬挂约定决定。

一般地，
$$
m_n^H=\sum_{T\in\operatorname{PBT}_n}\pm\, m_T,
$$
其中 $\operatorname{PBT}_n$ 是有 $n$ 个叶的平面二叉树集合。给定树 $T$：

- 每个叶放入 $i$；
- 每个内部顶点放入乘法 $\mu$；
- 每条内部边放入 $h$；
- 根部放入 $p$。

**命题 13.9.** 公式 13.8 给出的低阶关系满足 $A_\infty$ 恒等式的 $n=1,2,3$ 部分。

**证明.** $n=1$ 部分是 $d_H^2=0$。$n=2$ 部分要求 $d_H$ 对 $m_2^H$ 是导子；这由 $i,p$ 是链映射以及 $\mu$ 是链映射推出。$n=3$ 部分比较
$$
m_2^H(m_2^H(x,y),z)
\quad\text{和}\quad
m_2^H(x,m_2^H(y,z)).
$$
把 $ip$ 替换为
$$
\operatorname{id}_A+d h+h d
$$
后，严格结合的乘法项相消，剩余项正是 $m_3^H$ 与 $d_H$ 组成的边界项。完整符号由悬挂约定给出。$\square$

**外部输入定理 13.10.** 公式 13.8 的全体树和给出 $H$ 上的 $A_\infty$-代数结构，并且 $i$ 延拓为 $A_\infty$ quasi-isomorphism
$$
H\rightsquigarrow A.
$$
该定理通常称为 Kadeishvili transfer theorem 或 $A_\infty$ homotopy transfer theorem。

**说明 13.10.1.** 附录 J 给出本书使用的平面二叉树递归、normalized contraction side conditions、低阶恒等式检查和 $A_\infty$ quasi-isomorphism 分量。正文中的 $\pm$ 默认由附录 E 的 suspended convention 和附录 J 的树递归共同决定。

## 13.5 $L_\infty$ 转移公式

**设定 13.11.** 设 $(\mathfrak g,d,[-,-])$ 是 dg Lie algebra，并有 contraction
$$
H\xrightarrow{i}\mathfrak g\xrightarrow{p}H,\qquad h:\mathfrak g\to\mathfrak g[1].
$$

**公式 13.12.** 转移后的 $L_\infty$ 结构 $\ell_n^H$ 由有根二叉树求和给出：

- 每个叶放入 $i$；
- 每个内部顶点放入 Lie bracket；
- 每条内部边放入 $h$；
- 根部放入 $p$；
- 对所有叶标号的 shuffle 取带 Koszul 符号的反对称化。

低阶项为
$$
\ell_1^H=d_H,
$$
$$
\ell_2^H(x,y)=p[i x,i y],
$$
以及
$$
\ell_3^H(x,y,z)
=
\sum_{\text{cyclic}}\pm\,p[h[i x,i y],i z].
$$

**外部输入定理 13.13.** 公式 13.12 给出 $H$ 上的 $L_\infty$-代数结构，并且 $i$ 延拓为 $L_\infty$ quasi-isomorphism
$$
H\rightsquigarrow\mathfrak g.
$$

**说明 13.13.1.** 附录 J 把 $L_\infty$ 转移写成有根二叉树、shuffle 和 Koszul 反对称化的组合。完整 signs 仍属于外部输入定理 13.13 的一部分；正文只使用与附录 E 相容的 convention。

## 13.6 Minimal model

**定义 13.14.** 一个 $A_\infty$-代数或 $L_\infty$-代数称为 minimal，若其一元结构映射为零：
$$
m_1=0
\quad\text{或}\quad
\ell_1=0.
$$
更一般地，$\mathcal P_\infty$-代数称为 minimal，若底层链复形微分为零。

**外部输入定理 13.15.** 设 $A$ 是 dg associative algebra over a field。则 $H_\*(A)$ 上存在 minimal $A_\infty$-结构，并存在 $A_\infty$ quasi-isomorphism
$$
H_\*(A)\rightsquigarrow A.
$$
该 minimal model 在 $A_\infty$ quasi-isomorphism 意义下唯一。

**外部输入定理 13.16.** 设 $\mathfrak g$ 是 dg Lie algebra over a field。则 $H_\*(\mathfrak g)$ 上存在 minimal $L_\infty$-结构，并存在 $L_\infty$ quasi-isomorphism
$$
H_\*(\mathfrak g)\rightsquigarrow\mathfrak g.
$$
该 minimal model 在 $L_\infty$ quasi-isomorphism 意义下唯一。

**解释 13.17.** Minimal model 把微分消去，把原链级对象的同伦信息编码到高阶运算 $m_n$ 或 $\ell_n$ 中。若所有高阶运算都可通过 $\infty$-isomorphism 消去，则对象是 formal。

## 13.7 Formality 与 Massey products

**定义 13.18.** 一个 dg associative algebra $A$ 称为 formal，若它与其同调代数 $H_\*(A)$ 通过 dg algebra quasi-isomorphism 的 zigzag 相连，其中 $H_\*(A)$ 带零微分和诱导乘法。

**命题 13.19.** 若 $A$ 的 minimal $A_\infty$-model 可取为
$$
m_n=0\qquad(n\ge3),
$$
则 $A$ 是 formal。

**证明.** 若 minimal model 只有 $m_2$，则它正是同调代数 $H_\*(A)$ 作为 dg algebra 的结构，微分为零。minimal model theorem 给出
$$
H_\*(A)\rightsquigarrow A
$$
的 $A_\infty$ quasi-isomorphism。若该 quasi-isomorphism 可由 dg algebra quasi-isomorphism zigzag 严格实现，则得到 formality。标准最小模型理论说明在域上该条件等价于高阶 $A_\infty$ 结构可通过 $A_\infty$-isomorphism 消去。$\square$

**说明 13.20.** $m_3$ 常与三重 Massey product 相关。更高 $m_n$ 可视为更高 Massey operations 的 operadic 组织方式。因此 nonzero higher operations 通常给出 formality obstruction。不过具体 Massey product 可能依赖选择；minimal $A_\infty$ 结构给出更系统的记录。

## 13.8 计算流程

实际计算 minimal model 通常按以下步骤：

1. 取链复形 $A$ 的同调 $H=H_\*(A)$。
2. 选择代表元嵌入 $i:H\to A$。
3. 选择投影 $p:A\to H$。
4. 构造同伦 $h$，满足 $ip-\operatorname{id}_A=dh+hd$。
5. 对每个 $n$，按树公式求 $m_n^H$ 或 $\ell_n^H$。
6. 判断高阶运算是否可通过 $\infty$-isomorphism 消去。

这个流程的数学内容在于第 4 步和第 6 步：选择不同 contraction 会给出不同公式，但所得 minimal models 在 $\infty$-isomorphism 意义下等价。

## 本章小结

同伦转移定理说明：若链复形 $A$ 上有 $\mathcal P_\infty$-代数结构，并且 $A$ 与 $H$ 同伦等价，则 $H$ 上也有自然的 $\mathcal P_\infty$-代数结构。对 dg associative algebra，转移公式由平面二叉树控制；对 dg Lie algebra，转移公式由反对称化的有根树控制。Minimal model 把微分信息转化为高阶运算，是 formality、Massey products 和同伦分类的基本工具。

附录 J 是本章的计算附录。需要实际计算 $m_3$、$m_4$、$\ell_3$ 或 $\infty$-morphism 分量时，应先固定 normalized contraction，再按附录 J 的树递归展开。

## 练习

**练习 13.1.** 证明 contraction 中 $i$ 和 $p$ 都是 quasi-isomorphism。

**练习 13.2.** 对一个给定 contraction，写出 $m_3^H$ 的两棵平面二叉树项。

**练习 13.3.** 设 $A$ 是 dg associative algebra，证明若 $h=0$ 且 $ip=\operatorname{id}_A$，则所有 $m_n^H$ 对 $n\ge3$ 为零。

**练习 13.4.** 写出 $\ell_3^H$ 的 cyclic 求和项，并说明反对称化为何必要。

**练习 13.5.** 查找一个有非零三重 Massey product 的 dg algebra，并解释它为何不能是 formal。
