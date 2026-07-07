# 附录 H：Kazhdan-Lusztig 多项式和 Soergel bimodule 计算

## 本章目标

本附录记录 KL 和 Soergel 的低阶计算模板。

## H.1 Hecke algebra 归一化

**约定 H.1.** 本书使用
$$
(T_s-v)(T_s+v^{-1})=0.
$$
KL basis 的精确归一化在 locator 阶段锁定。

**例 H.2.** 若 $W=\mathbb Z/2=\{e,s\}$，则
$$
C_e=1,\qquad C_s=T_s+v^{-1}
$$
在常见归一化下成立。

**命题 H.3.** 在上述归一化下，
$$
T_s^2=(v-v^{-1})T_s+1.
$$

**证明.** 展开二次关系：
$$
(T_s-v)(T_s+v^{-1})=T_s^2+(v^{-1}-v)T_s-1=0.
$$
移项即得结论。$\square$

**计算 H.4.** 对 $W=\mathbb Z/2$，
$$
C_s^2=(v+v^{-1})C_s.
$$
事实上
$$
(T_s+v^{-1})^2
=T_s^2+2v^{-1}T_s+v^{-2}
=((v-v^{-1})+2v^{-1})T_s+1+v^{-2}
=(v+v^{-1})(T_s+v^{-1}).
$$
这与几何上 $\operatorname{IC}_s\star\operatorname{IC}_s$ 的分解系数相匹配，但具体 shift 依赖第四章的 convention。

## H.2 Soergel 的 $A_1$ 计算

**例 H.5.** 对 $W=\mathbb Z/2$ 作用在一维 $\mathfrak h$ 上，$R=\mathbb R[\alpha]$，$\deg\alpha=2$，$R^s=\mathbb R[\alpha^2]$，
$$
B_s=R\otimes_{R^s}R(1).
$$

**命题 H.6.** $B_s$ 作为左 $R$-module 是自由秩 $2$。

**证明.** $R$ 作为 $R^s$-module 有基 $\{1,\alpha\}$。因此
$$
B_s=R\otimes_{R^s}R(1)
$$
作为左 $R$-module 同构于
$$
R\otimes_{R^s}(R^s\oplus R^s\alpha)(1)\simeq R(1)\oplus R(-1)
$$
其中最后的 shift 依赖 grading convention。$\square$

**命题 H.7.** 在 split Grothendieck group 中，$[B_s]^2=(v+v^{-1})[B_s]$，与 H.4 的 Hecke 计算相同。

**证明.** 作为 Bott-Samelson bimodule，
$$
B_s\otimes_R B_s
\simeq R\otimes_{R^s}R\otimes_{R^s}R(2).
$$
利用 $R\simeq R^s\oplus R^s\alpha$ 分解中间 $R$，得到
$$
B_s\otimes_R B_s\simeq B_s(1)\oplus B_s(-1)
$$
在标准 grading convention 下成立。取 split Grothendieck group 后得到
$$
[B_s]^2=(v+v^{-1})[B_s].
$$
$\square$

## H.3 Type $A_2$ 的第一批 KL 数据

**计算 H.8.** 对 $W=S_3$，长度小于等于 $2$ 的元素所对应 Schubert varieties 光滑，故相应 KL 多项式均为 $1$。在本书 convention 下，这意味着
$$
P_{x,w}=1\qquad (x\le w,\ \ell(w)\le2).
$$
最长元 $w_0$ 对应完整 flag variety 的顶层闭包；在 type $A_2$ 中仍不出现非平凡 KL 多项式。

**警告 H.9.** 非平凡 KL 多项式通常需要更高 rank 或更复杂 interval。低秩 $A_1$、$A_2$ 只能检查归一化、长度、shift 和 Hecke 二次关系，不能测试所有奇异性现象。

## 本章小结

本附录为 KL/Soergel 计算提供低阶模板：Hecke 二次关系、$A_1$ Soergel 分解和 type $A_2$ 的平凡 KL 数据。这些计算用于校对正文的 normalization。
