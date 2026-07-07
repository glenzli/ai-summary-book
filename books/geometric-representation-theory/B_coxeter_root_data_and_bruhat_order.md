# 附录 B：Coxeter groups、root data 与 Bruhat order

## 本章目标

本附录固定 Coxeter groups、root data、length、Bruhat order 和 dot action 的基础约定。

## B.1 Coxeter systems

**定义 B.1.** Coxeter system 是群 $W$ 和生成集 $S$，满足表示
$$
W=\langle s\in S\mid (st)^{m_{st}}=1\rangle,
$$
其中 $m_{ss}=1$，$m_{st}=m_{ts}\in\{2,3,\ldots,\infty\}$。

**定义 B.2.** 长度函数 $\ell(w)$ 是把 $w$ 写为 $S$ 中元素乘积所需的最小长度。若 $\ell(ws)>\ell(w)$，称 $s$ 是 $w$ 的右 ascent；否则为右 descent。

**定义 B.3.** Bruhat order 是由 subword property 刻画的偏序：若 $w=s_1\cdots s_r$ 为 reduced expression，则 $v\le w$ 当且仅当 $v$ 可由某个子表达式相乘得到。

**外部输入定理 B.4.** Bruhat order 与 Schubert closure order 一致：
$$
\overline X_w=\coprod_{v\le w}X_v.
$$

**外部输入定理 B.5.** Coxeter group 满足 exchange condition：若
$$
w=s_1\cdots s_r
$$
是 reduced expression 且 $\ell(ws)<\ell(w)$，则存在可删除位置 $i$ 使得
$$
ws=s_1\cdots\widehat{s_i}\cdots s_r.
$$
该定理是 Bruhat order、standard object 卷积和 Bott-Samelson 分解的组合基础。

**命题 B.6.** 若 $\ell(ws)=\ell(w)+1$，则 $w<ws$ 且区间 $[w,ws]$ 只含两个元素。

**证明.** 由 reduced expression $w=s_1\cdots s_r$ 得到 $ws=s_1\cdots s_rs$ 为 reduced expression。subword property 给出 $w\le ws$。Bruhat order 以长度函数分层：若 $x<y$，则 $\ell(x)<\ell(y)$。因此任意 $w\le v\le ws$ 只能满足 $\ell(v)=r$ 或 $\ell(v)=r+1$。第一种情形由 $w\le v$ 且长度相同推出 $v=w$；第二种情形由 $v\le ws$ 且长度相同推出 $v=ws$。$\square$

**例 B.7.** 对 type $A_2$，$W=S_3=\langle s_1,s_2\mid s_1^2=s_2^2=1,\ s_1s_2s_1=s_2s_1s_2\rangle$。Bruhat order 的长度层为
$$
e;\qquad s_1,s_2;\qquad s_1s_2,s_2s_1;\qquad w_0=s_1s_2s_1.
$$
两个长度 $2$ 元素都小于 $w_0$，但互不可比。

## B.2 Root datum 和 dot action

**定义 B.8.** Root datum 为
$$
(X,\Phi,X^\vee,\Phi^\vee)
$$
连同完美配对 $X\times X^\vee\to\mathbb Z$，满足根和 coroot 的反射公理。

**例 B.9.** 对 $G=GL_n$，
$$
X^\ast(T)\simeq\mathbb Z^n,\qquad X_\ast(T)\simeq\mathbb Z^n,
$$
根为 $e_i-e_j$，coroot 仍为 $e_i^\vee-e_j^\vee$。对 $G=SL_n$，character lattice 要取满足总和为零的子格，coweight lattice 与 simply-connected/adjoint form 的选择有关。

**定义 B.10.** 对 $\lambda\in\mathfrak t^\ast$，dot action 定义为
$$
w\cdot\lambda=w(\lambda+\rho)-\rho.
$$

**命题 B.11.** 若 $\lambda+\rho$ 在 reflection hyperplane 上，即存在根 $\alpha$ 使得
$$
\langle\lambda+\rho,\alpha^\vee\rangle=0,
$$
则 $s_\alpha\cdot\lambda=\lambda$。

**证明.** 反射公式给出
$$
s_\alpha(\lambda+\rho)=\lambda+\rho-\langle\lambda+\rho,\alpha^\vee\rangle\alpha.
$$
括号项为零，所以 $s_\alpha(\lambda+\rho)=\lambda+\rho$。减去 $\rho$ 得 $s_\alpha\cdot\lambda=\lambda$。$\square$

**命题 B.12.** 若 $\lambda$ integral regular，则 $W\cdot\lambda$ 的 stabilizer 平凡。

**证明.** 若 $w\cdot\lambda=\lambda$，则 $w(\lambda+\rho)=\lambda+\rho$。regularity 表示 $\lambda+\rho$ 不在任一 reflection hyperplane 上。有限 Weyl group 中非平凡元素的固定空间包含在若干 reflection hyperplanes 的并中；因此 $w$ 必为单位元。$\square$

## 本章小结

本附录固定 Coxeter convention、Bruhat order、root datum 和 dot action，并给出最常用的低阶检查。Schubert 几何中的闭包定理、exchange condition 等深层性质作为外部输入登记。
