# 内部证明核 II：低阶计算

本文件提供正式教材中不可省略的低阶计算模板。

## L.1 $\mathfrak{sl}_2$ Verma module

取基 $e,f,h$，满足
$$
[h,e]=2e,\qquad [h,f]=-2f,\qquad [e,f]=h.
$$
令 $M(\lambda)$ 由最高权向量 $v_\lambda$ 生成：
$$
ev_\lambda=0,\qquad hv_\lambda=\lambda v_\lambda.
$$

**命题 L.1.** $M(\lambda)$ 有基 $\{f^n v_\lambda\}_{n\ge0}$，且
$$
h f^n v_\lambda=(\lambda-2n)f^n v_\lambda,
$$
$$
e f^n v_\lambda=n(\lambda-n+1)f^{n-1}v_\lambda.
$$

**证明.** PBW 给出 $M(\lambda)\simeq U(\mathbb C f)$，故 $\{f^n v_\lambda\}$ 为基。第一式由
$$
hf^n=f^n h+[h,f^n]=f^n h-2n f^n
$$
得到。第二式用归纳：
$$
e f^n=f^n e+[e,f^n].
$$
由 $[e,f]=h$ 和 $[h,f]=-2f$ 可得
$$
[e,f^n]=n f^{n-1}h-n(n-1)f^{n-1}.
$$
作用到 $v_\lambda$ 上即得
$$
e f^n v_\lambda=n(\lambda-n+1)f^{n-1}v_\lambda.
$$
$\square$

**推论 L.2.** 若 $\lambda\in\mathbb Z_{\ge0}$，则 $f^{\lambda+1}v_\lambda$ 是 singular vector，生成 proper submodule。

**证明.** 在 L.1 的公式中取 $n=\lambda+1$，系数
$(\lambda+1)(\lambda-(\lambda+1)+1)$ 为零，故
$e f^{\lambda+1}v_\lambda=0$。该向量权为
$\lambda-2(\lambda+1)=-\lambda-2$，与最高权向量权不同；它生成的
最高权子模不含 $v_\lambda$，所以是 proper submodule。$\square$

## L.2 $SL_2$ Springer fiber

**命题 L.3.** 对 $G=SL_2$，非零 nilpotent $x$ 的 Springer fiber 是一点。

**证明.** $SL_2/B$ 参数化 $\mathbb C^2$ 中的直线。nilpotent $x$ 属于 Borel $\mathfrak b_L$ 的 nilradical 当且仅当 $x(L)=0$ 且 $\operatorname{im}x\subset L$。对非零 nilpotent，$\operatorname{im}x=\ker x$ 是唯一一条直线。因此满足条件的 $L$ 唯一。$\square$

## L.3 $A_1$ Hecke 和 Soergel

令 $W=\{e,s\}$。Hecke algebra 满足
$$
(T_s-v)(T_s+v^{-1})=0.
$$
定义
$$
C_s=T_s+v^{-1}.
$$
则 bar involution 下 $\overline v=v^{-1}$，$\overline{T_s}=T_s^{-1}=T_s-(v-v^{-1})$，因此
$$
\overline{C_s}=T_s-(v-v^{-1})+v=T_s+v^{-1}=C_s.
$$
这验证 $A_1$ 情形 KL 基的 bar-invariance。

## L.4 $GL_1$ geometric Satake

对 $G=GL_1$，
$$
LG/L^+G=\mathbb C((z))^\times/\mathbb C[[z]]^\times\simeq\mathbb Z.
$$
每个连通分支是一点。Satake category 等价于 $\mathbb Z$-graded finite-dimensional vector spaces，卷积对应 grading 相加。Langlands dual group 仍为 $GL_1$，其表示按 characters $\mathbb Z$ 分解。这给出 geometric Satake 的最小检验例。
