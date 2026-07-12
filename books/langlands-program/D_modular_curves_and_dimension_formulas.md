# 附录 D：模曲线和维数公式

收口归一化回指：本附录支撑 classical modular form normalization、权二微分、Hecke 算子和级 `2` 矛盾；与 adelic 表示、Galois 表示和费马应用比较时使用 [NORMALIZATION_TABLE.md](NORMALIZATION_TABLE.md) 第 5、6、8 节。

## D.1 模曲线

**定义 D.1.** 对同余子群 $\Gamma\subset\operatorname{SL}_2(\mathbb Z)$，模曲线的复点为紧化商
$$
X(\Gamma)=\Gamma\backslash(\mathfrak H\cup\mathbb P^1(\mathbb Q)).
$$

**定义 D.2.** $X_0(N)$ 对应 $\Gamma_0(N)$，$X_1(N)$ 对应 $\Gamma_1(N)$。

**外部输入定理 D.3（模曲线代数化与微分形式）.** $X(\Gamma)$ 是紧 Riemann surface，并有代数曲线模型。权 $2$ cusp forms 与 holomorphic differentials 对应：
$$
S_2(\Gamma)\cong H^0(X(\Gamma),\Omega^1).
$$

## D.2 Genus 与权 2 cusp forms

**命题 D.4.** 有
$$
\dim S_2(\Gamma)=g(X(\Gamma)).
$$

**证明.** 由定理 D.3，$S_2(\Gamma)$ 同构于紧 Riemann surface 上 holomorphic differentials 空间。该空间维数等于 genus。$\square$

**外部输入定理 D.5（genus formula for $X_0(N)$）.** 设
$$
\mu=[\operatorname{SL}_2(\mathbb Z):\Gamma_0(N)],
$$
$e_2,e_3$ 为椭圆点数，$c$ 为 cusp 数，则
$$
g(X_0(N))=1+\frac{\mu}{12}-\frac{e_2}{4}-\frac{e_3}{3}-\frac{c}{2}.
$$

## D.3 级 2 的计算

**命题 D.6.**
$$
S_2(\Gamma_0(2))=0.
$$

**证明.** 对 $N=2$，附录 D.5 给出 genus formula。下列数值在 D.10-D.12 中计算：
$$
\mu=[\operatorname{SL}_2(\mathbb Z):\Gamma_0(2)]=3.
$$
该曲线有两个 cusp，故 $c=2$。椭圆点贡献为 $e_2=1$、$e_3=0$。代入 genus formula：
$$
g(X_0(2))=1+\frac{3}{12}-\frac14-0-\frac{2}{2}=0.
$$
由命题 D.4，
$$
\dim S_2(\Gamma_0(2))=0.
$$
$\square$

**注 D.7.** 费马应用章只需要命题 D.6。完整维数公式还涉及一般权 $k$、nebentypus、old/new decomposition 和 Atkin-Lehner 理论。

## D.4 Newforms

**外部输入定理 D.8（Atkin-Lehner-Li newform theory）.** 空间 $S_k(\Gamma_0(N),\varepsilon)$ 分解为 old subspace 与 new subspace；newforms 可归一化为 Hecke eigenforms，并对应导子为 $N$ 的 automorphic representations of `GL(2)`.

**注 D.9.** 第七至十章使用 newform theory 作为经典模形式与 adelic 自守表示之间的桥梁。

## D.5 级 2 Genus 计算细节

**命题 D.10.** 有
$$
[\operatorname{SL}_2(\mathbb Z):\Gamma_0(2)]=3.
$$

**证明.** 群 $\operatorname{SL}_2(\mathbb Z)$ 作用在 $\mathbb P^1(\mathbb Z/2\mathbb Z)$ 上。该集合有三个点：
$$
[1:0],\quad [0:1],\quad [1:1].
$$
稳定点 $[1:0]$ 的元素正是下三角项满足 $c\equiv0\pmod2$ 的矩阵，即 $\Gamma_0(2)$。约化映射
$$
\operatorname{SL}_2(\mathbb Z)\to\operatorname{SL}_2(\mathbb Z/2\mathbb Z)
$$
满射，且 $\operatorname{SL}_2(\mathbb Z/2\mathbb Z)$ 对 $\mathbb P^1(\mathbb F_2)$ 传递。因此陪集集合与 $\mathbb P^1(\mathbb F_2)$ 等势，指数为 $3$。$\square$

**命题 D.11.** $X_0(2)$ 的 cusp 数为 $2$。

**证明.** 对 $\Gamma_0(N)$，cusp 可由 $\Gamma_0(N)$ 在 $\mathbb P^1(\mathbb Q)$ 上的轨道表示。$N=2$ 时，两个代表可取
$$
\infty=[1:0],\qquad 0=[0:1].
$$
取互素整数 $a,c$ 表示 cusp $a/c$。若 $c$ 为偶数，由 Bezout 等式可取 $b,d\in\mathbb Z$ 使
$$
ad-bc=1.
$$
则
$$
\begin{pmatrix}a&b\\ c&d\end{pmatrix}\in\Gamma_0(2)
$$
且把 $\infty$ 送到 $a/c$。

若 $c$ 为奇数，需要构造 $\Gamma_0(2)$ 中矩阵把 $0$ 送到 $a/c$。求偶数 $\beta$ 使
$$
a\beta\equiv -1\pmod c.
$$
由于 $(a,c)=1$ 且 $c$ 为奇数，先取任意解，再用 $\beta\mapsto\beta+tc$ 并选择 $t$ 的奇偶性来调整 $\beta$ 的奇偶性。令
$$
\alpha=\frac{1+a\beta}{c}.
$$
则
$$
\begin{pmatrix}\alpha&a\\ \beta&c\end{pmatrix}\in\Gamma_0(2)
$$
且该矩阵把 $0$ 送到 $a/c$。

最后说明两个代表不等价。若 $\gamma=\begin{pmatrix}r&s\\2t&u\end{pmatrix}\in\Gamma_0(2)$，则
$$
\gamma(a/c)=\frac{ra+sc}{2ta+uc}.
$$
当 $c$ 偶且 $(a,c)=1$ 时，$a$ 为奇数，分母 $2ta+uc$ 为偶数，分子 $ra+sc$ 为奇数；约分后分母仍为偶数。当 $c$ 奇时，分母 $2ta+uc$ 为奇数，约分后分母仍为奇数。因此分母奇偶性是 $\Gamma_0(2)$-轨道不变量。故 cusp 数为 $2$。$\square$

**命题 D.12.** 对 $X_0(2)$，椭圆点数满足
$$
e_2=1,\qquad e_3=0.
$$

**证明路线（外部输入）.** 对 $\Gamma_0(N)$，椭圆点计数可化为下列同余方程的解数：
$$
e_2=\#\{x\in\mathbb Z/N\mathbb Z:x^2+1\equiv0\pmod N\},
$$
$$
e_3=\#\{x\in\mathbb Z/N\mathbb Z:x^2+x+1\equiv0\pmod N\},
$$
其中该公式来自把阶 $2$ 和阶 $3$ 椭圆稳定子的固定点写成二次型的根，并按 $\Gamma_0(N)$-等价类取模。

当 $N=2$ 时，
$$
x^2+1\equiv0\pmod2
$$
有唯一解 $x\equiv1$，故 $e_2=1$。另一方面
$$
x^2+x+1\equiv1\pmod2
$$
对 $x=0,1$ 均成立，因此无解，故 $e_3=0$。$\square$

**推论 D.13.** $X_0(2)$ 的 genus 为 $0$。

**证明.** 将 D.10、D.11 和 D.12 的数值代入 D.5：
$$
g(X_0(2))=1+\frac{3}{12}-\frac14-\frac03-\frac22=0.
$$
$\square$

## D.6 权 2 形式和微分形式的局部计算

**命题 D.14.** 若 $f\in S_2(\Gamma)$，则
$$
\omega_f=f(z)\,dz
$$
在 $Y(\Gamma)=\Gamma\backslash\mathfrak H$ 上下降为 holomorphic differential，并在 cusp 处全纯延拓。

**证明路线（外部输入）.** 权 $2$ 变换公式为
$$
f(\gamma z)(cz+d)^{-2}=f(z).
$$
又
$$
d(\gamma z)=\frac{dz}{(cz+d)^2}
$$
对 $\gamma\in\operatorname{SL}_2(\mathbb Z)$ 成立，因此 $f(\gamma z)d(\gamma z)=f(z)dz$。这说明 $\omega_f$ 在商上良定义。若 $q=e^{2\pi iz/h}$ 为 cusp 处局部参数，尖点条件给出
$$
f(z)=\sum_{n\ge1}a_nq^n.
$$
并且
$$
dz=\frac{h}{2\pi i}\frac{dq}{q},
$$
故
$$
\omega_f=\frac{h}{2\pi i}\sum_{n\ge1}a_nq^{n-1}dq
$$
在 $q=0$ 处全纯。$\square$

**命题 D.15.** 若 $X(\Gamma)$ genus 为 $0$，则 $S_2(\Gamma)=0$。

**证明.** 由 D.3 和 D.4，
$$
S_2(\Gamma)\simeq H^0(X(\Gamma),\Omega^1)
$$
且右侧维数等于 genus。若 genus 为 $0$，该空间维数为 $0$。$\square$
