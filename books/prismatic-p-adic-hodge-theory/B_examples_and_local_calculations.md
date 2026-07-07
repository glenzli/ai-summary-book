# 附录 B：基本 prism 例子与局部计算

## 本附录目标

本附录对正文中出现的基本 prism 例子进行局部检查。当前版本只完成可直接验证的部分；涉及 perfectoid equivalence、Breuil-Kisin distinguished 条件和 $q$-crystalline comparison 的深层部分标为外部输入。

## B.1 Crystalline prism

**命题 B.1.** 令 $A$ 为 $p$-torsionfree、$p$-complete 的 $\delta$-环。若 $p$ 是 nonzerodivisor，则 $(A,(p))$ 是 prism。

**证明.** $(p)$ 由 nonzerodivisor 生成，因此定义 Cartier divisor。$A$ 的 $p$-完备性给出 derived $(p)$-complete，在 noetherian/无 torsion 条件下与普通完备相容；一般 derived statement 需按附录 A 解释。Prism 条件第四项为
$$
p\in(p)+\phi((p))A,
$$
它由 $p\in(p)$ 直接成立。证毕。

**例 B.2.** 对完美域 $k$，$(W(k),(p))$ 是 crystalline prism。其 quotient 为 $k$。

## B.2 Perfect prism

**外部输入定理 B.3.** 若 $R$ 是 perfectoid ring，则
$$
(A_{\inf}(R),\ker\theta)
$$
是 perfect prism，且 perfect prisms 与 perfectoid rings 等价。

**可检查部分 B.4.** 在选择 $\xi$ 生成 $\ker\theta$ 后，oriented 记号为 $(A_{\inf},\xi)$。Prism 条件中的 quotient 为
$$
A_{\inf}/(\xi)\cong R.
$$
若 $R=\mathcal O_C$，这就是 BMS $A_{\inf}$-cohomology 的 base prism。

**警告 B.5.** $\ker\theta$ principal 和 $\xi$ distinguished 是 perfectoid theory 的深层性质，不能只由 Witt vector 定义形式推出。

## B.3 Breuil-Kisin prism

**命题 B.6.** 令 $K/\mathbf Q_p$ 为 complete discretely valued field，剩余域 $k$ 完美，uniformizer 为 $\pi$。设
$$
\mathfrak S=W(k)[[u]],\qquad \phi(u)=u^p,
$$
且 $E(u)$ 为 $\pi$ 的 Eisenstein polynomial。则 $E(u)$ 是 $\mathfrak S$ 中的 nonzerodivisor，且
$$
\mathfrak S/(E(u))\cong\mathcal O_K.
$$

**证明.** $E(u)$ 是首一多项式，作为形式幂级数环中的元素不是零因子。映射 $W(k)[[u]]\to\mathcal O_K$ 由 $u\mapsto\pi$ 给出，核由 $\pi$ 的 Eisenstein polynomial 生成，因此得到商同构。证毕。

**外部输入定理 B.7.** $E(u)$ 在 $\mathfrak S$ 的标准 $\delta$-结构下为 distinguished，因此 $(\mathfrak S,(E(u)))$ 是 prism。

## B.4 $q$-crystalline prism

**定义 B.8.** 令
$$
A=\mathbf Z_p[[q-1]],\qquad \phi(q)=q^p,\qquad [p]_q=\frac{q^p-1}{q-1}.
$$

**命题 B.9.** $[p]_q$ 在 $A$ 中是 nonzerodivisor，且
$$
A/([p]_q)
$$
是 $q$-crystalline specialization 的 quotient ring。

**证明.** $A$ 是整环，$[p]_q$ 非零，故不是零因子。商环的解释来自定义。证毕。

**外部输入定理 B.10.** $(\mathbf Z_p[[q-1]],([p]_q))$ 是 prism。该 prism 的 prismatic cohomology 与 $q$-de Rham theory 相接。

## B.5 四个基本例子的比较

| Prism | 底环 $A$ | 理想 $I$ | quotient $A/I$ | 主要 specialization |
| --- | --- | --- | --- | --- |
| crystalline | $p$-complete $\delta$-ring | $(p)$ | $A/p$ | crystalline |
| perfect | $A_{\inf}(R)$ | $\ker\theta$ | $R$ | $A_{\inf}$, etale |
| Breuil-Kisin | $W(k)[[u]]$ | $(E(u))$ | $\mathcal O_K$ | Breuil-Kisin |
| $q$-crystalline | $\mathbf Z_p[[q-1]]$ | $([p]_q)$ | $A/([p]_q)$ | $q$-de Rham |

**警告 B.11.** 这张表只比较输入 prism，不表示四个 cohomology theories 在无条件下相同。它们通过 base change、comparison theorem 和 specialization 连接。

## B.6 展开计算

**例 B.12（$q$-整数的二阶展开）.** 令 $q=1+t$。则
$$
[p]_q=\frac{(1+t)^p-1}{t}
=p+\binom p2t+\binom p3t^2+\cdots+t^{p-1}.
$$
因此常数项为 $p$。这说明 $[p]_q$ 是 $p$ 的 $q$-变形；当 $q\to1$ 时，$[p]_q$ 退化为 $p$。

**例 B.13（Breuil-Kisin topology 的比较）.** 在 $\mathfrak S=W(k)[[u]]$ 中，若 $E(u)$ 是 Eisenstein polynomial，则
$$
E(u)=u^e+p\cdot a(u)
$$
其中 $a(u)\in W(k)[[u]]$ 且常数项为单位。于是 $E(u)\in(p,u^e)$，并且 $u^e\in(E(u),p)$。这给出 $(p,E(u))$-adic topology 与 $(p,u)$-adic topology 紧密相关的基本原因。

**说明 B.14.** 例 B.13 不替代 Breuil-Kisin prism 的 distinguished 条件。它只说明 quotient 和完备拓扑的局部可计算部分；$\delta$-结构与 Frobenius 横截条件仍属于外部输入或后续 prismatic proof。

## 本附录小结

基本 prism 例子都围绕同一形式：$\delta$-环 $A$、Cartier divisor ideal $I$、quotient $A/I$ 和 Frobenius 横截条件。可直接检查的是 nonzerodivisor、quotient 和完备性；distinguished 性、perfectoid equivalence 和 comparison theorem 属于外部输入。

## 练习

**练习 B.1.** 对 $A=W(k)$，验证 $(A,(p))$ 的 boundedness。

**练习 B.2.** 在 Breuil-Kisin 情形中说明 $(p,E(u))$-adic topology 与 $(p,u)$-adic topology 为什么相关。

**练习 B.3.** 令 $q=1+t$，展开 $[p]_q$ 到 $t^2$ 项，并说明其常数项为 $p$。
