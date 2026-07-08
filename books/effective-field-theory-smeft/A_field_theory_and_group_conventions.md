# 附录 A：场论与群论约定

## A.1 度规与 gamma 矩阵

本书采用
$$
\eta_{\mu\nu}=\mathrm{diag}(+1,-1,-1,-1),
$$
并令
$$
\{\gamma^\mu,\gamma^\nu\}=2\eta^{\mu\nu}.
$$

## A.2 协变导数

对标准模型场 $\Phi$，
$$
D_\mu
=
\partial_\mu
-ig_s T^A G_\mu^A
-ig t^I W_\mu^I
-ig'Y B_\mu,
$$
其中只保留该场所在表示中非平凡的生成元。对 $SU(2)_L$ 基本表示，$t^I=\tau^I/2$；本书在算符表中出现的 $\tau^I$ 指 Pauli 矩阵本身。

## A.3 场强

$$
G_{\mu\nu}^A
=
\partial_\mu G_\nu^A-\partial_\nu G_\mu^A
+g_s f^{ABC}G_\mu^B G_\nu^C,
$$
$$
W_{\mu\nu}^I
=
\partial_\mu W_\nu^I-\partial_\nu W_\mu^I
+g\epsilon^{IJK}W_\mu^J W_\nu^K,
$$
$$
B_{\mu\nu}=\partial_\mu B_\nu-\partial_\nu B_\mu.
$$

## A.4 维数表

| 对象 | 维数 |
|---|---:|
| 标量 $H,\phi$ | 1 |
| 规范场 $A_\mu$ | 1 |
| 导数 $\partial_\mu$ | 1 |
| Weyl/Dirac 费米子 $\psi$ | $3/2$ |
| 场强 $X_{\mu\nu}$ | 2 |
| 拉氏量 $\mathcal L$ | 4 |
| 作用量 $S$ | 0 |
