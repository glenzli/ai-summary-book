# 第二章：路径积分、匹配与 Wilson 系数

## 本章目标

本章定义匹配（matching）和 Wilson 系数。匹配是把 UV 理论在低能区的振幅或 Green 函数等同于 EFT 的对应量，从而确定 Wilson 系数。

## 依赖前置知识

需要第一章的局域展开和基本路径积分记号。

## 2.1 积掉重场

设 UV 理论含轻场 $\phi$ 和重场 $\Phi$：
$$
Z[J]=\int D\phi\,D\Phi\,
\exp\left(iS_{\mathrm{UV}}[\phi,\Phi]+i\int J\phi\right).
$$

**定义 2.1（Wilsonian 有效作用量）.** 形式上定义
$$
\exp(iS_{\mathrm{eff}}[\phi])
\coloneqq
\int D\Phi\,\exp(iS_{\mathrm{UV}}[\phi,\Phi]).
$$
将 $S_{\mathrm{eff}}$ 在低能区展开为局域项，即得 EFT 作用量。

**警告 2.2.** $S_{\mathrm{eff}}$ 一般含非局域项。只有在低于重阈值并按 $E/\Lambda$ 展开后，才转化为局域算符级数。

## 2.2 树级匹配例子

**例 2.3（重标量诱导四轻场算符）.** 令
$$
\mathcal L_{\mathrm{UV}}
=
\frac12(\partial\phi)^2-\frac12m^2\phi^2
+\frac12(\partial\Phi)^2-\frac12M^2\Phi^2
-\frac{g}{2}\Phi\phi^2.
$$
在 $E\ll M$ 时，重场方程为
$$
(\Box+M^2)\Phi=-\frac{g}{2}\phi^2.
$$
形式求解得
$$
\Phi
=
-\frac{g}{2}\frac{1}{M^2+\Box}\phi^2
=
-\frac{g}{2M^2}\left(1-\frac{\Box}{M^2}+\cdots\right)\phi^2.
$$
代回拉氏量得到
$$
\Delta\mathcal L_{\mathrm{EFT}}
=
\frac{g^2}{8M^2}\phi^4
-\frac{g^2}{8M^4}\phi^2\Box\phi^2+\cdots,
$$
其中总导数按约定丢弃。

**推导状态.** 这是树级经典匹配。圈级匹配还需指定正规化和重整化方案。

## 2.3 振幅层面的同一例子

对例 2.3 的 $\phi\phi\to\phi\phi$ 散射，重场交换给出 $s,t,u$ 三个道。若顶点取为 $-ig$，则
$$
i{\cal A}_{\rm UV}
=(-ig)^2\left[
{i\over s-M^2}
+{i\over t-M^2}
+{i\over u-M^2}
\right].
$$
因此
$$
{\cal A}_{\rm UV}
=-g^2\left[
{1\over s-M^2}
+{1\over t-M^2}
+{1\over u-M^2}
\right].
$$
在 $|s|,|t|,|u|\ll M^2$ 下，
$$
{\cal A}_{\rm UV}
={g^2\over M^2}
\left(3+{s+t+u\over M^2}
 +{s^2+t^2+u^2\over M^4}
 +\cdots\right).
$$
对相同质量外线，$s+t+u=4m^2$。若在无质量近似下取 $m=0$，则第一阶导数修正从 $s^2+t^2+u^2$ 开始。

另一方面，EFT 中
$$
\Delta{\cal L}_{\rm EFT}\supset {g^2\over8M^2}\phi^4
$$
给出四点接触顶点
$$
i{\cal A}_{\rm EFT}^{(0)}
=i\,4!\,{g^2\over8M^2}
=i{3g^2\over M^2},
$$
正好对应三个交换道的低能常数项。导数算符
$$
-{g^2\over8M^4}\phi^2\Box\phi^2
$$
给出下一阶动量依赖项。

**警告 2.4（归一化）.** 若把 $\phi^4$ 写成 $-\lambda\phi^4/4!$ 或把源写成 $J=\phi^2/2$，顶点因子会重新分配。匹配时必须固定拉氏量归一化，不能只比较符号和量纲。

## 2.4 匹配条件

**定义 2.5（振幅匹配）.** 固定外部轻态和低能动量，将 UV 振幅展开为 $E/\Lambda$ 级数，并要求
$$
\mathcal A_{\mathrm{UV}}
=
\mathcal A_{\mathrm{EFT}}
$$
到指定阶数成立。由此确定 Wilson 系数。

**定义 2.6（Green 函数匹配）.** 固定规范、正规化方案和外部场归一化，使 UV 与 EFT 的重整化 Green 函数在低能展开后相等。

**命题 2.7（匹配系数依赖方案）.** Wilson 系数的数值依赖算符基、场重定义、正规化方案和匹配尺度；物理可观测量不依赖这些中间选择。

**证明说明.** 算符基变换对应 Wilson 系数坐标变换；重整化方案改变会被矩阵元的相反改变抵消。完整证明需引入重整化群和 S-matrix 等价定理，见第三、四章。$\square$

## 2.5 匹配尺度

匹配通常在 $\mu\simeq M$ 处做。若一圈匹配给出
$$
C_i(\mu)=a_i+b_i\log{M^2\over\mu^2},
$$
则选择 $\mu=M$ 可避免大对数。若实验尺度 $\mu_{\rm exp}\ll M$，大对数应由 RGE 从 $M$ 运行到 $\mu_{\rm exp}$ 得到，而不是留在固定阶匹配系数中。

**原则 2.8（阈值与运行分工）.** 匹配负责短程阈值常数，运行负责跨尺度对数。把二者混在一起会导致重复计数或遗漏对数。

## 本章小结

匹配把高能理论的信息压缩到 Wilson 系数。EFT 的局域算符给出低能展开的结构，Wilson 系数给出 UV 物理的数值内容。

## 练习

**练习 2.1.** 对例 2.3 计算 $\phi\phi\to\phi\phi$ 的树级振幅，并与 EFT 的 $\phi^4$ 顶点比较。

**练习 2.2.** 说明 on-shell 振幅匹配为什么看不到某些 EOM 冗余算符。

**练习 2.3.** 用本章的顶点归一化，验证 $\phi^4$ 接触项给出的常数振幅为 $3g^2/M^2$。
