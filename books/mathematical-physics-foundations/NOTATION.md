# 符号与约定

## 逻辑状态

| 标记 | 含义 |
|---|---|
| `P` | 相对于明示先修知识，正文给出覆盖全部结论的完整证明 |
| `S` | 带正规化、截断、微扰阶数或能区边界的标准物理形式推导 |
| `E` | 精确陈述、登记用途并给出可追溯定位的外部输入定理 |

## 几何

| 符号 | 含义 |
|---|---|
| $M,N$ | 光滑流形 |
| $T_pM,T^*_pM$ | 切空间与余切空间 |
| $TM,T^*M$ | 切丛与余切丛 |
| $\mathfrak X(M)$ | 光滑向量场 |
| $\Omega^k(M)$ | 光滑 $k$-形式 |
| $d$ | 外微分 |
| $\iota_X$ | 向量场 $X$ 的内乘 |
| $\mathcal L_X$ | Lie 导数 |
| $g$ | Riemann 或 pseudo-Riemann 度量 |
| $\nabla$ | 联络或协变导数 |
| $R^\nabla$ | 曲率 |
| $\omega$ | 辛形式 |
| $X_f$ | Hamilton 向量场，$\iota_{X_f}\omega=df$ |
| $\{f,h\}$ | Poisson 括号，$\omega(X_f,X_h)$ |

## Lie 理论与表示

| 符号 | 含义 |
|---|---|
| $G$ | Lie 群 |
| $\mathfrak g$ | Lie 代数 |
| $\exp:\mathfrak g\to G$ | 指数映射 |
| $(\rho,V)$ | $G$ 或 $\mathfrak g$ 的表示 |
| $V_\lambda$ | 权 $\lambda$ 的权空间 |
| $\widehat G$ | 紧群不可约酉表示等价类 |
| $\operatorname{Ad},\operatorname{ad}$ | 伴随表示及其微分 |
| $C_2$ | 二次 Casimir 元或其表示中的本征值 |

## 分析与量子论

| 符号 | 含义 |
|---|---|
| $\mathcal H$ | Hilbert 空间 |
| $\mathcal D(A)$ | 无界算符 $A$ 的定义域 |
| $A^*$ | Hilbert 空间伴随 |
| $\sigma(A)$ | 谱 |
| $E_A(\Delta)$ | 自伴算符 $A$ 的谱投影 |
| $\mathcal S(\mathbb R^n)$ | Schwartz 空间 |
| $\mathcal S'(\mathbb R^n)$ | tempered distributions |
| $[\hat q,\hat p]=i$ | 自然单位下正则对易关系 |
| $U(g)$ | 对称群元素 $g$ 的酉实现 |

## 场论

| 符号 | 含义 |
|---|---|
| $X$ | 时空流形 |
| $\eta_{\mu\nu}$ | Minkowski 度量，默认 mostly plus |
| $\phi$ | 标量场或场构型 |
| $\mathcal L$ | Lagrange 密度 |
| $S[\phi]$ | 作用量 |
| $T_{\mu\nu}$ | 能动张量 |
| $A$ | 主丛联络局部一形式 |
| $F_A=dA+\frac12[A\wedge A]$ | 曲率 |
| $D_A$ | 规范协变导数 |
| $Z[J]$ | 带源生成泛函 |
| $\Gamma[\varphi]$ | 有效作用量 |
| $Q_{\rm BRST}$ | BRST 荷 |

## 全书默认约定

1. 所有有限维流形默认光滑、Hausdorff、第二可数。
2. Hilbert 内积默认对第二个变量线性。
3. 重复上下指标默认求和；若指标均在同一位置，则不默认求和，除非局部说明。
4. Fourier 变换采用
   $$
   \widehat f(k)=\int_{\mathbb R^n}e^{-ikx}f(x)\,dx,\qquad
   f(x)=\frac1{(2\pi)^n}\int_{\mathbb R^n}e^{ikx}\widehat f(k)\,dk.
   $$
5. 量子场论中默认自然单位 $\hbar=c=1$。
