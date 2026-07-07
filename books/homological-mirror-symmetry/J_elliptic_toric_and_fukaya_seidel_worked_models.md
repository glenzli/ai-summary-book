# 附录 J：椭圆曲线、toric Fano 与 Fukaya-Seidel 计算模型

## J.1 二维环面斜率圆

**计算 J.1.** 令 $L_{r,d}$、$L_{r',d'}$ 为 $T^2$ 上 primitive 斜率圆。其有向交数为
$$
I(L_{r,d},L_{r',d'})=rd'-r'd.
$$
若位置横截且无重合分量，则几何交点数为 $|rd'-r'd|$。

**例 J.2.** $L_{2,1}$ 与 $L_{3,5}$ 的交点数为
$$
|2\cdot5-3\cdot1|=7.
$$

**计算 J.3.** 若 stable bundles $E_{r,d}$ 与 $E_{r',d'}$ 对应这些斜率，则 Riemann-Roch 给出 Euler characteristic
$$
\chi(E_{r,d},E_{r',d'})=r d'-r'd
$$
到本书采用的左右 Hom 约定符号。该式匹配有向交数。

## J.2 $\mathbb P^1$ 的 Jacobian ring

**计算 J.4.** 对
$$
W(z)=z+qz^{-1}
$$
有
$$
z\frac{dW}{dz}=z-qz^{-1}.
$$
所以
$$
\operatorname{Jac}(W)=k[z^{\pm1}]/(z-qz^{-1})
\cong k[z]/(z^2-q).
$$

**计算 J.5.** Critical points 满足 $z^2=q$。若 $k=\mathbb C$ 且 $q\ne0$，则有两个 critical points $z=\pm q^{1/2}$，critical values 为
$$
W(\pm q^{1/2})=\pm2q^{1/2}.
$$

## J.3 $\mathbb P^n$ 的 potential

**计算 J.6.** 对
$$
W=z_1+\cdots+z_n+q(z_1\cdots z_n)^{-1},
$$
critical equations 是
$$
z_i\frac{\partial W}{\partial z_i}=z_i-q(z_1\cdots z_n)^{-1}=0.
$$
因此所有 $z_i$ 相等，设为 $z$，则
$$
z=qz^{-n},\qquad z^{n+1}=q.
$$

**解释 J.7.** 这给出 $n+1$ 个 critical points，与 semisimple quantum cohomology of $\mathbb P^n$ 的 $n+1$ 个幂等分量相匹配。

## J.4 $\mathbb P^1$ 的 directed algebra

**计算 J.8.** B-side exceptional collection $(\mathcal O,\mathcal O(1))$ 的 endomorphism algebra 满足
$$
\operatorname{Hom}(\mathcal O,\mathcal O)=k,\qquad
\operatorname{Hom}(\mathcal O(1),\mathcal O(1))=k,
$$
$$
\operatorname{Hom}(\mathcal O(1),\mathcal O)=0,\qquad
\operatorname{Hom}(\mathcal O,\mathcal O(1))=H^0(\mathbb P^1,\mathcal O(1))\cong k^2.
$$
因此 directed algebra 是 Kronecker quiver 的 path algebra，含两个从第一个顶点到第二个顶点的箭头。

**解释 J.9.** A-side $W=z+qz^{-1}$ 的 Fukaya-Seidel category 有两个 thimbles；它们之间的 morphism space 维数为 $2$，与 Kronecker quiver 的两个箭头匹配。完整 HMS 还需要乘法和高阶结构比较。

## J.5 Pair-of-pants 局部模型

**计算 J.10.** 一维 pair of pants 可写作
$$
P^1=\{(x,y)\in(\mathbb C^\ast)^2\mid x+y=1\}.
$$
投影到 $x$ 坐标给出
$$
P^1\simeq\mathbb C\setminus\{0,1\},
$$
即三点去除球面。

## 本附录小结

这些计算是标准 HMS 例子的最低可检验层：交点数、Euler characteristic、Jacobian ring、critical values、exceptional collection endomorphism algebra 和 pair-of-pants 局部模型。它们不构成完整 HMS 证明，但能排除错误字典并支撑生成元比较。
