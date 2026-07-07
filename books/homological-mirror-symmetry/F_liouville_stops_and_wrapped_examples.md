# 附录 F：Liouville sectors、stops 与 wrapped examples

## F.1 Cotangent bundle

**例 F.1.** $T^\ast Q$ 是 Liouville manifold，Liouville form 为
$$
\lambda=\sum_i p_i\,dq_i.
$$
Cotangent fiber $T_q^\ast Q$ 是 admissible exact Lagrangian。

**外部输入例 F.2.** Wrapped Floer cohomology $HW^\ast(T_q^\ast Q,T_q^\ast Q)$ 与 based loop space chains 存在深刻关系。具体版本依赖 Abouzaid 等工作。

## F.2 Weinstein handles

**定义 F.3.** Weinstein manifold 是带 Morse Lyapunov function 的 Liouville manifold，其 critical handles 的 cocores 是 exact Lagrangian disks。

**外部输入定理 F.4.** 在合适 Weinstein sector 假设下，critical cocores split-generate wrapped Fukaya category。

## F.3 Stop 示例

**例 F.5.** 在二维 Liouville sector 中，一个无穷远 Legendrian 点可作为 stop。其 linking disk 是靠近该点的小 Lagrangian arc。

**命题 F.6.** 若移除该 stop，则 linking disk 在 stop-removed category 中变成零对象。

**证明.** 由 stop removal equals localization，移除 stop 等于 quotient by linking disk 生成的子范畴。quotient functor 把该子范畴对象送零。证毕。

## F.4 Wrapped HMS 使用模式

Wrapped examples 的证明通常按以下步骤：

1. 找 Weinstein skeleton；
2. 找 cocores 或 linking disks；
3. 用 generation theorem 证明生成；
4. 计算 endomorphism algebra；
5. 与 B-side tilting/exceptional/singularity 生成对象比较。

## 本附录小结

Liouville、Weinstein、stops 和 cocores 提供 wrapped HMS 的可计算对象。Stop removal 和 cocore generation 把几何 handle decomposition 转化为范畴生成与局部化。
