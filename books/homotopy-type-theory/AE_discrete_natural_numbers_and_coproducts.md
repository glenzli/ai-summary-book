# 附录 AE：自然数与和类型的离散性证明核

本附录补全附录 M.1 中自然数集合性，以及 M.5 中和类型集合性的证明。核心方法都是 encode-decode：为路径类型定义一个计算型 code 族，证明路径类型与 code 等价，再利用 code 是命题推出路径空间是命题。

## AE.1 自然数 no-confusion

**定义 AE.1（自然数路径 code）.** 定义
$$
\mathsf{Code}_{\mathbb N}:\mathbb N\to\mathbb N\to\mathcal U
$$
为双递归：
$$
\mathsf{Code}_{\mathbb N}(0,0)\coloneqq\mathbf 1,
$$
$$
\mathsf{Code}_{\mathbb N}(0,\mathsf{succ}(n))\coloneqq\mathbf 0,
$$
$$
\mathsf{Code}_{\mathbb N}(\mathsf{succ}(m),0)\coloneqq\mathbf 0,
$$
$$
\mathsf{Code}_{\mathbb N}(\mathsf{succ}(m),\mathsf{succ}(n))
\coloneqq
\mathsf{Code}_{\mathbb N}(m,n).
$$
定义
$$
\mathsf{r}_m:\mathsf{Code}_{\mathbb N}(m,m)
$$
为对 $m$ 归纳得到的规范点：$m=0$ 时为 $\star$，successor 情形沿递归定义沿用归纳点。

**定义 AE.2（自然数 encode/decode）.** 对 $m,n:\mathbb N$，定义
$$
\mathsf{encode}_{m,n}:(m=n)\to\mathsf{Code}_{\mathbb N}(m,n)
$$
为
$$
\mathsf{encode}_{m,n}(p)
\coloneqq
\mathsf{transport}^{\lambda k.\mathsf{Code}_{\mathbb N}(m,k)}(p,\mathsf{r}_m).
$$
定义
$$
\mathsf{decode}_{m,n}:\mathsf{Code}_{\mathbb N}(m,n)\to(m=n)
$$
按 $m,n$ 双归纳：

1.  $(0,0)$ 情形送 $\star$ 到 $\mathsf{refl}_0$；
2.  $(0,\mathsf{succ}(n))$ 与 $(\mathsf{succ}(m),0)$ 情形由空类型消去；
3.  $(\mathsf{succ}(m),\mathsf{succ}(n))$ 情形把
    $q:\mathsf{Code}_{\mathbb N}(m,n)$ 送到
    $$
    \mathsf{ap}_{\mathsf{succ}}(\mathsf{decode}_{m,n}(q)).
    $$

**定理 AE.3（自然数路径等价 code）.** 对任意 $m,n:\mathbb N$，
$$
(m=n)\simeq\mathsf{Code}_{\mathbb N}(m,n).
$$

**证明.** 两个方向为 AE.2。证明
$$
\mathsf{decode}(\mathsf{encode}(p))=p
$$
时，对路径 $p$ 作路径归纳，反身情形化为 $\mathsf{decode}(\mathsf{r}_m)=\mathsf{refl}_m$，再对 $m$ 归纳。

证明
$$
\mathsf{encode}(\mathsf{decode}(c))=c
$$
时，对 $m,n$ 双归纳。两个混合零/successor 情形由空类型消去；$(0,0)$ 情形为单位类型中路径；双 successor 情形归约为归纳假设，并使用 transport 与 $\mathsf{ap}_{\mathsf{succ}}$ 的计算相容。$\square$

**推论 AE.4（自然数是集合）.** $\mathbb N$ 是集合。

**证明.** 固定 $m,n:\mathbb N$。由 AE.3，路径类型 $m=n$ 等价于
$\mathsf{Code}_{\mathbb N}(m,n)$。后者按定义递归化为 $\mathbf 0$ 或 $\mathbf 1$，因此是命题。等价保持命题性，故 $m=n$ 是命题。于是 $\mathbb N$ 是集合。$\square$

## AE.2 和类型 no-confusion

设 $A,B:\mathcal U$。

**定义 AE.5（和类型路径 code）.** 对 $u,v:A+B$，定义
$$
\mathsf{Code}_{+}(u,v):\mathcal U
$$
按 $u,v$ 分情形：
$$
\mathsf{Code}_{+}(\mathsf{inl}(a),\mathsf{inl}(a'))
\coloneqq
(a=a'),
$$
$$
\mathsf{Code}_{+}(\mathsf{inr}(b),\mathsf{inr}(b'))
\coloneqq
(b=b'),
$$
$$
\mathsf{Code}_{+}(\mathsf{inl}(a),\mathsf{inr}(b))
\coloneqq\mathbf 0,
$$
$$
\mathsf{Code}_{+}(\mathsf{inr}(b),\mathsf{inl}(a))
\coloneqq\mathbf 0.
$$
对每个 $u:A+B$，有规范点
$$
\mathsf{r}_u:\mathsf{Code}_{+}(u,u)
$$
由 $u$ 分情形取反身路径。

**定义 AE.6（和类型 encode/decode）.** 定义
$$
\mathsf{encode}_{u,v}:(u=v)\to\mathsf{Code}_{+}(u,v)
$$
为沿路径 transport $\mathsf{r}_u$。反向
$$
\mathsf{decode}_{u,v}:\mathsf{Code}_{+}(u,v)\to(u=v)
$$
按 $u,v$ 分情形：

1.  左左情形 $q:a=a'$ 送到 $\mathsf{ap}_{\mathsf{inl}}(q)$；
2.  右右情形 $q:b=b'$ 送到 $\mathsf{ap}_{\mathsf{inr}}(q)$；
3.  混合情形由空类型消去。

**定理 AE.7（和类型 no-confusion）.** 对任意 $u,v:A+B$，
$$
(u=v)\simeq\mathsf{Code}_{+}(u,v).
$$

**证明.** 两个方向为 AE.6。证明两个复合为恒等：对路径复合方向，对 $p:u=v$ 作路径归纳，反身情形按 $u$ 分情形化为反身路径；对 code 方向，按 $u,v$ 分情形，左左和右右情形再对路径 $q$ 作路径归纳，混合情形由空类型消去。$\square$

**推论 AE.8（和类型保持集合性）.** 若 $A$ 与 $B$ 是集合，则 $A+B$ 是集合。

**证明.** 固定 $u,v:A+B$。由 AE.7，路径类型 $u=v$ 等价于 $\mathsf{Code}_{+}(u,v)$。若同侧，则 code 是 $A$ 或 $B$ 的路径类型，因 $A,B$ 是集合而为命题；若异侧，则 code 是 $\mathbf 0$，也是命题。等价保持命题性，故 $u=v$ 是命题。于是 $A+B$ 是集合。$\square$

**推论 AE.9（归纳整数是集合）.** $\mathbb Z_{\mathsf{ind}}\equiv\mathbb N+\mathbb N$ 是集合。

**证明.** 由 AE.4，$\mathbb N$ 是集合。由 AE.8，$\mathbb N+\mathbb N$ 是集合。$\square$
