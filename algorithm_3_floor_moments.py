from math import gcd, isqrt

def floor_moments(n: int, m: int, a: int, b: int):
    if n == 0:
        return 0, 0, 0, 0, 0, 0

    if a >= m or b >= m:
        q, r = divmod(a, m)
        s, t = divmod(b, m)

        B01, B11, B21, B02, B12, B03 = floor_moments(n, m, r, t)

        s0 = n
        s1 = n * (n - 1) // 2
        s2 = n * (n - 1) * (2 * n - 1) // 6
        s3 = s1 * s1

        qq = q * q
        qqq = qq * q
        ss = s * s
        sss = ss * s
        qs2 = 2 * q * s

        A01 = q * s1 + s * s0 + B01
        A11 = q * s2 + s * s1 + B11
        A21 = q * s3 + s * s2 + B21

        A02 = qq * s2 + ss * s0 + B02 + qs2 * s1 + 2 * q * B11 + 2 * s * B01
        A12 = qq * s3 + ss * s1 + B12 + qs2 * s2 + 2 * q * B21 + 2 * s * B11

        A03 = (
            qqq * s3
            + 3 * qq * s * s2
            + 3 * q * ss * s1
            + sss * s0
            + 3 * qq * B21
            + 6 * q * s * B11
            + 3 * ss * B01
            + 3 * q * B12
            + 3 * s * B02
            + B03
        )
        return A01, A11, A21, A02, A12, A03

    y = (a * (n - 1) + b) // m
    if y == 0:
        return 0, 0, 0, 0, 0, 0

    bp = m + a - 1 - b
    B01, B11, B21, B02, B12, B03 = floor_moments(y, a, m, bp)

    t1 = n * (n - 1) // 2
    t2 = n * (n - 1) * (2 * n - 1) // 6
    yy = y * y

    A01 = n * y - B01
    A11 = y * t1 - (B02 - B01) // 2
    A21 = y * t2 - (2 * B03 - 3 * B02 + B01) // 6
    A02 = n * yy - 2 * B11 - B01
    A12 = yy * t1 - B12 - B02 // 2 + B11 + B01 // 2
    A03 = n * y * yy - 3 * B21 - 3 * B11 - B01

    return A01, A11, A21, A02, A12, A03


def count_rectangles_uv(n: int, u: int, v: int) -> int:
    if n <= 0 or u <= 0 or v <= 0 or gcd(u, v) != 1:
        return 0

    if u < v:
        u, v = v, u

    U, V = u, v
    UV = U * V
    U2pV2 = U * U + V * V
    fm = floor_moments
    N = n

    def segment_sum(lo: int, hi: int, a: int, b: int, m: int) -> int:
        if lo > hi:
            return 0

        L = hi - lo + 1
        A01, A11, A21, A02, A12, A03 = fm(L, m, a, b)

        hi2 = hi * hi
        sum_t = A01
        sum_kt = hi * A01 - A11
        sum_k2t = hi2 * A01 - 2 * hi * A11 + A21
        sum_t2 = A02
        sum_kt2 = hi * A02 - A12
        sum_t3 = A03

        num = (
            6 * UV * sum_k2t
            + 3 * U2pV2 * sum_kt2
            + (-6 * N * (U + V) + 3 * U2pV2 - 6 * (U + V)) * sum_kt
            + 2 * UV * sum_t3
            + (-3 * N * (U + V) + 3 * UV - 3 * (U + V)) * sum_t2
            + (6 * N * N - 3 * N * (U + V) + 12 * N + UV - 3 * (U + V) + 6) * sum_t
        )
        return num // 6

    m1 = N // (U + V)
    m2 = N // U

    ans = 0
    if m1:
        ans += segment_sum(1, m1, V, N - V * m1, U)
    if m2 > m1:
        ans += segment_sum(m1 + 1, m2, U, N - U * m2, V)

    return ans

def small_part(n, B):
    s = count_rectangles_uv(n, 1, 1)
    for u in range(2, B + 1):
        for v in range(1, u):
            if gcd(u, v) == 1:
                s += 2*count_rectangles_uv(n, u, v)
    return s



### big part

def build_divs_mu(n):
    spf = list(range(n + 1))
    for i in range(2, int(n ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i

    out = [[] for _ in range(n + 1)]
    if n >= 1:
        out[1] = [(1, 1)]

    for x in range(2, n + 1):
        y = x
        ps = []
        while y > 1:
            p = spf[y]
            ps.append(p)
            while y % p == 0:
                y //= p

        cur = [(1, 1)]
        for p in ps:
            base_len = len(cur)
            for i in range(base_len):
                dd, mu = cur[i]
                cur.append((dd * p, -mu))

        out[x] = cur

    return out


def coprime_interval_sums(divs_mu_r, L, R):
    if R < L:
        return 0, 0, 0

    def pref(M):
        if M <= 0:
            return 0, 0, 0
        cnt = s1 = s2 = 0
        for d, mu in divs_mu_r:
            t = M // d
            cnt += mu * t
            s1 += mu * d * t * (t + 1) // 2
            s2 += mu * d * d * t * (t + 1) * (2 * t + 1) // 6
        return cnt, s1, s2

    cR, sR1, sR2 = pref(R)
    cL, sL1, sL2 = pref(L - 1)
    return cR - cL, sR1 - sL1, sR2 - sL2


def fast_large_part_by_r(n, B, divs_mu):
    total = 0

    for r in range(B + 1, n):
        k = (n - 1) // r
        for a in range(1, k + 1):
            for b in range(1, a + 1):
                mult = 1 if a == b else 2
                s_max = min(r, (n - 1 - a * r) // b)
                cnt, s1, s2 = coprime_interval_sums(divs_mu[r], 1, s_max)
                n1 = n - a * r
                n2 = n - b * r
                total += mult * (
                    n1 * n2 * cnt
                    - (a * n1 + b * n2) * s1
                    + a * b * s2
                )

    return total


### result
def rect_fastestest(n):
    B = int(n ** (2 / 3)/2)
    divs_mu = build_divs_mu(n)
    S2 = fast_large_part_by_r(n, B, divs_mu)
    S1 = small_part(n-1, B)
    S3 = n*(n-1)*n*(n-1)//4
    return S3 + 2*S2 + S1