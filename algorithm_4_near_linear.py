import math


POWER = [[0] for _ in range(5)]


def ensure_power(n):
    P0 = POWER[0]
    cur = len(P0) - 1
    if n <= cur:
        return

    need = n - cur
    for row in POWER:
        row.extend([0] * need)

    P0, P1, P2, P3, P4 = POWER
    for x in range(cur + 1, n + 1):
        xm1 = x - 1
        s = x * xm1 // 2
        t = x * xm1 * (2 * x - 1)

        P0[x] = x
        P1[x] = s
        P2[x] = t // 6
        P3[x] = s * s
        P4[x] = t * (3 * x * x - 3 * x - 1) // 30


def squarefree_divisors_only(M):
    spf = [0] * (M + 1)
    primes = []

    for i in range(2, M + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        si = spf[i]
        for p in primes:
            v = i * p
            if v > M:
                break
            spf[v] = p
            if p == si:
                break

    divs = [()] * (M + 1)
    divs[1] = ((1, 1),)
    one = ((1, 1),)

    for x in range(2, M + 1):
        p = spf[x]
        y = x
        while y % p == 0:
            y //= p

        base = divs[y] if y != 1 else one
        m = len(base)
        cur = [None] * (m << 1)

        for i in range(m):
            d, sgn = base[i]
            cur[i] = (d, sgn)
            cur[i + m] = (d * p, -sgn)

        cur.sort(key=lambda z: z[0])
        divs[x] = tuple(cur)

    return divs


def _affine_combine(H, n, A, B):
    h01, h11, h21, h31, h02, h12, h22, h03, h13, h04 = H

    P0, P1, P2, P3, P4 = POWER
    p0 = P0[n]
    p1 = P1[n]
    p2 = P2[n]
    p3 = P3[n]
    p4 = P4[n]

    A2 = A * A
    A3 = A2 * A
    A4 = A3 * A

    if B == 0:
        return (
            h01 + A * p1,
            h11 + A * p2,
            h21 + A * p3,
            h31 + A * p4,
            h02 + 2 * A * h11 + A2 * p2,
            h12 + 2 * A * h21 + A2 * p3,
            h22 + 2 * A * h31 + A2 * p4,
            h03 + 3 * A * h12 + 3 * A2 * h21 + A3 * p3,
            h13 + 3 * A * h22 + 3 * A2 * h31 + A3 * p4,
            h04 + 4 * A * h13 + 6 * A2 * h22 + 4 * A3 * h31 + A4 * p4,
        )

    B2 = B * B
    B3 = B2 * B
    B4 = B3 * B

    AB = A * B
    A2B = A2 * B
    AB2 = A * B2

    return (
        h01 + A * p1 + B * p0,
        h11 + A * p2 + B * p1,
        h21 + A * p3 + B * p2,
        h31 + A * p4 + B * p3,
        h02 + 2 * A * h11 + 2 * B * h01 + A2 * p2 + 2 * AB * p1 + B2 * p0,
        h12 + 2 * A * h21 + 2 * B * h11 + A2 * p3 + 2 * AB * p2 + B2 * p1,
        h22 + 2 * A * h31 + 2 * B * h21 + A2 * p4 + 2 * AB * p3 + B2 * p2,
        h03
        + 3 * A * h12
        + 3 * B * h02
        + 3 * A2 * h21
        + 6 * AB * h11
        + 3 * B2 * h01
        + A3 * p3
        + 3 * A2B * p2
        + 3 * AB2 * p1
        + B3 * p0,
        h13
        + 3 * A * h22
        + 3 * B * h12
        + 3 * A2 * h31
        + 6 * AB * h21
        + 3 * B2 * h11
        + A3 * p4
        + 3 * A2B * p3
        + 3 * AB2 * p2
        + B3 * p1,
        h04
        + 4 * A * h13
        + 4 * B * h03
        + 6 * A2 * h22
        + 12 * AB * h12
        + 6 * B2 * h02
        + 4 * A3 * h31
        + 12 * A2B * h21
        + 12 * AB2 * h11
        + 4 * B3 * h01
        + A4 * p4
        + 4 * A3 * B * p3
        + 6 * A2 * B2 * p2
        + 4 * A * B3 * p1
        + B4 * p0,
    )


def _recip_combine(G, n, Y):
    g01, g11, g21, g31, g02, g12, g22, g03, g13, g04 = G

    P0, P1, P2, P3 = POWER[:4]
    p0 = P0[n]
    p1 = P1[n]
    p2 = P2[n]
    p3 = P3[n]

    q0 = P0[Y]
    q1 = P1[Y]
    q2 = P2[Y]
    q3 = P3[Y]

    Y2 = Y * Y
    Y3 = Y2 * Y
    Y4 = Y3 * Y

    return (
        p0 * Y - q0 - g01,
        (2 * p1 * Y - g01 - g02) // 2,
        (6 * p2 * Y - g01 - 3 * g02 - 2 * g03) // 6,
        (4 * p3 * Y - g02 - 2 * g03 - g04) // 4,
        p0 * Y2 - q0 - g01 - 2 * q1 - 2 * g11,
        (2 * p1 * Y2 - g01 - g02 - 2 * g11 - 2 * g12) // 2,
        (6 * p2 * Y2 - g01 - 3 * g02 - 2 * g03 - 2 * g11 - 6 * g12 - 4 * g13) // 6,
        p0 * Y3 - q0 - g01 - 3 * q1 - 3 * g11 - 3 * q2 - 3 * g21,
        (2 * p1 * Y3 - g01 - g02 - 3 * g11 - 3 * g12 - 3 * g21 - 3 * g22) // 2,
        p0 * Y4 - q0 - g01 - 4 * q1 - 4 * g11 - 6 * q2 - 6 * g21 - 4 * q3 - 4 * g31,
    )


def solve_moments(n, m, a, b):
    if n == 0:
        return (0,) * 10

    if a == 0:
        c = b // m
        c2 = c * c
        c3 = c2 * c
        c4 = c3 * c
        P0, P1, P2, P3, _ = POWER
        return (
            P0[n] * c,
            P1[n] * c,
            P2[n] * c,
            P3[n] * c,
            P0[n] * c2,
            P1[n] * c2,
            P2[n] * c2,
            P0[n] * c3,
            P1[n] * c3,
            P0[n] * c4,
        )

    if a >= m or b >= m:
        A, a2 = divmod(a, m)
        B, b2 = divmod(b, m)
        return _affine_combine(solve_moments(n, m, a2, b2), n, A, B)

    Y = (a * (n - 1) + b) // m
    if Y == 0:
        return (0,) * 10

    return _recip_combine(solve_moments(Y, a, m, m - b - 1), n, Y)


def rect_const_cap_sum(M, u, y, d, x0, x1, T):
    if x0 > x1 or T <= 0:
        return 0

    cnt = x1 - x0 + 1
    sx = (x0 + x1) * cnt // 2

    sx2 = (
        x1 * (x1 + 1) * (2 * x1 + 1) // 6
        - ((x0 - 1) * x0 * (2 * x0 - 1) // 6 if x0 else 0)
    )

    st = T * (T + 1) // 2
    st2 = T * (T + 1) * (2 * T + 1) // 6

    Mp1 = M + 1
    a = Mp1 - u * y
    dd = d * d

    return (
        Mp1 * a * cnt * T
        - u * a * sx * T
        - d * y * a * cnt * st
        - d * Mp1 * sx * st
        + d * u * sx2 * st
        + dd * y * sx * st2
    )


def linear_region_sum_from_x0(M, u, y, d, x0):
    B = y * d
    alpha = M - u * x0 - B
    if alpha < 0:
        return 0

    S, rem = divmod(alpha, u)
    h01, h11, h21, h31, h02, h12, h22, h03, h13, _ = solve_moments(S + 1, B, u, rem)

    P0, P1, P2 = POWER[:3]
    Sp1 = S + 1
    h00 = P0[Sp1]
    h10 = P1[Sp1]
    h20 = P2[Sp1]

    S2 = S * S

    U10 = S * h00 - h10
    U11 = S * h01 - h11
    U12 = S * h02 - h12
    U13 = S * h03 - h13

    U20 = S2 * h00 - 2 * S * h10 + h20
    U21 = S2 * h01 - 2 * S * h11 + h21
    U22 = S2 * h02 - 2 * S * h12 + h22

    V01 = h01 + h00
    V02 = h02 + 2 * h01 + h00
    V03 = h03 + 3 * h02 + 3 * h01 + h00

    V11 = U11 + U10
    V12 = U12 + 2 * U11 + U10
    V13 = U13 + 3 * U12 + 3 * U11 + U10

    V21 = U21 + U20
    V22 = U22 + 2 * U21 + U20

    Mp1 = M + 1
    a = Mp1 - u * x0
    b = Mp1 - u * y

    c0 = a * b
    cr = -u * b
    ct = d * (u * x0 * x0 + u * y * y - Mp1 * (x0 + y))
    crt = d * (2 * u * x0 - M - 1)
    cr2t = d * u

    dd = d * d
    ctt = dd * x0 * y
    crtt = dd * y

    return (
        6 * c0 * V01
        + 6 * cr * V11
        + 3 * ct * (V02 + V01)
        + 3 * crt * (V12 + V11)
        + 3 * cr2t * (V22 + V21)
        + ctt * (2 * V03 + 3 * V02 + V01)
        + crtt * (2 * V13 + 3 * V12 + V11)
    ) // 6


def triangle_polynomial_sum(M, u, y, d):
    T = (u - 1) // d
    if T <= 0:
        return 0

    q = M // u
    x_cap = q - y

    total = 0
    if x_cap >= y:
        total += rect_const_cap_sum(M, u, y, d, y, x_cap, T)
        x0 = x_cap + 1
    else:
        x0 = y

    total += linear_region_sum_from_x0(M, u, y, d, x0)

    M0 = (M - u * y) // (y * d)
    if M0 > T:
        M0 = T

    if M0 > 0:
        a = M + 1 - u * y
        s1 = M0 * (M0 + 1) // 2
        s2 = M0 * (M0 + 1) * (2 * M0 + 1) // 6
        diag = a * a * M0 - 2 * d * y * a * s1 + d * d * y * y * s2
    else:
        diag = 0

    return 4 * total - 2 * diag


def count_oblique_rectangles(npts):
    if npts <= 1:
        return 0

    M = npts - 1
    ensure_power(M + 2)
    divs = squarefree_divisors_only(M)

    ans = 0
    tps = triangle_polynomial_sum

    for u in range(2, M + 1):
        ymax = M // u
        du = divs[u]
        for y in range(1, ymax + 1):
            dlim = M // y - u
            sub = 0
            for d, mu in du:
                if d >= u or d > dlim:
                    break
                sub += mu * tps(M, u, y, d)
            ans += sub

    return ans


def count_all_rectangles(n):
    n1 = n - 1
    return n * n1 * n1 * (2 * n - 1) // 6 + count_oblique_rectangles(n)