#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using i64 = long long;
using i128 = __int128_t;

static std::array<std::vector<i128>, 5> POWER = {
    std::vector<i128>(1, 0),
    std::vector<i128>(1, 0),
    std::vector<i128>(1, 0),
    std::vector<i128>(1, 0),
    std::vector<i128>(1, 0),
};

using Moments = std::array<i128, 10>;

static std::string to_string_i128(i128 x) {
    if (x == 0) return "0";
    bool neg = (x < 0);
    if (neg) x = -x;

    std::string s;
    while (x > 0) {
        int digit = static_cast<int>(x % 10);
        s.push_back(static_cast<char>('0' + digit));
        x /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

static void ensure_power(i64 n) {
    auto &P0 = POWER[0];
    i64 cur = static_cast<i64>(P0.size()) - 1;
    if (n <= cur) return;

    i64 need = n - cur;
    for (auto &row : POWER) row.resize(static_cast<size_t>(row.size() + need), 0);

    auto &Q0 = POWER[0];
    auto &Q1 = POWER[1];
    auto &Q2 = POWER[2];
    auto &Q3 = POWER[3];
    auto &Q4 = POWER[4];

    for (i64 x = cur + 1; x <= n; ++x) {
        i128 X = static_cast<i128>(x);
        i128 xm1 = X - 1;
        i128 s = X * xm1 / 2;
        i128 t = X * xm1 * (2 * X - 1);

        Q0[x] = X;
        Q1[x] = s;
        Q2[x] = t / 6;
        Q3[x] = s * s;
        Q4[x] = t * (3 * X * X - 3 * X - 1) / 30;
    }
}

static std::vector<std::vector<std::pair<i64, int>>> squarefree_divisors_only(i64 M) {
    std::vector<i64> spf(static_cast<size_t>(M + 1), 0);
    std::vector<i64> primes;
    primes.reserve(static_cast<size_t>(M / 10 + 10));

    for (i64 i = 2; i <= M; ++i) {
        if (spf[i] == 0) {
            spf[i] = i;
            primes.push_back(i);
        }
        i64 si = spf[i];
        for (i64 p : primes) {
            i64 v = i * p;
            if (v > M) break;
            spf[v] = p;
            if (p == si) break;
        }
    }

    std::vector<std::vector<std::pair<i64, int>>> divs(static_cast<size_t>(M + 1));
    divs[1] = {{1, 1}};
    static const std::vector<std::pair<i64, int>> one = {{1, 1}};

    for (i64 x = 2; x <= M; ++x) {
        i64 p = spf[x];
        i64 y = x;
        while (y % p == 0) y /= p;

        const auto &base = (y != 1 ? divs[y] : one);
        size_t m = base.size();
        std::vector<std::pair<i64, int>> cur;
        cur.resize(m << 1);

        for (size_t i = 0; i < m; ++i) {
            cur[i] = base[i];
            cur[i + m] = {base[i].first * p, -base[i].second};
        }

        std::sort(cur.begin(), cur.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        divs[x] = std::move(cur);
    }

    return divs;
}

static Moments affine_combine(const Moments &H, i64 n, i64 A, i64 B) {
    const i128 h01 = H[0], h11 = H[1], h21 = H[2], h31 = H[3], h02 = H[4],
               h12 = H[5], h22 = H[6], h03 = H[7], h13 = H[8], h04 = H[9];

    const auto &P0 = POWER[0];
    const auto &P1 = POWER[1];
    const auto &P2 = POWER[2];
    const auto &P3 = POWER[3];
    const auto &P4 = POWER[4];

    i128 p0 = P0[n], p1 = P1[n], p2 = P2[n], p3 = P3[n], p4 = P4[n];
    i128 a = static_cast<i128>(A), b = static_cast<i128>(B);
    i128 a2 = a * a, a3 = a2 * a, a4 = a3 * a;

    if (B == 0) {
        return {{
            h01 + a * p1,
            h11 + a * p2,
            h21 + a * p3,
            h31 + a * p4,
            h02 + 2 * a * h11 + a2 * p2,
            h12 + 2 * a * h21 + a2 * p3,
            h22 + 2 * a * h31 + a2 * p4,
            h03 + 3 * a * h12 + 3 * a2 * h21 + a3 * p3,
            h13 + 3 * a * h22 + 3 * a2 * h31 + a3 * p4,
            h04 + 4 * a * h13 + 6 * a2 * h22 + 4 * a3 * h31 + a4 * p4,
        }};
    }

    i128 b2 = b * b, b3 = b2 * b, b4 = b3 * b;
    i128 ab = a * b, a2b = a2 * b, ab2 = a * b2;

    return {{
        h01 + a * p1 + b * p0,
        h11 + a * p2 + b * p1,
        h21 + a * p3 + b * p2,
        h31 + a * p4 + b * p3,
        h02 + 2 * a * h11 + 2 * b * h01 + a2 * p2 + 2 * ab * p1 + b2 * p0,
        h12 + 2 * a * h21 + 2 * b * h11 + a2 * p3 + 2 * ab * p2 + b2 * p1,
        h22 + 2 * a * h31 + 2 * b * h21 + a2 * p4 + 2 * ab * p3 + b2 * p2,
        h03 + 3 * a * h12 + 3 * b * h02 + 3 * a2 * h21 + 6 * ab * h11 + 3 * b2 * h01
            + a3 * p3 + 3 * a2b * p2 + 3 * ab2 * p1 + b3 * p0,
        h13 + 3 * a * h22 + 3 * b * h12 + 3 * a2 * h31 + 6 * ab * h21 + 3 * b2 * h11
            + a3 * p4 + 3 * a2b * p3 + 3 * ab2 * p2 + b3 * p1,
        h04 + 4 * a * h13 + 4 * b * h03 + 6 * a2 * h22 + 12 * ab * h12 + 6 * b2 * h02
            + 4 * a3 * h31 + 12 * a2b * h21 + 12 * ab2 * h11 + 4 * b3 * h01
            + a4 * p4 + 4 * a3 * b * p3 + 6 * a2 * b2 * p2 + 4 * a * b3 * p1 + b4 * p0,
    }};
}

static Moments recip_combine(const Moments &G, i64 n, i64 Y) {
    const i128 g01 = G[0], g11 = G[1], g21 = G[2], g31 = G[3], g02 = G[4],
               g12 = G[5], g22 = G[6], g03 = G[7], g13 = G[8], g04 = G[9];

    const auto &P0 = POWER[0];
    const auto &P1 = POWER[1];
    const auto &P2 = POWER[2];
    const auto &P3 = POWER[3];

    i128 p0 = P0[n], p1 = P1[n], p2 = P2[n], p3 = P3[n];
    i128 q0 = P0[Y], q1 = P1[Y], q2 = P2[Y], q3 = P3[Y];
    i128 y = static_cast<i128>(Y), y2 = y * y, y3 = y2 * y, y4 = y3 * y;

    return {{
        p0 * y - q0 - g01,
        (2 * p1 * y - g01 - g02) / 2,
        (6 * p2 * y - g01 - 3 * g02 - 2 * g03) / 6,
        (4 * p3 * y - g02 - 2 * g03 - g04) / 4,
        p0 * y2 - q0 - g01 - 2 * q1 - 2 * g11,
        (2 * p1 * y2 - g01 - g02 - 2 * g11 - 2 * g12) / 2,
        (6 * p2 * y2 - g01 - 3 * g02 - 2 * g03 - 2 * g11 - 6 * g12 - 4 * g13) / 6,
        p0 * y3 - q0 - g01 - 3 * q1 - 3 * g11 - 3 * q2 - 3 * g21,
        (2 * p1 * y3 - g01 - g02 - 3 * g11 - 3 * g12 - 3 * g21 - 3 * g22) / 2,
        p0 * y4 - q0 - g01 - 4 * q1 - 4 * g11 - 6 * q2 - 6 * g21 - 4 * q3 - 4 * g31,
    }};
}

static Moments solve_moments(i64 n, i64 m, i64 a, i64 b) {
    if (n == 0) {
        return {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    }

    if (a == 0) {
        i128 c = static_cast<i128>(b / m);
        i128 c2 = c * c, c3 = c2 * c, c4 = c3 * c;
        const auto &P0 = POWER[0];
        const auto &P1 = POWER[1];
        const auto &P2 = POWER[2];
        const auto &P3 = POWER[3];

        return {{
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
        }};
    }

    if (a >= m || b >= m) {
        i64 A = a / m, a2 = a % m;
        i64 B = b / m, b2 = b % m;
        return affine_combine(solve_moments(n, m, a2, b2), n, A, B);
    }

    i64 Y = (a * (n - 1) + b) / m;
    if (Y == 0) {
        return {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    }

    return recip_combine(solve_moments(Y, a, m, m - b - 1), n, Y);
}

static i128 rect_const_cap_sum(i64 M, i64 u, i64 y, i64 d, i64 x0, i64 x1, i64 T) {
    if (x0 > x1 || T <= 0) return 0;

    i128 cnt = static_cast<i128>(x1 - x0 + 1);
    i128 sx = (static_cast<i128>(x0) + x1) * cnt / 2;

    i128 sx2 =
        static_cast<i128>(x1) * (x1 + 1) * (2 * static_cast<i128>(x1) + 1) / 6
        - (x0 ? static_cast<i128>(x0 - 1) * x0 * (2 * static_cast<i128>(x0) - 1) / 6 : 0);

    i128 t = static_cast<i128>(T);
    i128 st = t * (t + 1) / 2;
    i128 st2 = t * (t + 1) * (2 * t + 1) / 6;

    i128 Mp1 = static_cast<i128>(M) + 1;
    i128 a = Mp1 - static_cast<i128>(u) * y;
    i128 dd = static_cast<i128>(d) * d;

    return Mp1 * a * cnt * t
         - static_cast<i128>(u) * a * sx * t
         - static_cast<i128>(d) * y * a * cnt * st
         - static_cast<i128>(d) * Mp1 * sx * st
         + static_cast<i128>(d) * u * sx2 * st
         + dd * y * sx * st2;
}

static i128 linear_region_sum_from_x0(i64 M, i64 u, i64 y, i64 d, i64 x0) {
    i64 B = y * d;
    i64 alpha = M - u * x0 - B;
    if (alpha < 0) return 0;

    i64 S = alpha / u;
    i64 rem = alpha % u;

    Moments H = solve_moments(S + 1, B, u, rem);
    i128 h01 = H[0], h11 = H[1], h21 = H[2], h31 = H[3], h02 = H[4],
         h12 = H[5], h22 = H[6], h03 = H[7], h13 = H[8];

    const auto &P0 = POWER[0];
    const auto &P1 = POWER[1];
    const auto &P2 = POWER[2];

    i64 Sp1 = S + 1;
    i128 h00 = P0[Sp1];
    i128 h10 = P1[Sp1];
    i128 h20 = P2[Sp1];

    i128 s = static_cast<i128>(S);
    i128 s2 = s * s;

    i128 U10 = s * h00 - h10;
    i128 U11 = s * h01 - h11;
    i128 U12 = s * h02 - h12;
    i128 U13 = s * h03 - h13;

    i128 U20 = s2 * h00 - 2 * s * h10 + h20;
    i128 U21 = s2 * h01 - 2 * s * h11 + h21;
    i128 U22 = s2 * h02 - 2 * s * h12 + h22;

    i128 V01 = h01 + h00;
    i128 V02 = h02 + 2 * h01 + h00;
    i128 V03 = h03 + 3 * h02 + 3 * h01 + h00;

    i128 V11 = U11 + U10;
    i128 V12 = U12 + 2 * U11 + U10;
    i128 V13 = U13 + 3 * U12 + 3 * U11 + U10;

    i128 V21 = U21 + U20;
    i128 V22 = U22 + 2 * U21 + U20;

    i128 Mp1 = static_cast<i128>(M) + 1;
    i128 ix0 = static_cast<i128>(x0);
    i128 iu = static_cast<i128>(u);
    i128 iy = static_cast<i128>(y);
    i128 id = static_cast<i128>(d);

    i128 a = Mp1 - iu * ix0;
    i128 b = Mp1 - iu * iy;

    i128 c0 = a * b;
    i128 cr = -iu * b;
    i128 ct = id * (iu * ix0 * ix0 + iu * iy * iy - Mp1 * (ix0 + iy));
    i128 crt = id * (2 * iu * ix0 - M - 1);
    i128 cr2t = id * iu;

    i128 dd = id * id;
    i128 ctt = dd * ix0 * iy;
    i128 crtt = dd * iy;

    return (
        6 * c0 * V01
        + 6 * cr * V11
        + 3 * ct * (V02 + V01)
        + 3 * crt * (V12 + V11)
        + 3 * cr2t * (V22 + V21)
        + ctt * (2 * V03 + 3 * V02 + V01)
        + crtt * (2 * V13 + 3 * V12 + V11)
    ) / 6;
}

static i128 triangle_polynomial_sum(i64 M, i64 u, i64 y, i64 d) {
    i64 T = (u - 1) / d;
    if (T <= 0) return 0;

    i64 q = M / u;
    i64 x_cap = q - y;

    i128 total = 0;
    i64 x0;
    if (x_cap >= y) {
        total += rect_const_cap_sum(M, u, y, d, y, x_cap, T);
        x0 = x_cap + 1;
    } else {
        x0 = y;
    }

    total += linear_region_sum_from_x0(M, u, y, d, x0);

    i64 M0 = (M - u * y) / (y * d);
    if (M0 > T) M0 = T;

    i128 diag = 0;
    if (M0 > 0) {
        i128 a = static_cast<i128>(M) + 1 - static_cast<i128>(u) * y;
        i128 m0 = static_cast<i128>(M0);
        i128 s1 = m0 * (m0 + 1) / 2;
        i128 s2 = m0 * (m0 + 1) * (2 * m0 + 1) / 6;
        i128 id = static_cast<i128>(d);
        i128 iy = static_cast<i128>(y);
        diag = a * a * m0 - 2 * id * iy * a * s1 + id * id * iy * iy * s2;
    }

    return 4 * total - 2 * diag;
}

static i128 count_oblique_rectangles(i64 npts) {
    if (npts <= 1) return 0;

    i64 M = npts - 1;
    ensure_power(M + 2);
    auto divs = squarefree_divisors_only(M);

    i128 ans = 0;
    for (i64 u = 2; u <= M; ++u) {
        i64 ymax = M / u;
        const auto &du = divs[u];
        for (i64 y = 1; y <= ymax; ++y) {
            i64 dlim = M / y - u;
            i128 sub = 0;
            for (const auto &[d, mu] : du) {
                if (d >= u || d > dlim) break;
                sub += static_cast<i128>(mu) * triangle_polynomial_sum(M, u, y, d);
            }
            ans += sub;
        }
    }
    return ans;
}

static i128 count_all_rectangles(i64 n) {
    i128 nn = static_cast<i128>(n);
    i128 n1 = static_cast<i128>(n - 1);
    return nn * n1 * n1 * (2 * nn - 1) / 6 + count_oblique_rectangles(n);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " k\n";
        std::cerr << "benchmarks n = 2^1, 2^2, ..., 2^k\n";
        return 1;
    }

    int k = std::atoi(argv[1]);
    if (k < 1 || k > 62) {
        std::cerr << "k must be in [1, 62]\n";
        return 1;
    }

    for (int e = 1; e <= k; ++e) {
        i64 n = (1LL << e);

        auto t0 = std::chrono::steady_clock::now();
        i128 ans = count_all_rectangles(n);
        auto t1 = std::chrono::steady_clock::now();

        std::chrono::duration<double> dt = t1 - t0;

        std::cout << "n=" << n
                  << " result=" << to_string_i128(ans)
                  << " time=" << dt.count()
                  << "\n";
        std::cout.flush();
    }

    return 0;
}
