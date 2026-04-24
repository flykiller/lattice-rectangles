#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iomanip>

using i64 = long long;
using u64 = unsigned long long;
using i128 = __int128_t;
using u128 = __uint128_t;

struct Moments {
    i128 A01, A11, A21, A02, A12, A03;
};

static std::string to_string_i128(i128 x) {
    if (x == 0) return "0";
    bool neg = x < 0;
    u128 v = neg ? (u128)(-x) : (u128)x;
    std::string s;
    while (v > 0) {
        int digit = (int)(v % 10);
        s.push_back(char('0' + digit));
        v /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

static Moments floor_moments(i64 n, i64 m, i64 a, i64 b) {
    if (n == 0) {
        return {0, 0, 0, 0, 0, 0};
    }

    if (a >= m || b >= m) {
        i64 q = a / m, r = a % m;
        i64 s = b / m, t = b % m;

        Moments B = floor_moments(n, m, r, t);

        i128 s0 = n;
        i128 s1 = (i128)n * (n - 1) / 2;
        i128 s2 = (i128)n * (n - 1) * (2 * (i128)n - 1) / 6;
        i128 s3 = s1 * s1;

        i128 qq = (i128)q * q;
        i128 qqq = qq * q;
        i128 ss = (i128)s * s;
        i128 sss = ss * s;
        i128 qs2 = 2 * (i128)q * s;

        i128 A01 = (i128)q * s1 + (i128)s * s0 + B.A01;
        i128 A11 = (i128)q * s2 + (i128)s * s1 + B.A11;
        i128 A21 = (i128)q * s3 + (i128)s * s2 + B.A21;

        i128 A02 = qq * s2 + ss * s0 + B.A02 + qs2 * s1 + 2 * (i128)q * B.A11 + 2 * (i128)s * B.A01;
        i128 A12 = qq * s3 + ss * s1 + B.A12 + qs2 * s2 + 2 * (i128)q * B.A21 + 2 * (i128)s * B.A11;

        i128 A03 =
            qqq * s3
            + 3 * qq * s * s2
            + 3 * (i128)q * ss * s1
            + sss * s0
            + 3 * qq * B.A21
            + 6 * (i128)q * s * B.A11
            + 3 * ss * B.A01
            + 3 * (i128)q * B.A12
            + 3 * (i128)s * B.A02
            + B.A03;

        return {A01, A11, A21, A02, A12, A03};
    }

    i64 y = ((i128)a * (n - 1) + b) / m;
    if (y == 0) {
        return {0, 0, 0, 0, 0, 0};
    }

    i64 bp = m + a - 1 - b;
    Moments B = floor_moments(y, a, m, bp);

    i128 t1 = (i128)n * (n - 1) / 2;
    i128 t2 = (i128)n * (n - 1) * (2 * (i128)n - 1) / 6;
    i128 yy = (i128)y * y;

    i128 A01 = (i128)n * y - B.A01;
    i128 A11 = (i128)y * t1 - (B.A02 - B.A01) / 2;
    i128 A21 = (i128)y * t2 - (2 * B.A03 - 3 * B.A02 + B.A01) / 6;
    i128 A02 = (i128)n * yy - 2 * B.A11 - B.A01;
    i128 A12 = yy * t1 - B.A12 - B.A02 / 2 + B.A11 + B.A01 / 2;
    i128 A03 = (i128)n * y * yy - 3 * B.A21 - 3 * B.A11 - B.A01;

    return {A01, A11, A21, A02, A12, A03};
}

static i128 count_rectangles_uv(i64 n, i64 u, i64 v) {
    if (n <= 0 || u <= 0 || v <= 0 || std::gcd(u, v) != 1) {
        return 0;
    }

    if (u < v) std::swap(u, v);

    i64 U = u, V = v;
    i128 UV = (i128)U * V;
    i128 U2pV2 = (i128)U * U + (i128)V * V;
    i64 N = n;

    auto segment_sum = [&](i64 lo, i64 hi, i64 a, i64 b, i64 m) -> i128 {
        if (lo > hi) return 0;

        i64 L = hi - lo + 1;
        Moments M = floor_moments(L, m, a, b);

        i128 hi2 = (i128)hi * hi;
        i128 sum_t = M.A01;
        i128 sum_kt = (i128)hi * M.A01 - M.A11;
        i128 sum_k2t = hi2 * M.A01 - 2 * (i128)hi * M.A11 + M.A21;
        i128 sum_t2 = M.A02;
        i128 sum_kt2 = (i128)hi * M.A02 - M.A12;
        i128 sum_t3 = M.A03;

        i128 num =
            6 * UV * sum_k2t
            + 3 * U2pV2 * sum_kt2
            + ((-6 * (i128)N * (U + V)) + 3 * U2pV2 - 6 * (U + V)) * sum_kt
            + 2 * UV * sum_t3
            + ((-3 * (i128)N * (U + V)) + 3 * UV - 3 * (U + V)) * sum_t2
            + ((6 * (i128)N * N) - 3 * (i128)N * (U + V) + 12 * N + UV - 3 * (U + V) + 6) * sum_t;

        return num / 6;
    };

    i64 m1 = N / (U + V);
    i64 m2 = N / U;

    i128 ans = 0;
    if (m1) ans += segment_sum(1, m1, V, N - V * m1, U);
    if (m2 > m1) ans += segment_sum(m1 + 1, m2, U, N - U * m2, V);

    return ans;
}

static i128 small_part(i64 n, i64 B) {
    i128 s = count_rectangles_uv(n, 1, 1);
    for (i64 u = 2; u <= B; ++u) {
        for (i64 v = 1; v < u; ++v) {
            if (std::gcd(u, v) == 1) {
                s += 2 * count_rectangles_uv(n, u, v);
            }
        }
    }
    return s;
}

static std::vector<std::vector<std::pair<int, int>>> build_divs_mu(int n) {
    std::vector<int> spf(n + 1);
    for (int i = 0; i <= n; ++i) spf[i] = i;

    for (int i = 2; (i64)i * i <= n; ++i) {
        if (spf[i] == i) {
            for (int j = i * i; j <= n; j += i) {
                if (spf[j] == j) spf[j] = i;
            }
        }
    }

    std::vector<std::vector<std::pair<int, int>>> out(n + 1);
    if (n >= 1) out[1].push_back({1, 1});

    for (int x = 2; x <= n; ++x) {
        int y = x;
        std::vector<int> ps;
        while (y > 1) {
            int p = spf[y];
            ps.push_back(p);
            while (y % p == 0) y /= p;
        }

        std::vector<std::pair<int, int>> cur;
        cur.push_back({1, 1});

        for (int p : ps) {
            int base_len = (int)cur.size();
            for (int i = 0; i < base_len; ++i) {
                auto [dd, mu] = cur[i];
                cur.push_back({dd * p, -mu});
            }
        }

        out[x] = std::move(cur);
    }

    return out;
}

static std::tuple<i128, i128, i128>
coprime_interval_sums(const std::vector<std::pair<int, int>>& divs_mu_r, i64 L, i64 R) {
    if (R < L) return {0, 0, 0};

    auto pref = [&](i64 M) -> std::tuple<i128, i128, i128> {
        if (M <= 0) return {0, 0, 0};

        i128 cnt = 0, s1 = 0, s2 = 0;
        for (auto [d, mu] : divs_mu_r) {
            i64 t = M / d;
            cnt += (i128)mu * t;
            s1 += (i128)mu * d * (i128)t * (t + 1) / 2;
            s2 += (i128)mu * d * d * (i128)t * (t + 1) * (2 * (i128)t + 1) / 6;
        }
        return {cnt, s1, s2};
    };

    auto [cR, sR1, sR2] = pref(R);
    auto [cL, sL1, sL2] = pref(L - 1);
    return {cR - cL, sR1 - sL1, sR2 - sL2};
}

static i128 fast_large_part_by_r(i64 n, i64 B, const std::vector<std::vector<std::pair<int, int>>>& divs_mu) {
    i128 total = 0;

    for (i64 r = B + 1; r < n; ++r) {
        i64 k = (n - 1) / r;
        for (i64 a = 1; a <= k; ++a) {
            for (i64 b = 1; b <= a; ++b) {
                i64 mult = (a == b ? 1 : 2);
                i64 s_max = std::min<i64>(r, (n - 1 - a * r) / b);

                auto [cnt, s1, s2] = coprime_interval_sums(divs_mu[(size_t)r], 1, s_max);

                i128 n1 = n - a * r;
                i128 n2 = n - b * r;

                total += (i128)mult * (
                    n1 * n2 * cnt
                    - ((i128)a * n1 + (i128)b * n2) * s1
                    + (i128)a * b * s2
                );
            }
        }
    }

    return total;
}

static i128 rect_fastestest(i64 n) {
    i64 B = (i64)(std::pow((long double)n, 2.0L / 3.0L) / 2.0L);
    auto divs_mu = build_divs_mu((int)n);
    i128 S2 = fast_large_part_by_r(n, B, divs_mu);
    i128 S1 = small_part(n - 1, B);
    i128 S3 = (i128)n * (n - 1) * n * (n - 1) / 4;
    return S3 + 2 * S2 + S1;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <k>\n";
        std::cerr << "Benchmark: rect_fastestest(2^1), rect_fastestest(2^2), ..., rect_fastestest(2^k)\n";
        return 1;
    }

    int kmax = 0;
    try {
        kmax = std::stoi(argv[1]);
    } catch (...) {
        std::cerr << "Invalid k\n";
        return 1;
    }

    if (kmax < 1 || kmax > 30) {
        std::cerr << "k must be in range [1, 30]\n";
        std::cerr << "Upper bound is practical, not mathematical.\n";
        return 1;
    }

    for (int k = 1; k <= kmax; ++k) {
        i64 n = 1LL << k;

        auto t1 = std::chrono::steady_clock::now();
        i128 ans = rect_fastestest(n);
        auto t2 = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed = t2 - t1;

        std::cout
            << "k = " << k
            << ", n = " << n
            << ", rect_fastestest(n) = " << to_string_i128(ans)
            << ", time = " << std::fixed << std::setprecision(6)
            << elapsed.count() << " s\n";

        std::cout.flush();
    }

    return 0;
}