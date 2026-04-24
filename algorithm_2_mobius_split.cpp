#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

using i64 = long long;
using i128 = __int128_t;
using u128 = __uint128_t;

static std::string to_string_i128(i128 x) {
    if (x == 0) return "0";
    bool neg = (x < 0);
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

static i64 isqrt_i64(i64 x) {
    long double r = std::sqrt((long double)x);
    i64 y = (i64)r;
    while ((y + 1) > 0 && (i128)(y + 1) * (y + 1) <= x) ++y;
    while ((i128)y * y > x) --y;
    return y;
}

static std::vector<int> build_primes(int limit) {
    if (limit < 2) return {};
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; (i64)p * p <= limit; ++p) {
        if (is_prime[p]) {
            for (int q = p * p; q <= limit; q += p) {
                is_prime[q] = false;
            }
        }
    }
    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i) {
        if (is_prime[i]) primes.push_back(i);
    }
    return primes;
}

static std::vector<int> distinct_prime_factors(i64 x, const std::vector<int>& primes) {
    std::vector<int> ps;
    i64 y = x;
    for (int p : primes) {
        if ((i64)p * p > y) break;
        if (y % p == 0) {
            ps.push_back(p);
            while (y % p == 0) y /= p;
        }
    }
    if (y > 1) ps.push_back((int)y);
    return ps;
}

static i128 sum_h1w1(i64 h, i64 w, i64 s, i64 r) {
    i64 m = 1 + std::min((h - 1) / s, (w - 1) / r);
    i128 mm = (i128)m;
    i128 sk  = mm * (mm - 1) / 2;
    i128 sk2 = (mm - 1) * mm * (2 * mm - 1) / 6;
    return mm * h * w - ((i128)h * r + (i128)w * s) * sk + (i128)r * s * sk2;
}

struct CoprimeSums {
    i128 cnt = 0;
    i128 s1 = 0;
    i128 s2 = 0;
};

static void coprime_interval_sums_dfs(
    const std::vector<int>& pf,
    int idx,
    i64 d,
    int mu,
    i64 L,
    i64 U,
    CoprimeSums& out
) {
    if (idx == (int)pf.size()) {
        i64 lo = (L + d - 1) / d;
        i64 hi = U / d;
        if (lo > hi) return;

        i128 ilo = lo;
        i128 ihi = hi;
        i128 cnt = (ihi - ilo + 1);
        i128 sum1 = ihi * (ihi + 1) / 2 - (ilo - 1) * ilo / 2;
        i128 sum2 = ihi * (ihi + 1) * (2 * ihi + 1) / 6
                  - (ilo - 1) * ilo * (2 * ilo - 1) / 6;

        out.cnt += (i128)mu * cnt;
        out.s1  += (i128)mu * d * sum1;
        out.s2  += (i128)mu * d * d * sum2;
        return;
    }

    coprime_interval_sums_dfs(pf, idx + 1, d, mu, L, U, out);
    coprime_interval_sums_dfs(pf, idx + 1, d * pf[idx], -mu, L, U, out);
}

static CoprimeSums coprime_interval_sums(i64 r, i64 L, i64 U, const std::vector<int>& primes) {
    CoprimeSums out;
    if (L > U) return out;
    auto pf = distinct_prime_factors(r, primes);
    coprime_interval_sums_dfs(pf, 0, 1, 1, L, U, out);
    return out;
}

static i128 small_part(i64 n, i64 B, const std::vector<std::vector<int>>& residues) {
    i128 total = 0;
    for (i64 s = 1; s < B; ++s) {
        i64 start = (s == 1 ? 2 : s);
        for (int a : residues[(size_t)s]) {
            i64 first = a;
            if (first < start) {
                first = a + ((start - a + s - 1) / s) * s;
            }
            for (i64 r = first; r < n - s; r += s) {
                i64 h0 = n - r - s;
                i64 km = 1 + std::min((h0 - 1) / r, (h0 - 1) / s);
                for (i64 k = 0; k < km; ++k) {
                    total += sum_h1w1(h0 - k * r, h0 - k * s, s, r);
                }
            }
        }
    }
    return total;
}

static i128 large_part(i64 n, i64 B, const std::vector<int>& primes) {
    i128 total = 0;
    i64 T = (n - 1) / B;

    for (i64 t = 0; t < T; ++t) {
        i64 a = t + 1;
        i64 umax = std::min(t, T - t - 2);
        if (umax < 0) continue;

        for (i64 u = 0; u <= umax; ++u) {
            i64 b = u + 1;
            i64 mult = (t == u ? 1 : 2);

            i64 r_lo = std::max<i64>(B, 2);
            i64 r_hi = (n - 1 - b * B) / a;
            if (r_lo > r_hi) continue;

            for (i64 r = r_lo; r <= r_hi; ++r) {
                i64 U = std::min<i64>(r, (n - 1 - a * r) / b);
                if (U < B) continue;

                CoprimeSums cs = coprime_interval_sums(r, B, U, primes);
                i128 n1 = (i128)n - (i128)a * r;
                i128 n2 = (i128)n - (i128)b * r;

                total += (i128)mult * (
                    n1 * n2 * cs.cnt
                    - ((i128)a * n1 + (i128)b * n2) * cs.s1
                    + (i128)a * b * cs.s2
                );
            }
        }
    }
    return total;
}

static i128 rect_fastest(i64 n, const std::vector<int>& primes) {
    i64 B = isqrt_i64(n);

    std::vector<std::vector<int>> residues((size_t)B);
    for (i64 s = 1; s < B; ++s) {
        auto& vec = residues[(size_t)s];
        vec.reserve((size_t)s);
        for (i64 a = 1; a <= s; ++a) {
            if (std::gcd(a, s) == 1) vec.push_back((int)a);
        }
    }

    i128 nn = n;
    i128 base = (nn - 1) * (nn - 1) * nn * (2 * nn - 1) / 6;
    i128 sp = small_part(n, B, residues);
    i128 lp = large_part(n, B, primes);

    return base + 2 * (sp + lp);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " K\n";
        std::cerr << "Runs for n = 2^1, 2^2, ..., 2^K\n";
        return 1;
    }

    int K = std::atoi(argv[1]);
    if (K < 1 || K > 30) {
        std::cerr << "K must be in [1, 30]\n";
        return 1;
    }

    i64 max_n = 1LL << K;
    int prime_limit = (int)isqrt_i64(max_n);
    std::vector<int> primes = build_primes(prime_limit);

    std::cout << "k\tn\tvalue\ttime_sec\n";
    for (int k = 1; k <= K; ++k) {
        i64 n = 1LL << k;

        auto t0 = std::chrono::steady_clock::now();
        i128 ans = rect_fastest(n, primes);
        auto t1 = std::chrono::steady_clock::now();

        std::chrono::duration<double> dt = t1 - t0;
        std::cout << k
                  << '\t' << n
                  << '\t' << to_string_i128(ans)
                  << '\t' << dt.count()
                  << '\n';
    }

    return 0;
}