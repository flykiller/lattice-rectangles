#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cmath>

using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using i128 = __int128_t;

struct Div {
    u32 d;
    int32_t mu;
};

std::string to_string_i128(i128 n) {
    if (n == 0) return "0";
    if (n < 0) return "-" + to_string_i128(-n);
    std::string s;
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

void floor_moments(u64 n, u64 m, u64 a, u64 b, 
                   i128& A01, i128& A11, i128& A21, i128& A02, i128& A12, i128& A03) {
    if (n == 0) {
        A01 = A11 = A21 = A02 = A12 = A03 = 0;
        return;
    }

    if (a >= m || b >= m) {
        u64 q = a / m; u64 r = a % m;
        u64 s = b / m; u64 t = b % m;

        i128 B01, B11, B21, B02, B12, B03;
        floor_moments(n, m, r, t, B01, B11, B21, B02, B12, B03);

        i128 s0 = n;
        i128 s1 = ((i128)n * (n - 1)) >> 1;
        i128 s2 = (i128)n * (n - 1) * (2 * n - 1) / 6;
        i128 s3 = s1 * s1;

        i128 qq = (i128)q * q;
        i128 qqq = qq * q;
        i128 ss = (i128)s * s;
        i128 sss = ss * s;
        i128 qs2 = 2 * (i128)q * s;

        A01 = q * s1 + s * s0 + B01;
        A11 = q * s2 + s * s1 + B11;
        A21 = q * s3 + s * s2 + B21;

        A02 = qq * s2 + ss * s0 + B02 + qs2 * s1 + 2 * q * B11 + 2 * s * B01;
        A12 = qq * s3 + ss * s1 + B12 + qs2 * s2 + 2 * q * B21 + 2 * s * B11;

        A03 = qqq * s3 + 3 * qq * s * s2 + 3 * q * ss * s1 + sss * s0
            + 3 * qq * B21 + 6 * q * s * B11 + 3 * ss * B01
            + 3 * q * B12 + 3 * s * B02 + B03;
        return;
    }

    u64 y = (a * (n - 1) + b) / m;
    if (y == 0) {
        A01 = A11 = A21 = A02 = A12 = A03 = 0;
        return;
    }

    u64 bp = m + a - 1 - b;
    i128 B01, B11, B21, B02, B12, B03;
    floor_moments(y, a, m, bp, B01, B11, B21, B02, B12, B03);

    i128 t1 = ((i128)n * (n - 1)) >> 1;
    i128 t2 = (i128)n * (n - 1) * (2 * n - 1) / 6;
    i128 yy = (i128)y * y;
    i128 n128 = n;

    A01 = n128 * y - B01;
    A11 = y * t1 - (B02 - B01) / 2;
    A21 = y * t2 - (2 * B03 - 3 * B02 + B01) / 6;
    A02 = n128 * yy - 2 * B11 - B01;
    A12 = yy * t1 - B12 - B02 / 2 + B11 + B01 / 2;
    A03 = n128 * y * yy - 3 * B21 - 3 * B11 - B01;
}

i128 count_rectangles_uv(u64 N, u64 u, u64 v) {
    if (N == 0) return 0;

    u64 U = u < v ? v : u;
    u64 V = u < v ? u : v;
    
    auto segment_sum = [&](u64 lo, u64 hi, u64 a, u64 b, u64 m) -> i128 {
        if (lo > hi) return 0;
        u64 L = hi - lo + 1;
        
        i128 A01, A11, A21, A02, A12, A03;
        floor_moments(L, m, a, b, A01, A11, A21, A02, A12, A03);

        i128 hi128 = hi;
        i128 hi2 = hi128 * hi128;

        i128 sum_t = A01;
        i128 sum_kt = hi128 * A01 - A11;
        i128 sum_k2t = hi2 * A01 - 2 * hi128 * A11 + A21;
        i128 sum_t2 = A02;
        i128 sum_kt2 = hi128 * A02 - A12;
        i128 sum_t3 = A03;

        i128 iUV = (i128)U * V;
        i128 iU2pV2 = (i128)U * U + (i128)V * V;
        i128 iUpV = (i128)U + V;
        i128 iN = N;

        i128 term1 = 6 * iUV * sum_k2t;
        i128 term2 = 3 * iU2pV2 * sum_kt2;
        i128 term3 = (-6 * iN * iUpV + 3 * iU2pV2 - 6 * iUpV) * sum_kt;
        i128 term4 = 2 * iUV * sum_t3;
        i128 term5 = (-3 * iN * iUpV + 3 * iUV - 3 * iUpV) * sum_t2;
        i128 term6 = (6 * iN * iN - 3 * iN * iUpV + 12 * iN + iUV - 3 * iUpV + 6) * sum_t;

        return (term1 + term2 + term3 + term4 + term5 + term6) / 6;
    };

    u64 m1 = N / (U + V);
    u64 m2 = N / U;

    i128 ans = 0;
    if (m1 > 0) ans += segment_sum(1, m1, V, N - V * m1, U);
    if (m2 > m1) ans += segment_sum(m1 + 1, m2, U, N - U * m2, V);

    return ans;
}

i128 small_part(u64 N, u64 B) {
    i128 s = count_rectangles_uv(N, 1, 1);
    u64 f_a = 0, f_b = 1, f_c = 1, f_d = B;
    
    while (f_c <= B) {
        u64 k_val = (B + f_b) / f_d;
        u64 next_c = k_val * f_c - f_a;
        u64 next_d = k_val * f_d - f_b;
        f_a = f_c; f_b = f_d; f_c = next_c; f_d = next_d;

        if (f_a == 1 && f_b == 1) break; 
        if (f_a > 0) { 
            s += 2 * count_rectangles_uv(N, f_b, f_a); 
        }
    }
    return s;
}

inline void pref(const std::vector<Div>& flat_divs, u32 start_idx, u32 end_idx, u64 M,
                 i64& cnt, i128& s1, i128& s2) {
    cnt = 0; s1 = 0; s2 = 0;
    if (M == 0) return;
    
    for (u32 i = start_idx; i < end_idx; ++i) {
        u64 d = flat_divs[i].d;
        int32_t mu = flat_divs[i].mu;
        u64 t = M / d;
        if (t == 0) continue;

        i128 sum1 = ((i128)t * (t + 1)) >> 1;
        
        u64 ta = t, tb = t + 1, tc = 2 * t + 1;
        if (ta % 2 == 0) ta /= 2; else tb /= 2;
        if (ta % 3 == 0) ta /= 3; else if (tb % 3 == 0) tb /= 3; else tc /= 3;
        i128 sum2 = (i128)ta * tb * tc;

        u64 dd = d * d;

        if (mu == 1) {
            cnt += t;
            s1 += (i128)d * sum1;
            s2 += (i128)dd * sum2;
        } else if (mu == -1) {
            cnt -= t;
            s1 -= (i128)d * sum1;
            s2 -= (i128)dd * sum2;
        }
    }
}

i128 fast_large_part_by_r(u64 n, u64 B, const std::vector<u32>& head, const std::vector<Div>& flat_divs) {
    i128 total = 0;
    for (u64 r = B + 1; r < n; ++r) {
        u64 k = (n - 1) / r;
        if (k == 0) break; 
        
        u32 start_idx = head[r];
        u32 end_idx = head[r + 1];

        for (u64 a = 1; a <= k; ++a) {
            for (u64 b = 1; b <= a; ++b) {
                u64 mult = (a == b) ? 1 : 2;
                u64 s_max = std::min(r, (n - 1 - a * r) / b);

                i64 cnt; i128 s1, s2;
                pref(flat_divs, start_idx, end_idx, s_max, cnt, s1, s2);

                i128 n1 = (i128)n - (i128)a * r;
                i128 n2 = (i128)n - (i128)b * r;
                i128 ia = a, ib = b;

                i128 term = n1 * n2 * cnt - (ia * n1 + ib * n2) * s1 + ia * ib * s2;
                total += (i128)mult * term;
            }
        }
    }
    return total;
}

i128 rect_fastestest(u64 n) {
    if (n <= 1) return 0;
    
    u64 B = std::pow(n, 2.0 / 3.0) / 2.0;

    std::vector<u32> head(n + 2, 0);
    std::vector<Div> flat_divs;
    flat_divs.reserve(n * 4); 

    std::vector<u32> spf(n + 1);
    for (u32 i = 2; i <= n; ++i) spf[i] = i;
    for (u32 i = 2; i * i <= n; ++i) {
        if (spf[i] == i) {
            for (u32 j = i * i; j <= n; j += i) {
                if (spf[j] == j) spf[j] = i;
            }
        }
    }

    head[1] = 0;
    flat_divs.push_back({1, 1});

    for (u32 x = 2; x <= n; ++x) {
        head[x] = flat_divs.size();
        u32 y = x;
        
        u32 ps[32]; 
        u32 ps_count = 0;

        while (y > 1) {
            u32 p = spf[y];
            ps[ps_count++] = p;
            while (y % p == 0) y /= p;
        }

        u32 start = head[x];
        flat_divs.push_back({1, 1});
        for (u32 j = 0; j < ps_count; ++j) {
            u32 p = ps[j];
            u32 sz = flat_divs.size() - start;
            for (u32 i = 0; i < sz; ++i) {
                flat_divs.push_back({
                    flat_divs[start + i].d * p,
                    -flat_divs[start + i].mu
                });
            }
        }
    }
    head[n + 1] = flat_divs.size();

    i128 S2 = fast_large_part_by_r(n, B, head, flat_divs);
    i128 S1 = small_part(n - 1, B);
    
    i128 in = n, in_minus = n - 1;
    i128 S3 = (in * in_minus * in * in_minus) / 4;

    return S3 + 2 * S2 + S1;
}

int main(int argc, char **argv) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " k\n";
        return 1;
    }

    int k = std::atoi(argv[1]);

    for (int e = 1; e <= k; ++e) {
        u64 n = (1ULL << e);

        auto t0 = std::chrono::steady_clock::now();
        i128 ans;
        try {
            ans = rect_fastestest(n);
        } catch (const std::bad_alloc& ex) {
            std::cerr << "\n[MEMORY LIMIT] Stopped at k=" << e << "\n";
            break; 
        }
        auto t1 = std::chrono::steady_clock::now();

        std::chrono::duration<double> dt = t1 - t0;
        std::cout << "n=" << n << " result=" << to_string_i128(ans) << " time=" << dt.count() << "\n";
    }

    return 0;
}
