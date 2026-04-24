#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <numeric>

using i64 = int64_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u128 = unsigned __int128;

struct Div {
    u32 d;
    int32_t mu;
};

std::string to_string_i128(u128 n) {
    if (n == 0) return "0";
    std::string s;
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

inline u128 sum_n(u64 x) {
    return ((u128)x * (x + 1)) >> 1;
}

inline u128 sum_sq(u64 x) {
    if (x == 0) return 0;
    u64 a = x, b = x + 1, c = 2 * x + 1;
    if (a % 2 == 0) a /= 2; else b /= 2;
    if (a % 3 == 0) a /= 3; else if (b % 3 == 0) b /= 3; else c /= 3;
    
    return (u128)a * (u128)b * (u128)c;
}

inline void sum_h1w1_split(u64 h, u64 w, u64 s, u64 r, u128& pos, u128& neg) {
    u64 m1 = (h - 1) / s;
    u64 m2 = (w - 1) / r;
    u64 m = 1 + (m1 < m2 ? m1 : m2);
    
    u128 sk = sum_n(m - 1);
    u128 sk2 = sum_sq(m - 1);
    
    pos = (u128)m * h * w + (u128)r * s * sk2;
    neg = ((u128)h * r + (u128)w * s) * sk;
}

inline void coprime_interval_sums(const std::vector<Div>& flat_divs, u32 start_idx, u32 end_idx, u64 L, u64 U, 
                                  u64& cnt_pos, u64& cnt_neg, u128& s1_pos, u128& s1_neg, u128& s2_pos, u128& s2_neg) {
    cnt_pos = cnt_neg = 0;
    s1_pos = s1_neg = 0;
    s2_pos = s2_neg = 0;
    
    for (u32 i = start_idx; i < end_idx; ++i) {
        u64 d = flat_divs[i].d;
        int32_t mu = flat_divs[i].mu;
        
        u64 lo = (L + d - 1) / d;
        u64 hi = U / d;
        if (lo > hi) continue;
        
        u64 c = hi - lo + 1;
        u128 sum1, sum2;
        
        if (lo == hi) {
            sum1 = hi;
            sum2 = (u128)hi * hi;
        } else {
            sum1 = sum_n(hi) - sum_n(lo - 1);
            sum2 = sum_sq(hi) - sum_sq(lo - 1);
        }
        
        if (mu == 1) {
            cnt_pos += c;
            s1_pos += (u128)d * sum1;
            s2_pos += (u128)d * d * sum2;
        } else {
            cnt_neg += c;
            s1_neg += (u128)d * sum1;
            s2_neg += (u128)d * d * sum2;
        }
    }
}

u128 count_all_rectangles(u64 n) {
    if (n <= 1) return 0;

    u64 B = std::sqrt(n);
    
    std::vector<u32> head(n + 2, 0);
    std::vector<Div> flat_divs;
    flat_divs.reserve(n * 4);

    std::vector<int32_t> spf(n + 1);
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
        std::vector<u32> ps;
        
        while (y > 1) {
            u32 p = spf[y];
            ps.push_back(p);
            while (y % p == 0) y /= p;
        }
        
        u32 start = head[x];
        flat_divs.push_back({1, 1});
        for (u32 p : ps) {
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

    std::vector<u32> res_head(B + 1, 0);
    std::vector<u32> flat_res;
    flat_res.reserve((u64)B * B / 3);
    for (u32 s = 1; s < B; ++s) {
        res_head[s] = flat_res.size();
        for (u32 a = 1; a <= s; ++a) {
            if (std::gcd(a, s) == 1) flat_res.push_back(a);
        }
    }
    res_head[B] = flat_res.size();

    u128 un = n;
    u128 base = (un - 1) * ((un - 1) * un * (2 * un - 1) / 6);

    u128 small_pos = 0, small_neg = 0;
    for (u32 s = 1; s < B; ++s) {
        u64 start = (s == 1) ? 2 : s;
        for (u32 idx = res_head[s]; idx < res_head[s+1]; ++idx) {
            u32 a = flat_res[idx];
            u64 first_r = a + ((start - a + s - 1) / s) * s;
            for (u64 r = first_r; r < n - s; r += s) {
                u64 h0 = n - r - s;
                u64 M = std::min((h0 - 1) / r, (h0 - 1) / s);
                for (u64 k = 0; k <= M; ++k) {
                    u128 pos, neg;
                    sum_h1w1_split(h0 - k * r, h0 - k * s, s, r, pos, neg);
                    small_pos += pos;
                    small_neg += neg;
                }
            }
        }
    }

    u128 large_pos = 0, large_neg = 0;
    u64 max_t = (n - 1) / B - 1;
    for (u64 t = 0; t <= max_t; ++t) {
        u64 a = t + 1;
        u64 max_u = std::min(t, (n - 1) / B - t - 2);
        for (u64 u = 0; u <= max_u; ++u) {
            u64 mult = (t == u) ? 1 : 2;
            u64 b = u + 1;
            
            u64 min_r = std::max((u64)B, 2ULL);
            u64 max_r = (n - 1 - b * B) / a;
            
            for (u64 r = min_r; r <= max_r; ++r) {
                u64 U = std::min(r, (n - 1 - a * r) / b);
                if (B > U) continue;
                
                u64 cnt_pos, cnt_neg;
                u128 s1_pos, s1_neg, s2_pos, s2_neg;
                coprime_interval_sums(flat_divs, head[r], head[r + 1], B, U, 
                                      cnt_pos, cnt_neg, s1_pos, s1_neg, s2_pos, s2_neg);
                
                u128 n1 = n - a * r;
                u128 n2 = n - b * r;
                u128 a128 = a, b128 = b;
                
                u128 term_pos = n1 * n2 * cnt_pos + (a128 * n1 + b128 * n2) * s1_neg + a128 * b128 * s2_pos;
                u128 term_neg = n1 * n2 * cnt_neg + (a128 * n1 + b128 * n2) * s1_pos + a128 * b128 * s2_neg;
                
                large_pos += (u128)mult * term_pos;
                large_neg += (u128)mult * term_neg;
            }
        }
    }

    u128 total_pos = base + 2 * small_pos + 2 * large_pos;
    u128 total_neg = 2 * small_neg + 2 * large_neg;
    return total_pos - total_neg;
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
        u128 ans;
        try {
            ans = count_all_rectangles(n);
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
