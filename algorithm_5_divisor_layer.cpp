#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

using u64 = uint64_t;
using i64 = int64_t;
using i128 = __int128_t;

struct Mom6 {
    i128 v0, v1, v2, v3, v4, v5;
};

struct Coeff3 {
    i128 A, B, C;
};

struct MuSum {
    i128 m0, m1, m2;
};

static constexpr Mom6 Z6{0, 0, 0, 0, 0, 0};
static constexpr Coeff3 Z3{0, 0, 0};
static constexpr MuSum ZM{0, 0, 0};

struct SplitMix64Hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t seed = chrono::steady_clock::now().time_since_epoch().count();
        return static_cast<size_t>(splitmix64(x + seed));
    }
};

static inline u64 isqrt_u64(u64 x) {
    u64 r = static_cast<u64>(sqrt(static_cast<long double>(x)));
    while ((__uint128_t)(r + 1) * (r + 1) <= x) ++r;
    while ((__uint128_t)r * r > x) --r;
    return r;
}

static string to_string_i128(i128 x) {
    if (x == 0) return "0";

    bool neg = x < 0;
    unsigned __int128 u;

    if (neg) {
        u = static_cast<unsigned __int128>(-(x + 1)) + 1;
    } else {
        u = static_cast<unsigned __int128>(x);
    }

    string s;
    while (u > 0) {
        s.push_back(char('0' + int(u % 10)));
        u /= 10;
    }

    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    return s;
}

static inline i128 sum1_upto(u64 n) {
    const i128 x = (i128)n;
    return x * (x + 1) / 2;
}

static inline i128 sum2_upto(u64 n) {
    const i128 x = (i128)n;
    return x * (x + 1) * (2 * x + 1) / 6;
}

static inline i128 sum1_range(u64 l, u64 r) {
    return sum1_upto(r) - sum1_upto(l - 1);
}

static inline i128 sum2_range(u64 l, u64 r) {
    return sum2_upto(r) - sum2_upto(l - 1);
}

static Mom6 hcalc_core(u64 n, u64 m, u64 a, u64 b) {
    if (n == 0) return Z6;

    if (a >= m || b >= m) {
        const u64 q = a / m;
        const u64 r = a % m;
        const u64 s = b / m;
        const u64 t = b % m;

        const Mom6 h = hcalc_core(n, m, r, t);

        const i128 nn = (i128)n;
        const i128 qv = (i128)q;
        const i128 sv = (i128)s;

        const i128 s1 = nn * (nn - 1) / 2;
        const i128 s2 = nn * (nn - 1) * (2 * nn - 1) / 6;
        const i128 s3 = s1 * s1;

        const i128 q2 = qv * qv;
        const i128 ss = sv * sv;

        return Mom6{
            qv * s1 + sv * nn + h.v0,
            qv * s2 + sv * s1 + h.v1,
            qv * s3 + sv * s2 + h.v2,
            q2 * s2 + 2 * qv * sv * s1 + ss * nn + 2 * qv * h.v1 + 2 * sv * h.v0 + h.v3,
            q2 * s3 + 2 * qv * sv * s2 + ss * s1 + 2 * qv * h.v2 + 2 * sv * h.v1 + h.v4,
            q2 * qv * s3
                + 3 * q2 * sv * s2
                + 3 * qv * ss * s1
                + ss * sv * nn
                + 3 * q2 * h.v2
                + 6 * qv * sv * h.v1
                + 3 * ss * h.v0
                + 3 * qv * h.v4
                + 3 * sv * h.v3
                + h.v5,
        };
    }

    if (a == 0) return Z6;

    const u64 y = (u64)(((i128)a * (n - 1) + b) / m);
    if (y == 0) return Z6;

    const Mom6 g = hcalc_core(y, a, m, m + a - 1 - b);

    const i128 nn = (i128)n;
    const i128 yy = (i128)y;

    const i128 t1 = nn * (nn - 1) / 2;
    const i128 t2 = nn * (nn - 1) * (2 * nn - 1) / 6;
    const i128 y2 = yy * yy;

    return Mom6{
        nn * yy - g.v0,
        yy * t1 - (g.v3 - g.v0) / 2,
        yy * t2 - (2 * g.v5 - 3 * g.v3 + g.v0) / 6,
        nn * y2 - 2 * g.v1 - g.v0,
        y2 * t1 - g.v4 - g.v3 / 2 + g.v1 + g.v0 / 2,
        nn * yy * y2 - 3 * g.v2 - 3 * g.v1 - g.v0,
    };
}

static Mom6 prefix_moments_layer(u64 Y, u64 N, u64 coef, u64 mod) {
    if (Y == 0) return Z6;

    const u64 B = (u64)((i128)N - (i128)coef * Y);
    const u64 s = B / mod;
    const u64 t = B % mod;

    const Mom6 u = hcalc_core(Y, mod, coef, t);

    const i128 YY = (i128)Y;
    const i128 sv = (i128)s;

    const i128 z1 = YY * (YY - 1) / 2;
    const i128 z2 = YY * (YY - 1) * (2 * YY - 1) / 6;
    const i128 ss = sv * sv;

    const i128 h01 = sv * YY + u.v0;
    const i128 h11 = sv * z1 + u.v1;
    const i128 h21 = sv * z2 + u.v2;
    const i128 h02 = ss * YY + 2 * sv * u.v0 + u.v3;
    const i128 h12 = ss * z1 + 2 * sv * u.v1 + u.v4;
    const i128 h03 = ss * sv * YY + 3 * ss * u.v0 + 3 * sv * u.v3 + u.v5;

    const i128 p1 = YY * (YY + 1) / 2;
    const i128 p2 = YY * (YY + 1) * (2 * YY + 1) / 6;
    const i128 p3 = p1 * p1;

    const i128 p2p1 = p2 - p1;
    const i128 p3p2 = p3 - p2;

    const i128 s11 = YY * h01 - h11;
    const i128 s12 = YY * h02 - h12;

    return Mom6{
        h01 - (p1 - YY),
        (h02 + h01 - p2p1) / 2,
        s11 - p2p1,
        (2 * h03 + 3 * h02 + h01 - (2 * p3 - 3 * p2 + p1)) / 6,
        (s12 + s11 - p3p2) / 2,
        YY * YY * h01 - 2 * YY * h11 + h21 - p3p2,
    };
}

static const vector<Mom6>& get_cap_moments(u64 cap) {
    static u64 cached_cap = numeric_limits<u64>::max();
    static vector<Mom6> arr;

    if (cached_cap == cap) return arr;

    cached_cap = cap;
    arr.assign((size_t)cap + 1, Z6);

    const i128 cc = (i128)cap;
    const i128 c1 = cc * (cc + 1) / 2;
    const i128 c2 = cc * (cc + 1) * (2 * cc + 1) / 6;

    for (u64 Y = 1; Y <= cap; ++Y) {
        const i128 y = (i128)Y;

        const i128 s1 = y * (y + 1) / 2;
        const i128 s2 = y * (y + 1) * (2 * y + 1) / 6;
        const i128 s3 = s1 * s1;

        const i128 s2s1 = s2 - s1;
        const i128 s3s2 = s3 - s2;

        arr[(size_t)Y] = Mom6{
            cc * y - (s1 - y),
            c1 * y - s2s1 / 2,
            cc * s1 - s2s1,
            c2 * y - (2 * s3 - 3 * s2 + s1) / 6,
            c1 * s1 - s3s2 / 2,
            cc * s2 - s3s2,
        };
    }

    return arr;
}

struct CPre {
    u64 Y;
    i128 S1;
    i128 S2;
    u64 C2;
};

static Coeff3 layer_coeff_uncached(u64 N) {
    if (N < 2) return Z3;

    const u64 cap = isqrt_u64(N);
    const vector<Mom6>& cap_moms = get_cap_moments(cap);

    const u64 maxc = 2 * cap;
    vector<CPre> cp((size_t)maxc + 1);

    for (u64 c = 3; c <= maxc; ++c) {
        const u64 Y = N / c;
        const i128 y = (i128)Y;

        cp[(size_t)c] = CPre{
            Y,
            y * (y + 1) / 2,
            y * (y + 1) * (2 * y + 1) / 6,
            c * c,
        };
    }

    i128 ZA = 0;
    i128 ZB = 0;
    i128 ZD = 0;

    for (u64 a = 2; a <= cap; ++a) {
        const i128 aa = (i128)a * a;
        const u64 a_cap = a * cap;

        for (u64 b = 1; b < a; ++b) {
            const u64 c = a + b;
            const CPre& p = cp[(size_t)c];

            const u64 Y = p.Y;
            const i128 sy1 = p.S1;
            const i128 sy2 = p.S2;
            const i128 c2 = (i128)p.C2;

            Mom6 m = prefix_moments_layer(Y, N, b, a);

            const i128 bb = (i128)b * b;
            const i128 pq = (i128)a * b;
            const i128 cc = (i128)c;

            i128 A = 8 * m.v0 - 6 * (i128)Y;
            i128 B = 8 * cc * (m.v1 + m.v2) - 12 * cc * sy1;
            i128 D = 8 * (pq * (m.v3 + m.v5) + (aa + bb) * m.v4) - 6 * c2 * sy2;

            const u64 Yi = min(cap, Y);

            if (Yi > 0) {
                Mom6 t;

                if ((N - b) / a <= cap) {
                    if (Yi == Y) {
                        t = m;
                    } else {
                        t = prefix_moments_layer(Yi, N, b, a);
                    }
                } else {
                    u64 Y0 = 0;

                    if (N >= a_cap) {
                        const u64 q = (N - a_cap) / b;
                        Y0 = min(Yi, q);
                    }

                    t = cap_moms[(size_t)Y0];

                    if (Y0 < Yi) {
                        Mom6 aM;

                        if (Yi == Y) {
                            aM = m;
                        } else {
                            aM = prefix_moments_layer(Yi, N, b, a);
                        }

                        if (Y0 > 0) {
                            const Mom6 bM = prefix_moments_layer(Y0, N, b, a);

                            aM.v0 -= bM.v0;
                            aM.v1 -= bM.v1;
                            aM.v2 -= bM.v2;
                            aM.v3 -= bM.v3;
                            aM.v4 -= bM.v4;
                            aM.v5 -= bM.v5;
                        }

                        t.v0 += aM.v0;
                        t.v1 += aM.v1;
                        t.v2 += aM.v2;
                        t.v3 += aM.v3;
                        t.v4 += aM.v4;
                        t.v5 += aM.v5;
                    }
                }

                const i128 yi = (i128)Yi;
                const i128 si1 = yi * (yi + 1) / 2;
                const i128 si2 = yi * (yi + 1) * (2 * yi + 1) / 6;

                A -= 4 * t.v0 - 2 * yi;
                B -= 4 * cc * (t.v1 + t.v2) - 4 * cc * si1;
                D -= 4 * (pq * (t.v3 + t.v5) + (aa + bb) * t.v4) - 2 * c2 * si2;
            }

            ZA += A;
            ZB += B;
            ZD += D;
        }
    }

    for (u64 x = 1; x <= cap; ++x) {
        const u64 M = N / x;

        if (M >= 3) {
            const u64 K = M / 2;
            const u64 L = (M - 1) / 2;

            const i128 k = (i128)K;
            const i128 l = (i128)L;
            const i128 xx = (i128)x;

            const i128 K1 = k * (k + 1) / 2;
            const i128 K2 = k * (k + 1) * (2 * k + 1) / 6;
            const i128 K3 = K1 * K1;

            const i128 L1 = l * (l + 1) / 2;
            const i128 L2 = l * (l + 1) * (2 * l + 1) / 6;
            const i128 L3 = L1 * L1;

            ZA += 2 * ((K1 - k) + L1);
            ZB += 4 * xx * (2 * (K2 - K1) + 2 * L2 + L1);
            ZD += 2 * xx * xx * (4 * (K3 - K2) + 4 * L3 + 4 * L2 + L1);
        }
    }

    return Coeff3{ZA, ZB, ZD};
}

static unordered_map<u64, Coeff3, SplitMix64Hash> layer_cache;

static const Coeff3& layer_coeff(u64 N) {
    auto it = layer_cache.find(N);
    if (it != layer_cache.end()) return it->second;

    Coeff3 z = layer_coeff_uncached(N);
    auto inserted = layer_cache.emplace(N, z);
    return inserted.first->second;
}

class MuPrefixSolver {
public:
    explicit MuPrefixSolver(u64 max_n) {
        limit_ = choose_limit(max_n);
        build_sieve(limit_);

        const u64 expected = 4 * isqrt_u64(max_n) + 1000;
        memo_.reserve((size_t)expected);
        memo_.max_load_factor(0.7f);
    }

    MuSum get(u64 n) {
        if (n == 0) return ZM;

        if (n <= limit_) {
            const size_t idx = (size_t)n;
            return MuSum{(i128)pref0_[idx], pref1_[idx], pref2_[idx]};
        }

        auto it = memo_.find(n);
        if (it != memo_.end()) return it->second;

        i128 m0 = 1;
        i128 m1 = 1;
        i128 m2 = 1;

        for (u64 l = 2; l <= n; ) {
            const u64 v = n / l;
            const u64 r = n / v;
            const MuSum sub = get(v);

            const i128 cnt = (i128)(r - l + 1);
            const i128 s1 = sum1_range(l, r);
            const i128 s2 = sum2_range(l, r);

            m0 -= cnt * sub.m0;
            m1 -= s1 * sub.m1;
            m2 -= s2 * sub.m2;

            l = r + 1;
        }

        MuSum res{m0, m1, m2};
        memo_.emplace(n, res);
        return res;
    }

private:
    u64 limit_ = 0;
    vector<i64> pref0_;
    vector<i128> pref1_;
    vector<i128> pref2_;
    unordered_map<u64, MuSum, SplitMix64Hash> memo_;

    static u64 choose_limit(u64 max_n) {
        if (max_n <= 2'000'000ull) return max_n;

        const long double x = pow((long double)max_n, 2.0L / 3.0L);
        u64 lim = (u64)(x * 1.25L) + 1000;

        lim = max<u64>(lim, 1'000'000ull);
        lim = min<u64>(lim, 5'000'000ull);
        lim = min<u64>(lim, max_n);

        return lim;
    }

    void build_sieve(u64 n) {
        vector<uint32_t> lp((size_t)n + 1, 0);
        vector<uint32_t> primes;
        vector<int8_t> mu((size_t)n + 1, 0);

        pref0_.assign((size_t)n + 1, 0);
        pref1_.assign((size_t)n + 1, 0);
        pref2_.assign((size_t)n + 1, 0);

        if (n == 0) return;

        mu[1] = 1;
        pref0_[1] = 1;
        pref1_[1] = 1;
        pref2_[1] = 1;

        i64 acc0 = 1;
        i128 acc1 = 1;
        i128 acc2 = 1;

        primes.reserve((size_t)(n / 10 + 10));

        for (u64 i = 2; i <= n; ++i) {
            if (lp[(size_t)i] == 0) {
                lp[(size_t)i] = (uint32_t)i;
                primes.push_back((uint32_t)i);
                mu[(size_t)i] = -1;
            }

            const int mui = mu[(size_t)i];
            const i128 ii = (i128)i;

            acc0 += mui;
            acc1 += (i128)mui * ii;
            acc2 += (i128)mui * ii * ii;

            pref0_[(size_t)i] = acc0;
            pref1_[(size_t)i] = acc1;
            pref2_[(size_t)i] = acc2;

            const uint32_t lpi = lp[(size_t)i];

            for (uint32_t p : primes) {
                const u64 v = i * (u64)p;
                if (v > n || p > lpi) break;

                lp[(size_t)v] = p;

                if (p == lpi) {
                    mu[(size_t)v] = 0;
                    break;
                }

                mu[(size_t)v] = (int8_t)-mui;
            }
        }
    }
};

static i128 f1(u64 n, MuPrefixSolver& mu_solver) {
    if (n <= 1) return 0;

    const i128 nn = (i128)n;
    const i128 n2 = nn * nn;

    i128 ans = 0;

    for (u64 left = 1; left <= n; ) {
        const u64 N = n / left;
        const u64 right = n / N;
        const u64 lm1 = left - 1;

        const MuSum mr = mu_solver.get(right);
        const MuSum ml = mu_solver.get(lm1);

        const i128 s0 = mr.m0 - ml.m0;
        const i128 s1 = mr.m1 - ml.m1;
        const i128 s2 = mr.m2 - ml.m2;

        if (s0 != 0 || s1 != 0 || s2 != 0) {
            const Coeff3& z = layer_coeff(N);

            ans += n2 * z.A * s0 - nn * z.B * s1 + z.C * s2;
        }

        left = right + 1;
    }

    ans += nn * (nn - 1) * (nn - 1) * (2 * nn - 1) / 6;

    return ans;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    unsigned k;

    if (argc >= 2) {
        k = static_cast<unsigned>(strtoul(argv[1], nullptr, 0));
    } else {
        if (!(cin >> k)) return 0;
    }

    if (k > 32) {
        cerr << "Error: k must be <= 32, because this version is intended for n <= 2^32.\n";
        return 1;
    }

    const u64 max_n = 1ull << k;
    MuPrefixSolver mu_solver(max_n);

    layer_cache.reserve((size_t)(4 * isqrt_u64(max_n) + 1000));
    layer_cache.max_load_factor(0.7f);

    cout << "power\tn\tresult\ttime_seconds\n";
    cout.flush();

    for (unsigned e = 1; e <= k; ++e) {
        const u64 n = 1ull << e;

        const auto t0 = chrono::steady_clock::now();
        const i128 ans = f1(n, mu_solver);
        const auto t1 = chrono::steady_clock::now();

        const chrono::duration<double> dt = t1 - t0;

        cout << e << '\t'
             << n << '\t'
             << to_string_i128(ans) << '\t'
             << fixed << setprecision(6) << dt.count()
             << '\n';
        cout.flush();
    }

    return 0;
}
