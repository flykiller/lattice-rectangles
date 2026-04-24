#include <pthread.h>
#ifdef __linux__
#include <sched.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using u64 = uint64_t;
using i64 = int64_t;
using u128 = unsigned __int128;
using i128 = __int128_t;

// ===========================
//  Uint256 / Int256
// ===========================

struct Uint256 {
    uint64_t w[4];

    Uint256() : w{0, 0, 0, 0} {}
    explicit Uint256(uint64_t x) : w{x, 0, 0, 0} {}

    bool is_zero() const {
        return (w[0] | w[1] | w[2] | w[3]) == 0;
    }

    static int cmp(const Uint256& a, const Uint256& b) {
        for (int i = 3; i >= 0; --i) {
            if (a.w[i] < b.w[i]) return -1;
            if (a.w[i] > b.w[i]) return 1;
        }
        return 0;
    }

    void add_u64(uint64_t x) {
        u128 cur = (u128)w[0] + x;
        w[0] = (uint64_t)cur;
        uint64_t carry = (uint64_t)(cur >> 64);
        for (int i = 1; i < 4 && carry; ++i) {
            cur = (u128)w[i] + carry;
            w[i] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
    }

    void add_u128(u128 x) {
        uint64_t lo = (uint64_t)x;
        uint64_t hi = (uint64_t)(x >> 64);

        u128 cur = (u128)w[0] + lo;
        w[0] = (uint64_t)cur;
        uint64_t carry = (uint64_t)(cur >> 64);

        cur = (u128)w[1] + hi + carry;
        w[1] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);

        for (int i = 2; i < 4 && carry; ++i) {
            cur = (u128)w[i] + carry;
            w[i] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
    }

    void sub_u128(u128 x) {
        uint64_t lo = (uint64_t)x;
        uint64_t hi = (uint64_t)(x >> 64);

        uint64_t old = w[0];
        w[0] -= lo;
        uint64_t borrow = (w[0] > old) ? 1 : 0;

        old = w[1];
        u128 sub1 = (u128)hi + borrow;
        w[1] = (uint64_t)((u128)w[1] - sub1);
        borrow = ((u128)old < sub1) ? 1 : 0;

        for (int i = 2; i < 4 && borrow; ++i) {
            old = w[i];
            w[i] -= borrow;
            borrow = (w[i] > old) ? 1 : 0;
        }
    }

    void add(const Uint256& b) {
        u128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            u128 cur = (u128)w[i] + b.w[i] + carry;
            w[i] = (uint64_t)cur;
            carry = cur >> 64;
        }
    }

    void sub(const Uint256& b) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            uint64_t old = w[i];
            u128 subtr = (u128)b.w[i] + borrow;
            w[i] = (uint64_t)((u128)w[i] - subtr);
            borrow = ((u128)old < subtr) ? 1 : 0;
        }
    }

    void mul_u64(uint64_t m) {
        u128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            u128 cur = (u128)w[i] * m + carry;
            w[i] = (uint64_t)cur;
            carry = cur >> 64;
        }
    }

    uint32_t div10() {
        u128 rem = 0;
        for (int i = 3; i >= 0; --i) {
            u128 cur = (rem << 64) | w[i];
            w[i] = (uint64_t)(cur / 10);
            rem = cur % 10;
        }
        return (uint32_t)rem;
    }

    std::string to_string() const {
        if (is_zero()) return "0";
        Uint256 tmp = *this;
        std::string s;
        while (!tmp.is_zero()) {
            uint32_t r = tmp.div10();
            s.push_back(char('0' + r));
        }
        std::reverse(s.begin(), s.end());
        return s;
    }
};

struct Int256 {
    bool neg;
    Uint256 mag;

    Int256() : neg(false), mag() {}

    bool is_zero() const { return mag.is_zero(); }

    void normalize() {
        if (mag.is_zero()) neg = false;
    }

    void add_i128(i128 v) {
        if (v == 0) return;

        if (v > 0) {
            u128 m = (u128)v;
            if (!neg) {
                mag.add_u128(m);
            } else {
                Uint256 t;
                t.add_u128(m);
                int c = Uint256::cmp(mag, t);
                if (c >= 0) {
                    mag.sub(t);
                } else {
                    Uint256 nm = t;
                    nm.sub(mag);
                    mag = nm;
                    neg = false;
                }
                normalize();
            }
        } else {
            u128 m = (u128)(-v);
            if (neg) {
                mag.add_u128(m);
            } else {
                Uint256 t;
                t.add_u128(m);
                int c = Uint256::cmp(mag, t);
                if (c >= 0) {
                    mag.sub(t);
                } else {
                    Uint256 nm = t;
                    nm.sub(mag);
                    mag = nm;
                    neg = true;
                }
                normalize();
            }
        }
    }

    void add(const Int256& other) {
        if (other.is_zero()) return;
        if (this->is_zero()) {
            *this = other;
            return;
        }

        if (neg == other.neg) {
            mag.add(other.mag);
            return;
        }

        int c = Uint256::cmp(mag, other.mag);
        if (c == 0) {
            mag = Uint256();
            neg = false;
        } else if (c > 0) {
            mag.sub(other.mag);
        } else {
            Uint256 nm = other.mag;
            nm.sub(mag);
            mag = nm;
            neg = other.neg;
        }
        normalize();
    }

    std::string to_string() const {
        if (is_zero()) return "0";
        return neg ? "-" + mag.to_string() : mag.to_string();
    }
};

// ===========================
//  Closed-form sums
// ===========================

static inline i128 p0(u64 n) {
    return (i128)n;
}

static inline i128 p1(u64 n) {
    return (i128)n * (i128)(n - 1) / 2;
}

static inline i128 p2(u64 n) {
    return (i128)n * (i128)(n - 1) * (i128)(2 * n - 1) / 6;
}

static inline i128 p3(u64 n) {
    i128 s = (i128)n * (i128)(n - 1) / 2;
    return s * s;
}

static inline i128 p4(u64 n) {
    i128 nn = (i128)n;
    return nn * (nn - 1) * (2 * nn - 1) * (3 * nn * nn - 3 * nn - 1) / 30;
}

using Moments = std::array<i128, 10>;

// ===========================
//  Port from Python
// ===========================

static inline Moments affine_combine(const Moments& H, u64 n, u64 A, u64 B) {
    const i128 h01 = H[0], h11 = H[1], h21 = H[2], h31 = H[3];
    const i128 h02 = H[4], h12 = H[5], h22 = H[6], h03 = H[7], h13 = H[8], h04 = H[9];

    const i128 P0 = p0(n);
    const i128 P1 = p1(n);
    const i128 P2 = p2(n);
    const i128 P3 = p3(n);
    const i128 P4 = p4(n);

    const i128 Ai = (i128)A;
    const i128 Bi = (i128)B;

    const i128 A2 = Ai * Ai;
    const i128 A3 = A2 * Ai;
    const i128 A4 = A3 * Ai;

    if (B == 0) {
        return Moments{
            h01 + Ai * P1,
            h11 + Ai * P2,
            h21 + Ai * P3,
            h31 + Ai * P4,
            h02 + 2 * Ai * h11 + A2 * P2,
            h12 + 2 * Ai * h21 + A2 * P3,
            h22 + 2 * Ai * h31 + A2 * P4,
            h03 + 3 * Ai * h12 + 3 * A2 * h21 + A3 * P3,
            h13 + 3 * Ai * h22 + 3 * A2 * h31 + A3 * P4,
            h04 + 4 * Ai * h13 + 6 * A2 * h22 + 4 * A3 * h31 + A4 * P4,
        };
    }

    const i128 B2 = Bi * Bi;
    const i128 B3 = B2 * Bi;
    const i128 B4 = B3 * Bi;

    const i128 AB = Ai * Bi;
    const i128 A2B = A2 * Bi;
    const i128 AB2 = Ai * B2;

    return Moments{
        h01 + Ai * P1 + Bi * P0,
        h11 + Ai * P2 + Bi * P1,
        h21 + Ai * P3 + Bi * P2,
        h31 + Ai * P4 + Bi * P3,
        h02 + 2 * Ai * h11 + 2 * Bi * h01 + A2 * P2 + 2 * AB * P1 + B2 * P0,
        h12 + 2 * Ai * h21 + 2 * Bi * h11 + A2 * P3 + 2 * AB * P2 + B2 * P1,
        h22 + 2 * Ai * h31 + 2 * Bi * h21 + A2 * P4 + 2 * AB * P3 + B2 * P2,
        h03 + 3 * Ai * h12 + 3 * Bi * h02 + 3 * A2 * h21 + 6 * AB * h11 + 3 * B2 * h01
            + A3 * P3 + 3 * A2B * P2 + 3 * AB2 * P1 + B3 * P0,
        h13 + 3 * Ai * h22 + 3 * Bi * h12 + 3 * A2 * h31 + 6 * AB * h21 + 3 * B2 * h11
            + A3 * P4 + 3 * A2B * P3 + 3 * AB2 * P2 + B3 * P1,
        h04 + 4 * Ai * h13 + 4 * Bi * h03 + 6 * A2 * h22 + 12 * AB * h12 + 6 * B2 * h02
            + 4 * A3 * h31 + 12 * A2B * h21 + 12 * AB2 * h11 + 4 * B3 * h01
            + A4 * P4 + 4 * A3 * Bi * P3 + 6 * A2 * B2 * P2 + 4 * Ai * B3 * P1 + B4 * P0,
    };
}

static inline Moments recip_combine(const Moments& G, u64 n, u64 Y) {
    const i128 g01 = G[0], g11 = G[1], g21 = G[2], g31 = G[3];
    const i128 g02 = G[4], g12 = G[5], g22 = G[6], g03 = G[7], g13 = G[8], g04 = G[9];

    const i128 P0 = p0(n);
    const i128 P1 = p1(n);
    const i128 P2 = p2(n);
    const i128 P3 = p3(n);

    const i128 Q0 = p0(Y);
    const i128 Q1 = p1(Y);
    const i128 Q2 = p2(Y);
    const i128 Q3 = p3(Y);

    const i128 Yi = (i128)Y;
    const i128 Y2 = Yi * Yi;
    const i128 Y3 = Y2 * Yi;
    const i128 Y4 = Y3 * Yi;

    return Moments{
        P0 * Yi - Q0 - g01,
        (2 * P1 * Yi - g01 - g02) / 2,
        (6 * P2 * Yi - g01 - 3 * g02 - 2 * g03) / 6,
        (4 * P3 * Yi - g02 - 2 * g03 - g04) / 4,
        P0 * Y2 - Q0 - g01 - 2 * Q1 - 2 * g11,
        (2 * P1 * Y2 - g01 - g02 - 2 * g11 - 2 * g12) / 2,
        (6 * P2 * Y2 - g01 - 3 * g02 - 2 * g03 - 2 * g11 - 6 * g12 - 4 * g13) / 6,
        P0 * Y3 - Q0 - g01 - 3 * Q1 - 3 * g11 - 3 * Q2 - 3 * g21,
        (2 * P1 * Y3 - g01 - g02 - 3 * g11 - 3 * g12 - 3 * g21 - 3 * g22) / 2,
        P0 * Y4 - Q0 - g01 - 4 * Q1 - 4 * g11 - 6 * Q2 - 6 * g21 - 4 * Q3 - 4 * g31,
    };
}

static Moments solve_moments(u64 n, u64 m, u64 a, u64 b) {
    if (n == 0) {
        return Moments{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }

    if (a == 0) {
        i128 c = (i128)(b / m);
        i128 c2 = c * c;
        i128 c3 = c2 * c;
        i128 c4 = c3 * c;
        return Moments{
            p0(n) * c,
            p1(n) * c,
            p2(n) * c,
            p3(n) * c,
            p0(n) * c2,
            p1(n) * c2,
            p2(n) * c2,
            p0(n) * c3,
            p1(n) * c3,
            p0(n) * c4,
        };
    }

    if (a >= m || b >= m) {
        u64 A = a / m, a2 = a % m;
        u64 B = b / m, b2 = b % m;
        return affine_combine(solve_moments(n, m, a2, b2), n, A, B);
    }

    u64 Y = (u64)(((u128)a * (u128)(n - 1) + (u128)b) / (u128)m);
    if (Y == 0) {
        return Moments{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    }

    return recip_combine(solve_moments(Y, a, m, m - b - 1), n, Y);
}

static inline i128 rect_const_cap_sum(u64 M, u64 u, u64 y, u64 d, u64 x0, u64 x1, u64 T) {
    if (x0 > x1 || T == 0) return 0;

    i128 cnt = (i128)(x1 - x0 + 1);
    i128 sx = (i128)(x0 + x1) * cnt / 2;

    i128 sx2 = (i128)x1 * (x1 + 1) * (2 * x1 + 1) / 6
             - (x0 ? (i128)(x0 - 1) * x0 * (2 * x0 - 1) / 6 : 0);

    i128 st = (i128)T * (T + 1) / 2;
    i128 st2 = (i128)T * (T + 1) * (2 * T + 1) / 6;

    i128 Mp1 = (i128)M + 1;
    i128 a = Mp1 - (i128)u * y;
    i128 dd = (i128)d * d;

    return Mp1 * a * cnt * T
         - (i128)u * a * sx * T
         - (i128)d * y * a * cnt * st
         - (i128)d * Mp1 * sx * st
         + (i128)d * u * sx2 * st
         + dd * y * sx * st2;
}

static inline i128 linear_region_sum_from_x0(u64 M, u64 u, u64 y, u64 d, u64 x0) {
    i128 B = (i128)y * d;
    i128 alpha = (i128)M - (i128)u * x0 - B;
    if (alpha < 0) return 0;

    u64 S = (u64)(alpha / u);
    u64 rem = (u64)(alpha % u);

    Moments H = solve_moments(S + 1, (u64)B, u, rem);

    i128 h01 = H[0], h11 = H[1], h21 = H[2], h31 = H[3];
    i128 h02 = H[4], h12 = H[5], h22 = H[6], h03 = H[7], h13 = H[8];

    i128 h00 = p0(S + 1);
    i128 h10 = p1(S + 1);
    i128 h20 = p2(S + 1);

    i128 Si = (i128)S;
    i128 S2 = Si * Si;

    i128 U10 = Si * h00 - h10;
    i128 U11 = Si * h01 - h11;
    i128 U12 = Si * h02 - h12;
    i128 U13 = Si * h03 - h13;

    i128 U20 = S2 * h00 - 2 * Si * h10 + h20;
    i128 U21 = S2 * h01 - 2 * Si * h11 + h21;
    i128 U22 = S2 * h02 - 2 * Si * h12 + h22;

    i128 V01 = h01 + h00;
    i128 V02 = h02 + 2 * h01 + h00;
    i128 V03 = h03 + 3 * h02 + 3 * h01 + h00;

    i128 V11 = U11 + U10;
    i128 V12 = U12 + 2 * U11 + U10;
    i128 V13 = U13 + 3 * U12 + 3 * U11 + U10;

    i128 V21 = U21 + U20;
    i128 V22 = U22 + 2 * U21 + U20;

    i128 Mp1 = (i128)M + 1;
    i128 a = Mp1 - (i128)u * x0;
    i128 b = Mp1 - (i128)u * y;

    i128 c0 = a * b;
    i128 cr = -(i128)u * b;
    i128 ct = (i128)d * ((i128)u * x0 * x0 + (i128)u * y * y - Mp1 * (x0 + y));
    i128 crt = (i128)d * ((i128)2 * u * x0 - M - 1);
    i128 cr2t = (i128)d * u;

    i128 dd = (i128)d * d;
    i128 ctt = dd * x0 * y;
    i128 crtt = dd * y;

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

static inline i128 triangle_polynomial_sum(u64 M, u64 u, u64 y, u64 d) {
    u64 T = (u - 1) / d;
    if (T == 0) return 0;

    u64 q = M / u;
    i64 x_cap_signed = (i64)q - (i64)y;

    i128 total = 0;
    u64 x0;

    if (x_cap_signed >= (i64)y) {
        u64 x_cap = (u64)x_cap_signed;
        total += rect_const_cap_sum(M, u, y, d, y, x_cap, T);
        x0 = x_cap + 1;
    } else {
        x0 = y;
    }

    total += linear_region_sum_from_x0(M, u, y, d, x0);

    u64 M0 = (u64)(((i128)M - (i128)u * y) / ((i128)y * d));
    if (M0 > T) M0 = T;

    i128 diag = 0;
    if (M0 > 0) {
        i128 a = (i128)M + 1 - (i128)u * y;
        i128 s1 = (i128)M0 * (M0 + 1) / 2;
        i128 s2 = (i128)M0 * (M0 + 1) * (2 * M0 + 1) / 6;
        diag = a * a * M0 - 2 * (i128)d * y * a * s1 + (i128)d * d * y * y * s2;
    }

    return 4 * total - 2 * diag;
}

// ===========================
//  Prime generation / factors
// ===========================

static std::vector<u64> build_primes(u64 limit) {
    std::vector<bool> is_composite(limit + 1, false);
    std::vector<u64> primes;
    for (u64 i = 2; i <= limit; ++i) {
        if (!is_composite[i]) {
            primes.push_back(i);
            if (i * i <= limit) {
                for (u64 j = i * i; j <= limit; j += i) {
                    is_composite[j] = true;
                }
            }
        }
    }
    return primes;
}

static inline void distinct_prime_factors(
    u64 x,
    const std::vector<u64>& primes,
    std::vector<u64>& out
) {
    out.clear();
    u64 t = x;
    for (u64 p : primes) {
        u64 pp = p * p;
        if (pp > t) break;
        if (t % p == 0) {
            out.push_back(p);
            do {
                t /= p;
            } while (t % p == 0);
        }
    }
    if (t > 1) out.push_back(t);
}

template <class F>
static void enumerate_squarefree_divisors_rec(
    const std::vector<u64>& pf,
    size_t idx,
    u64 cur_d,
    int mu,
    F&& f
) {
    if (idx == pf.size()) {
        f(cur_d, mu);
        return;
    }
    enumerate_squarefree_divisors_rec(pf, idx + 1, cur_d, mu, f);
    enumerate_squarefree_divisors_rec(pf, idx + 1, cur_d * pf[idx], -mu, f);
}

// ===========================
//  Axis-aligned rectangles
// ===========================

static Uint256 count_axis_rectangles(u64 n) {
    u64 a = n;
    u64 b = n - 1;
    u64 c = n - 1;
    u64 d = 2 * n - 1;

    auto divide_once = [](u64& x, u64 v) -> bool {
        if (x % v == 0) {
            x /= v;
            return true;
        }
        return false;
    };

    if (!divide_once(a, 2) && !divide_once(b, 2) && !divide_once(c, 2)) {
        divide_once(d, 2);
    }
    if (!divide_once(a, 3) && !divide_once(b, 3) && !divide_once(c, 3)) {
        divide_once(d, 3);
    }

    Uint256 res(1);
    res.mul_u64(a);
    res.mul_u64(b);
    res.mul_u64(c);
    res.mul_u64(d);
    return res;
}

// ===========================
//  Parallel
// ===========================

struct SharedContext {
    u64 M;
    u64 max_u;
    u64 chunk_size;
    int num_threads;
    std::atomic<u64> next_u;
    const std::vector<u64>* primes;
};

struct ThreadResult {
    Int256 sum;
};

struct ThreadArg {
    SharedContext* ctx;
    ThreadResult* result;
    int thread_index;
};

static void pin_thread_if_possible(int thread_index) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu > 0) {
        CPU_SET(thread_index % ncpu, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#else
    (void)thread_index;
#endif
}

static void* worker_func(void* ptr) {
    ThreadArg* arg = (ThreadArg*)ptr;
    SharedContext* ctx = arg->ctx;
    ThreadResult* out = arg->result;

    pin_thread_if_possible(arg->thread_index);

    const u64 M = ctx->M;
    const std::vector<u64>& primes = *ctx->primes;
    const u64 chunk = ctx->chunk_size;

    std::vector<u64> pf;
    pf.reserve(16);

    Int256 local_sum;

    while (true) {
        u64 start = ctx->next_u.fetch_add(chunk, std::memory_order_relaxed);
        if (start > ctx->max_u) break;

        u64 end = start + chunk - 1;
        if (end > ctx->max_u) end = ctx->max_u;

        for (u64 u = start; u <= end; ++u) {
            if (u < 2) continue;

            distinct_prime_factors(u, primes, pf);

            std::vector<std::pair<u64, int>> divs;
            divs.reserve((size_t)1 << pf.size());

            enumerate_squarefree_divisors_rec(
                pf, 0, 1, +1,
                [&](u64 d, int mu) {
                    divs.emplace_back(d, mu);
                }
            );

            u64 ymax = M / u;
            for (u64 y = 1; y <= ymax; ++y) {
                for (const auto& it : divs) {
                    u64 d = it.first;
                    int mu = it.second;
                    i128 tri = triangle_polynomial_sum(M, u, y, d);
                    i128 term = (mu == 1) ? tri : -tri;
                    local_sum.add_i128(term);
                }
            }
        }
    }

    out->sum = local_sum;
    return nullptr;
}

static Int256 count_oblique_rectangles_parallel(u64 n, int num_threads, u64 chunk_size) {
    u64 M = n - 1;
    u64 lim = (u64)std::sqrt((long double)M) + 1;
    std::vector<u64> primes = build_primes(lim);

    SharedContext ctx;
    ctx.M = M;
    ctx.max_u = M;
    ctx.chunk_size = std::max<u64>(1, chunk_size);
    ctx.num_threads = num_threads;
    ctx.next_u.store(2);
    ctx.primes = &primes;

    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadResult> results(num_threads);
    std::vector<ThreadArg> args(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        args[i].ctx = &ctx;
        args[i].result = &results[i];
        args[i].thread_index = i;
        int rc = pthread_create(&threads[i], nullptr, worker_func, &args[i]);
        if (rc != 0) {
            std::cerr << "pthread_create failed, rc=" << rc << "\n";
            std::exit(1);
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    Int256 total;
    for (int i = 0; i < num_threads; ++i) {
        total.add(results[i].sum);
    }
    return total;
}

static Int256 count_all_rectangles_parallel(u64 n, int num_threads, u64 chunk_size) {
    Int256 total = count_oblique_rectangles_parallel(n, num_threads, chunk_size);

    Uint256 axis = count_axis_rectangles(n);
    Int256 axis_i;
    axis_i.neg = false;
    axis_i.mag = axis;

    total.add(axis_i);
    return total;
}

// ===========================
//  Main
// ===========================

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " k1 k2 threads chunk_size\n";
        std::cerr << "Example: " << argv[0] << " 8 12 16 64\n";
        return 1;
    }

    int k1 = std::atoi(argv[1]);
    int k2 = std::atoi(argv[2]);
    int threads = std::atoi(argv[3]);
    u64 chunk_size = (u64)std::strtoull(argv[4], nullptr, 10);

    if (k1 < 1 || k2 < k1 || k2 > 62) {
        std::cerr << "Invalid k1/k2\n";
        return 1;
    }
    if (threads <= 0) {
        std::cerr << "threads must be > 0\n";
        return 1;
    }
    if (chunk_size == 0) {
        std::cerr << "chunk_size must be > 0\n";
        return 1;
    }

    for (int k = k1; k <= k2; ++k) {
        if (k >= 63) {
            std::cerr << "k too large for uint64 shift\n";
            return 1;
        }

        u64 n = 1ULL << k;

        auto t0 = std::chrono::steady_clock::now();
        Int256 ans = count_all_rectangles_parallel(n, threads, chunk_size);
        auto t1 = std::chrono::steady_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "k=" << k
                  << " n=" << n
                  << " result=" << ans.to_string()
                  << " time_sec=" << sec
                  << "\n";
    }

    return 0;
}