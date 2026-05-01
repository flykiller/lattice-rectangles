#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#  include <intrin.h>
#endif

using std::cerr;
using std::cout;
using std::fixed;
using std::setprecision;
using std::size_t;
using std::string;
using std::vector;

using u64 = uint64_t;

static bool g_progress = false;
static constexpr u64 F1_BUILTIN_MOBIUS_BLOCK = 8ULL << 20;
static constexpr int F1_BUILTIN_MOBIUS_CUDA_TPB = 256;
static constexpr u64 F1_BUILTIN_MOBIUS_MARK_CHUNK = 4096;
static constexpr u64 F1_BUILTIN_MOBIUS_REDUCE_CHUNK = 4096;
static constexpr int F1_BUILTIN_CUDA_TPB = 128;
static constexpr int F1_BUILTIN_CUDA_LAUNCH_MINB = 2;
static constexpr int F1_BUILTIN_CUDA_BLOCKS_PER_SM = 8;
static constexpr int F1_BUILTIN_PROGRESS_MS = 1000;

#define HD __host__ __device__
#define DEV __device__

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t _e = (call);                                                       \
    if (_e != cudaSuccess) {                                                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e)                     \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";             \
        std::exit(1);                                                              \
    }                                                                              \
} while (0)

struct I192 {
    uint32_t w[6];

    HD I192() : w{0,0,0,0,0,0} {}

    HD static I192 from_u64(u64 x) {
        I192 r;
        r.w[0] = (uint32_t)x;
        r.w[1] = (uint32_t)(x >> 32);
        return r;
    }

    HD static I192 from_i64(int64_t x) {
        I192 r;
        u64 ux = (u64)x;
        r.w[0] = (uint32_t)ux;
        r.w[1] = (uint32_t)(ux >> 32);
        uint32_t fill = x < 0 ? 0xffffffffu : 0u;
        for (int i = 2; i < 6; ++i) r.w[i] = fill;
        return r;
    }

    HD bool neg() const { return (w[5] >> 31) != 0; }
    HD bool is_zero() const {
        uint32_t x = 0;
        #pragma unroll
        for (int i = 0; i < 6; ++i) x |= w[i];
        return x == 0;
    }

    HD I192 bitnot() const {
        I192 r;
        #pragma unroll
        for (int i = 0; i < 6; ++i) r.w[i] = ~w[i];
        return r;
    }

    HD I192 negated() const {
        I192 r = bitnot();
        uint64_t carry = 1;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            uint64_t cur = (uint64_t)r.w[i] + carry;
            r.w[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
        return r;
    }

    HD static I192 abs_twos(I192 a) { return a.neg() ? a.negated() : a; }

    HD I192 operator+(const I192& b) const {
        I192 r;
#if defined(__CUDA_ARCH__)

        asm volatile(
            "add.cc.u32  %0,  %6, %12;\n\t"
            "addc.cc.u32 %1,  %7, %13;\n\t"
            "addc.cc.u32 %2,  %8, %14;\n\t"
            "addc.cc.u32 %3,  %9, %15;\n\t"
            "addc.cc.u32 %4, %10, %16;\n\t"
            "addc.u32    %5, %11, %17;"
            : "=r"(r.w[0]), "=r"(r.w[1]), "=r"(r.w[2]),
              "=r"(r.w[3]), "=r"(r.w[4]), "=r"(r.w[5])
            : "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]),
              "r"(b.w[0]), "r"(b.w[1]), "r"(b.w[2]), "r"(b.w[3]), "r"(b.w[4]), "r"(b.w[5]));
#else
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            uint64_t cur = (uint64_t)w[i] + b.w[i] + carry;
            r.w[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
#endif
        return r;
    }

    HD I192 operator-(const I192& b) const {
        I192 r;
#if defined(__CUDA_ARCH__)
        asm volatile(
            "sub.cc.u32  %0,  %6, %12;\n\t"
            "subc.cc.u32 %1,  %7, %13;\n\t"
            "subc.cc.u32 %2,  %8, %14;\n\t"
            "subc.cc.u32 %3,  %9, %15;\n\t"
            "subc.cc.u32 %4, %10, %16;\n\t"
            "subc.u32    %5, %11, %17;"
            : "=r"(r.w[0]), "=r"(r.w[1]), "=r"(r.w[2]),
              "=r"(r.w[3]), "=r"(r.w[4]), "=r"(r.w[5])
            : "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]),
              "r"(b.w[0]), "r"(b.w[1]), "r"(b.w[2]), "r"(b.w[3]), "r"(b.w[4]), "r"(b.w[5]));
#else
        r = *this + b.negated();
#endif
        return r;
    }
    HD I192& operator+=(const I192& b) { *this = *this + b; return *this; }
    HD I192& operator-=(const I192& b) { *this = *this - b; return *this; }

    HD I192 mul(const I192& b) const {
        I192 r;
        for (int i = 0; i < 6; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j + i < 6; ++j) {
                uint64_t cur = (uint64_t)w[i] * (uint64_t)b.w[j]
                             + (uint64_t)r.w[i + j] + carry;
                r.w[i + j] = (uint32_t)cur;
                carry = cur >> 32;
            }
        }
        return r;
    }

    HD I192 mul_u32(uint32_t b) const {
        I192 r;
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            uint64_t cur = (uint64_t)w[i] * (uint64_t)b + carry;
            r.w[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
        return r;
    }

    HD I192 mul_u64(u64 b) const {
        I192 r;
        uint32_t b0 = (uint32_t)b;
        uint32_t b1 = (uint32_t)(b >> 32);
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            uint64_t cur = (uint64_t)w[i] * b0 + r.w[i] + carry;
            r.w[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
        carry = 0;
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            uint64_t cur = (uint64_t)w[i] * b1 + r.w[i + 1] + carry;
            r.w[i + 1] = (uint32_t)cur;
            carry = cur >> 32;
        }
        return r;
    }

    HD I192 mul_u128_low(const I192& b) const {

        if ((b.w[2] | b.w[3]) == 0u) {
            u64 bb = ((u64)b.w[1] << 32) | (u64)b.w[0];
            return mul_u64(bb);
        }

        I192 r;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t bj = b.w[j];
            if (bj == 0u) continue;
            uint64_t carry = 0;
            #pragma unroll
            for (int i = 0; i + j < 6; ++i) {
                uint64_t cur = (uint64_t)w[i] * (uint64_t)bj
                             + (uint64_t)r.w[i + j] + carry;
                r.w[i + j] = (uint32_t)cur;
                carry = cur >> 32;
            }
        }
        return r;
    }

    HD I192 mul_i128_low(const I192& b) const {
        if (!b.neg()) return mul_u128_low(b);
        return mul_u128_low(b.negated()).negated();
    }

    HD I192 div2() const {
        bool sg = neg();
        I192 a = abs_twos(*this), q;
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 5; i >= 0; --i) {
            uint32_t wi = a.w[i];
            q.w[i] = (wi >> 1) | (carry << 31);
            carry = wi & 1u;
        }
        return sg ? q.negated() : q;
    }

    HD static uint32_t div3_u32_magic(uint64_t x) {

        return (uint32_t)((x * 0xAAAAAAABull) >> 33);
    }

    HD I192 div3() const {
        bool sg = neg();
        I192 a = abs_twos(*this), q;
        uint32_t rem = 0;
        const uint32_t B_DIV_3 = 0x55555555u;
        #pragma unroll
        for (int i = 5; i >= 0; --i) {
            uint64_t folded = (uint64_t)a.w[i] + rem;
            uint32_t qlo = div3_u32_magic(folded);
            q.w[i] = rem * B_DIV_3 + qlo;
            rem = (uint32_t)(folded - (uint64_t)qlo * 3ull);
        }
        return sg ? q.negated() : q;
    }

    HD I192 div6() const { return div2().div3(); }

    uint32_t divmod_u32_inplace(uint32_t d) {
        uint64_t rem = 0;
        for (int i = 5; i >= 0; --i) {
            uint64_t cur = (rem << 32) | (uint64_t)w[i];
            w[i] = (uint32_t)(cur / d);
            rem = cur % d;
        }
        return (uint32_t)rem;
    }

    string str() const {
        if (is_zero()) return "0";
        bool sg = neg();
        I192 x = abs_twos(*this);
        vector<uint32_t> parts;
        while (!x.is_zero()) parts.push_back(x.divmod_u32_inplace(1000000000u));
        string s = sg ? "-" : "";
        s += std::to_string(parts.back());
        char buf[16];
        for (int i = (int)parts.size() - 2; i >= 0; --i) {
            std::snprintf(buf, sizeof(buf), "%09u", parts[(size_t)i]);
            s += buf;
        }
        return s;
    }
};

HD static inline I192 operator-(const I192& a) { return a.negated(); }
HD static inline I192 W64(u64 x) { return I192::from_u64(x); }
HD static inline I192 WI64(int64_t x) { return I192::from_i64(x); }

struct Six192 { I192 v[6]; };
struct Three192 { I192 a, b, c; };

static_assert(sizeof(I192) == 24, "I192 must be 192 bits");
static_assert(sizeof(Three192) == 72, "Three192 must be 3 x I192");

HD static inline Six192 Z6() { Six192 z; return z; }
HD static inline Three192 Z3() { Three192 z; return z; }
HD static inline Three192 add3(Three192 x, Three192 y) {
    Three192 r;
    r.a = x.a + y.a;
    r.b = x.b + y.b;
    r.c = x.c + y.c;
    return r;
}

HD static inline I192 s1_0w(u64 n) {
    u64 a = n, b = n - 1;
    if ((a & 1ull) == 0) a >>= 1; else b >>= 1;
    return W64(a).mul_u64(b);
}
HD static inline I192 s1_1w(u64 n) {
    u64 a = n, b = n + 1;
    if ((a & 1ull) == 0) a >>= 1; else b >>= 1;
    return W64(a).mul_u64(b);
}
HD static inline I192 s2_0w(u64 n) {
    u64 a = n, b = n - 1, c = 2 * n - 1;
    if ((a & 1ull) == 0) a >>= 1; else b >>= 1;
    if (a % 3ull == 0) a /= 3ull;
    else if (b % 3ull == 0) b /= 3ull;
    else c /= 3ull;
    return W64(a).mul_u64(b).mul_u64(c);
}
HD static inline I192 s2_1w(u64 n) {
    u64 a = n, b = n + 1, c = 2 * n + 1;
    if ((a & 1ull) == 0) a >>= 1; else b >>= 1;
    if (a % 3ull == 0) a /= 3ull;
    else if (b % 3ull == 0) b /= 3ull;
    else c /= 3ull;
    return W64(a).mul_u64(b).mul_u64(c);
}
HD static inline I192 square_i192(I192 x) { return x.mul_u128_low(x); }

HD static u64 isqrt_u64(u64 x) {
    if (x <= 1) return x;
    u64 r = x;
    u64 y = (r + x / r) >> 1;
    while (y < r) {
        r = y;
        y = (r + x / r) >> 1;
    }
    while ((r + 1) != 0 && (r + 1) <= x / (r + 1)) ++r;
    while (r > x / r) --r;
    return r;
}

DEV static inline void mul64wide_dev(u64 a, u64 b, u64& hi, u64& lo) {
    lo = a * b;
    hi = __umul64hi(a, b);
}

DEV static u64 mul_add_div_u64(u64 a, u64 b, u64 add, u64 d) {
    u64 hi, lo;
    mul64wide_dev(a, b, hi, lo);
    u64 old = lo;
    lo += add;
    if (lo < old) ++hi;

    if (hi == 0) return lo / d;

    u64 q = 0;
    u64 rem_hi = 0, rem_lo = 0;
    for (int bit = 127; bit >= 0; --bit) {
        rem_hi = (rem_hi << 1) | (rem_lo >> 63);
        rem_lo <<= 1;
        u64 nb = (bit >= 64) ? ((hi >> (bit - 64)) & 1ull) : ((lo >> bit) & 1ull);
        rem_lo |= nb;
        if (rem_hi || rem_lo >= d) {
            u64 before = rem_lo;
            rem_lo -= d;
            if (before < d) --rem_hi;
            if (bit < 64) q |= (1ull << bit);
        }
    }
    return q;
}

#ifndef F1_HCALC_STACK
#define F1_HCALC_STACK 80
#endif

struct HFrame {
    u64 n;
    u64 b;
    u64 y_kind;
    uint32_t m;
    uint32_t a;
};
static_assert(sizeof(HFrame) <= 32, "compact HFrame should stay <= 32 bytes");

static constexpr u64 HFRAME_KIND2 = 1ULL << 63;
static constexpr u64 HFRAME_Y_MASK = ~HFRAME_KIND2;

DEV static inline HFrame make_hframe_dev(u64 n, u64 m, u64 a, u64 b, u64 y, uint8_t kind) {
    HFrame fr{};
    fr.n = n;
    fr.b = b;
    fr.y_kind = (kind == 2 ? HFRAME_KIND2 : 0ULL) | (y & HFRAME_Y_MASK);
    fr.m = (uint32_t)m;
    fr.a = (uint32_t)a;
    return fr;
}

DEV static inline uint8_t hframe_kind_dev(const HFrame& fr) {
    return (fr.y_kind & HFRAME_KIND2) ? 2u : 1u;
}

DEV static inline u64 hframe_y_dev(const HFrame& fr) {
    return fr.y_kind & HFRAME_Y_MASK;
}

DEV static Six192 hcalc_dev(u64 n0, u64 m0, u64 a0, u64 b0) {

    HFrame st[F1_HCALC_STACK];
    int sp = 0;
    Six192 child = Z6();
    bool have_child = false;

    u64 n = n0, m = m0, a = a0, b = b0;

    for (;;) {
        if (!have_child) {
            if (n == 0) {
                child = Z6();
                have_child = true;
            } else if (a >= m || b >= m) {
                if (sp >= F1_HCALC_STACK) return Z6();
                st[sp++] = make_hframe_dev(n, m, a, b, 0, 1);
                u64 r = a % m, t = b % m;
                a = r;
                b = t;
                continue;
            } else if (a == 0) {
                child = Z6();
                have_child = true;
            } else {
                u64 y = mul_add_div_u64(a, n - 1, b, m);
                if (y == 0) {
                    child = Z6();
                    have_child = true;
                } else {
                    u64 nb = m + a - 1 - b;
                    if (sp >= F1_HCALC_STACK) return Z6();
                    st[sp++] = make_hframe_dev(n, m, a, b, y, 2);
                    u64 old_m = m, old_a = a;
                    n = y;
                    m = old_a;
                    a = old_m;
                    b = nb;
                    continue;
                }
            }
        }

        if (sp == 0) return child;

        HFrame fr = st[--sp];
        Six192 h = child;
        Six192 res = Z6();

        if (hframe_kind_dev(fr) == 1) {
            u64 q = fr.a / fr.m;
            u64 s = fr.b / fr.m;

            I192 h01=h.v[0], h11=h.v[1], h21=h.v[2], h02=h.v[3], h12=h.v[4], h03=h.v[5];
            I192 s0 = W64(fr.n);
            I192 S1 = s1_0w(fr.n);
            I192 S2 = s2_0w(fr.n);
            I192 S3 = square_i192(S1);

            res.v[0] = S1.mul_u64(q) + s0.mul_u64(s) + h01;
            res.v[1] = S2.mul_u64(q) + S1.mul_u64(s) + h11;
            res.v[2] = S3.mul_u64(q) + S2.mul_u64(s) + h21;
            res.v[3] = S2.mul_u64(q).mul_u64(q)
                     + S1.mul_u64(q).mul_u64(s).mul_u32(2)
                     + s0.mul_u64(s).mul_u64(s)
                     + h11.mul_u64(q).mul_u32(2)
                     + h01.mul_u64(s).mul_u32(2)
                     + h02;
            res.v[4] = S3.mul_u64(q).mul_u64(q)
                     + S2.mul_u64(q).mul_u64(s).mul_u32(2)
                     + S1.mul_u64(s).mul_u64(s)
                     + h21.mul_u64(q).mul_u32(2)
                     + h11.mul_u64(s).mul_u32(2)
                     + h12;
            res.v[5] = S3.mul_u64(q).mul_u64(q).mul_u64(q)
                     + S2.mul_u64(q).mul_u64(q).mul_u64(s).mul_u32(3)
                     + S1.mul_u64(q).mul_u64(s).mul_u64(s).mul_u32(3)
                     + s0.mul_u64(s).mul_u64(s).mul_u64(s)
                     + h21.mul_u64(q).mul_u64(q).mul_u32(3)
                     + h11.mul_u64(q).mul_u64(s).mul_u32(6)
                     + h01.mul_u64(s).mul_u64(s).mul_u32(3)
                     + h12.mul_u64(q).mul_u32(3)
                     + h02.mul_u64(s).mul_u32(3)
                     + h03;
        } else {
            u64 y = hframe_y_dev(fr);
            I192 g01=h.v[0], g11=h.v[1], g21=h.v[2], g02=h.v[3], g12=h.v[4], g03=h.v[5];
            I192 T1 = s1_0w(fr.n);
            I192 T2 = s2_0w(fr.n);

            res.v[0] = W64(fr.n).mul_u64(y) - g01;
            res.v[1] = T1.mul_u64(y) - (g02 - g01).div2();
            res.v[2] = T2.mul_u64(y) - (g03.mul_u32(2) - g02.mul_u32(3) + g01).div6();
            res.v[3] = W64(fr.n).mul_u64(y).mul_u64(y) - g11.mul_u32(2) - g01;
            res.v[4] = T1.mul_u64(y).mul_u64(y) - g12 - g02.div2() + g11 + g01.div2();
            res.v[5] = W64(fr.n).mul_u64(y).mul_u64(y).mul_u64(y) - g21.mul_u32(3) - g11.mul_u32(3) - g01;
        }

        child = res;
        have_child = true;
    }
}

DEV static Six192 prefix_moments_dev(u64 Y, u64 N, u64 coef, u64 mod) {
    if (Y == 0) return Z6();
    u64 b = N - coef * Y;
    Six192 h = hcalc_dev(Y, mod, coef, b);
    I192 h01=h.v[0], h11=h.v[1], h21=h.v[2], h02=h.v[3], h12=h.v[4], h03=h.v[5];

    I192 p1 = s1_1w(Y);
    I192 p2 = s2_1w(Y);
    I192 p3 = square_i192(p1);
    I192 p1p0 = p1 - W64(Y);
    I192 p2p1 = p2 - p1;
    I192 p3p2 = p3 - p2;

    I192 s11 = h01.mul_u64(Y) - h11;
    I192 s12 = h02.mul_u64(Y) - h12;

    Six192 res = Z6();
    res.v[0] = h01 - p1p0;
    res.v[1] = (h02 + h01 - p2p1).div2();
    res.v[2] = s11 - p2p1;
    res.v[3] = (h03.mul_u32(2) + h02.mul_u32(3) + h01
              - (p3.mul_u32(2) - p2.mul_u32(3) + p1)).div6();
    res.v[4] = (s12 + s11 - p3p2).div2();
    res.v[5] = h01.mul_u64(Y).mul_u64(Y) - h11.mul_u64(Y).mul_u32(2) + h21 - p3p2;
    return res;
}

DEV static Six192 capped_moments_dev(u64 Y, u64 cap) {
    if (Y == 0) return Z6();

    I192 c1 = s1_1w(cap);
    I192 c2 = s2_1w(cap);
    I192 S1 = s1_1w(Y);
    I192 S2 = s2_1w(Y);
    I192 S3 = square_i192(S1);
    I192 s1s0 = S1 - W64(Y);
    I192 s2s1 = S2 - S1;
    I192 s3s2 = S3 - S2;

    Six192 res = Z6();
    res.v[0] = W64(cap).mul_u64(Y) - s1s0;
    res.v[1] = c1.mul_u64(Y) - s2s1.div2();
    res.v[2] = S1.mul_u64(cap) - s2s1;
    res.v[3] = c2.mul_u64(Y) - (S3.mul_u32(2) - S2.mul_u32(3) + S1).div6();
    res.v[4] = c1.mul_u128_low(S1) - s3s2.div2();
    res.v[5] = S2.mul_u64(cap) - s3s2;
    return res;
}

DEV static void add_ab_pair_dev(Three192& acc, u64 N, u64 cap, u64 c, u64 a) {
    u64 b = c - a;
    u64 Y = N / c;
    if (Y == 0) return;

    Six192 mm = prefix_moments_dev(Y, N, b, a);
    I192 m00=mm.v[0], m10=mm.v[1], m01=mm.v[2], m20=mm.v[3], m11=mm.v[4], m02=mm.v[5];

    u64 aa64 = a * a;
    u64 bb64 = b * b;
    u64 pq64 = a * b;
    u64 c2_64 = c * c;
    I192 sy1 = s1_1w(Y);
    I192 sy2 = s2_1w(Y);

    I192 A = m00.mul_u32(8) - W64(Y).mul_u32(6);
    I192 B = (m10 + m01).mul_u64(c).mul_u32(8) - sy1.mul_u64(c).mul_u32(12);
    I192 D = ((m20 + m02).mul_u64(pq64) + m11.mul_u64(aa64 + bb64)).mul_u32(8)
            - sy2.mul_u64(c2_64).mul_u32(6);

    u64 Yi = cap < Y ? cap : Y;
    if (Yi > 0) {
        u64 Y0 = 0;
        u64 a_cap = a * cap;
        if (N >= a_cap) {
            u64 q = (N - a_cap) / b;
            Y0 = Yi < q ? Yi : q;
        }

        Six192 tt = capped_moments_dev(Y0, cap);
        I192 t00=tt.v[0], t10=tt.v[1], t01=tt.v[2], t20=tt.v[3], t11=tt.v[4], t02=tt.v[5];

        if (Y0 < Yi) {
            Six192 am = (Yi == Y) ? mm : prefix_moments_dev(Yi, N, b, a);
            I192 a00=am.v[0], a10=am.v[1], a01=am.v[2], a20=am.v[3], a11=am.v[4], a02=am.v[5];
            if (Y0 > 0) {
                Six192 bm = prefix_moments_dev(Y0, N, b, a);
                a00 -= bm.v[0]; a10 -= bm.v[1]; a01 -= bm.v[2];
                a20 -= bm.v[3]; a11 -= bm.v[4]; a02 -= bm.v[5];
            }
            t00 += a00; t10 += a10; t01 += a01;
            t20 += a20; t11 += a11; t02 += a02;
        }

        I192 si1 = s1_1w(Yi);
        I192 si2 = s2_1w(Yi);
        A -= t00.mul_u32(4) - W64(Yi).mul_u32(2);
        B -= (t10 + t01).mul_u64(c).mul_u32(4) - si1.mul_u64(c).mul_u32(4);
        D -= ((t20 + t02).mul_u64(pq64) + t11.mul_u64(aa64 + bb64)).mul_u32(4)
           - si2.mul_u64(c2_64).mul_u32(2);
    }

    acc.a += A;
    acc.b += B;
    acc.c += D;
}

DEV static void add_x_value_dev(Three192& acc, u64 N, u64 x) {
    u64 M = N / x;
    if (M < 3) return;

    u64 K = M / 2;
    u64 L = (M - 1) / 2;
    I192 K1=s1_1w(K), K2=s2_1w(K), K3=square_i192(K1);
    I192 L1=s1_1w(L), L2=s2_1w(L), L3=square_i192(L1);
    u64 xx = x * x;

    acc.a += ((K1 - W64(K)) + L1).mul_u32(2);
    acc.b += ((K2 - K1).mul_u32(2) + L2.mul_u32(2) + L1).mul_u64(x).mul_u32(4);
    acc.c += ((K3 - K2).mul_u32(4) + L3.mul_u32(4) + L2.mul_u32(4) + L1).mul_u64(xx).mul_u32(2);
}

struct LayerInfo {
    u64 N;
    u64 cap;
    I192 s0;
    I192 s1;
    I192 s2;
};

struct WorkItem {
    uint32_t layer;
    uint8_t kind;
    uint8_t pad[3];
    u64 lo;
    u64 hi;
    u64 a_lo;
    u64 a_hi;
    u64 cost;
};

struct GpuStats {
    unsigned long long next_item;
    unsigned long long done_items;
    unsigned long long done_cost;
};

DEV static inline Three192 warp_reduce_sum3(Three192 v) {
    const unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        Three192 o;
        #pragma unroll
        for (int i = 0; i < 6; ++i) o.a.w[i] = __shfl_down_sync(mask, v.a.w[i], off);
        #pragma unroll
        for (int i = 0; i < 6; ++i) o.b.w[i] = __shfl_down_sync(mask, v.b.w[i], off);
        #pragma unroll
        for (int i = 0; i < 6; ++i) o.c.w[i] = __shfl_down_sync(mask, v.c.w[i], off);
        v = add3(v, o);
    }
    return v;
}

template<int TPB, int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(TPB, MIN_BLOCKS_PER_SM)
void f1_work_kernel_persistent(const LayerInfo* __restrict__ layers,
                               const WorkItem* __restrict__ work,
                               Three192* __restrict__ out,
                               GpuStats* __restrict__ stats,
                               int work_count,
                               int track_progress_cost) {
    extern __shared__ __align__(16) unsigned char smem[];
    Three192* sh = reinterpret_cast<Three192*>(smem);
    __shared__ unsigned long long sh_item;

    int tid = threadIdx.x;

    for (;;) {
        if (tid == 0) sh_item = atomicAdd(&stats->next_item, 1ULL);
        __syncthreads();

        unsigned long long item = sh_item;
        if (item >= (unsigned long long)work_count) break;

        Three192 local = Z3();
        WorkItem w = work[item];
        LayerInfo L = layers[w.layer];

        if (w.kind == 2) {
            for (u64 c = w.lo; c < w.hi; ++c) {
                u64 Y = L.N / c;
                if (Y == 0) break;

                u64 alo = c / 2 + 1;
                if (alo < 2) alo = 2;
                u64 ahi = c - 1;
                if (ahi > L.cap) ahi = L.cap;
                if (w.a_hi != 0 && w.hi == w.lo + 1) {
                    if (alo < w.a_lo) alo = w.a_lo;
                    if (ahi + 1 > w.a_hi) ahi = w.a_hi - 1;
                }
                if (ahi < alo) continue;

                for (u64 a = alo + (u64)tid; a <= ahi; a += (u64)blockDim.x) {
                    add_ab_pair_dev(local, L.N, L.cap, c, a);
                }
            }
        } else {
            for (u64 x = w.lo + (u64)tid; x < w.hi; x += (u64)blockDim.x) {
                add_x_value_dev(local, L.N, x);
            }
        }

        Three192 wsum = warp_reduce_sum3(local);
        int lane = tid & 31;
        int warp = tid >> 5;
        int num_warps = (blockDim.x + 31) >> 5;

        if (lane == 0) sh[warp] = wsum;
        __syncthreads();

        if (warp == 0) {
            Three192 bsum = Z3();
            if (lane < num_warps) bsum = sh[lane];
            bsum = warp_reduce_sum3(bsum);
            if (lane == 0) {
                out[item] = bsum;
                if (track_progress_cost) {
                    atomicAdd(&stats->done_items, 1ULL);
                    atomicAdd(&stats->done_cost, (unsigned long long)w.cost);
                }
            }
        }
        __syncthreads();
    }

}

static int bitlen_u64(u64 x) {
    if (!x) return 0;
#ifdef _MSC_VER
    unsigned long idx = 0;
    _BitScanReverse64(&idx, x);
    return (int)idx + 1;
#else
    return 64 - __builtin_clzll(x);
#endif
}

static bool fast_ab_pair_acc64_ok(u64 c, u64 Y) {
    if (Y == 0) return true;
    int bc = bitlen_u64(c), bY = bitlen_u64(Y);
    return 4*bY + 12 < 126 && 2*bc + 4*bY + 20 < 126;
}

static bool fast_x_acc64_ok(u64 x, u64 M) {
    if (M < 3) return true;
    int bx = bitlen_u64(x), bM = bitlen_u64(M);
    return 4*bM + 10 < 126 && 2*bx + 4*bM + 16 < 126;
}

static vector<uint32_t> simple_primes_upto(u64 limit64) {
    size_t limit = (size_t)limit64;
    vector<uint8_t> comp(limit + 1, 0);
    vector<uint32_t> primes;
    if (limit >= 2) primes.reserve(limit / 10 + 16);
    for (size_t i = 2; i <= limit; ++i) {
        if (!comp[i]) {
            primes.push_back((uint32_t)i);
            if (i * i <= limit) {
                for (size_t j = i * i; j <= limit; j += i) comp[j] = 1;
            }
        }
    }
    return primes;
}

struct LayerRequest { u64 N, cap, right; };

static void format_duration_compact(double seconds, char* buf, size_t n) {
    if (!(seconds >= 0.0) || seconds > 365.0 * 24.0 * 3600.0) {
        std::snprintf(buf, n, "?");
        return;
    }
    unsigned long long s = (unsigned long long)(seconds + 0.5);
    unsigned long long h = s / 3600ULL;
    unsigned long long m = (s / 60ULL) % 60ULL;
    unsigned long long sec = s % 60ULL;
    if (h) std::snprintf(buf, n, "%lluh%02llum%02llus", h, m, sec);
    else if (m) std::snprintf(buf, n, "%llum%02llus", m, sec);
    else std::snprintf(buf, n, "%llus", sec);
}

static void format_elapsed_total(double elapsed, double pct, bool final_line,
                                 char* buf, size_t n) {
    char elapsed_buf[32], total_buf[32];
    format_duration_compact(elapsed, elapsed_buf, sizeof(elapsed_buf));
    if (final_line || pct >= 99.999) {
        format_duration_compact(elapsed, total_buf, sizeof(total_buf));
    } else if (pct > 0.01 && elapsed > 0.0) {
        double total_est = elapsed * 100.0 / pct;
        format_duration_compact(total_est, total_buf, sizeof(total_buf));
    } else {
        std::snprintf(total_buf, sizeof(total_buf), "?");
    }
    std::snprintf(buf, n, "%s/%s", elapsed_buf, total_buf);
}

static void progress_overwrite(const char* line, bool final_line) {

    static const char clear_line[] =
        "                                                                              ";
    std::fprintf(stderr, "\r%s\r%s", clear_line, line ? line : "");
    if (final_line) std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

static void format_progress_bar(double pct, char* buf, size_t n) {
    const int width = 24;
    if (!buf || n < (size_t)width + 3) return;
    if (pct < 0.0) pct = 0.0;
    if (pct > 100.0) pct = 100.0;
    int filled = (int)((pct * width / 100.0) + 0.5);
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;
    size_t pos = 0;
    buf[pos++] = '[';
    for (int i = 0; i < width; ++i) buf[pos++] = (i < filled) ? '#' : '-';
    buf[pos++] = ']';
    buf[pos] = '\0';
}

static void print_stage_progress_line(const char* stage,
                                      u64 done, u64 total,
                                      std::chrono::steady_clock::time_point t0,
                                      bool final_line) {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - t0).count();
    double pct = total ? (100.0 * (double)done / (double)total) : 100.0;
    if (pct > 100.0) pct = 100.0;

    char time_buf[64], bar_buf[32], line[256];
    format_progress_bar(pct, bar_buf, sizeof(bar_buf));
    format_elapsed_total(elapsed, pct, final_line || done >= total, time_buf, sizeof(time_buf));
    std::snprintf(line, sizeof(line), "# %-6s %6.1f%% %s time=%s",
                  stage ? stage : "stage", pct, bar_buf, time_buf);
    progress_overwrite(line, final_line);
}

struct MobiusMarkTask {
    u64 first_idx;
    u64 stride;
    u64 count;
    uint32_t p;
    uint32_t kind;
};

struct MobiusReduceTask {
    u64 first_idx;
    u64 end_idx;
    uint32_t interval;
    uint32_t pad;
};

DEV static inline void atomic_mul_u64_dev(u64* addr, u64 v) {
    unsigned long long* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *p;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, assumed * (unsigned long long)v);
    } while (old != assumed);
}

__global__ void mobius_init_kernel(u64* __restrict__ prod,
                                   uint32_t* __restrict__ flags,
                                   size_t len) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t step = (size_t)blockDim.x * (size_t)gridDim.x;
    for (; i < len; i += step) {
        prod[i] = 1;
        flags[i] = 0;
    }
}

__global__ void mobius_mark_kernel(const MobiusMarkTask* __restrict__ tasks,
                                   u64* __restrict__ prod,
                                   uint32_t* __restrict__ flags) {
    int ti = (int)blockIdx.x;
    MobiusMarkTask t = tasks[ti];
    for (u64 j = (u64)threadIdx.x; j < t.count; j += (u64)blockDim.x) {
        u64 idx = t.first_idx + j * t.stride;
        if (t.kind == 0) {
            atomicXor(&flags[idx], 1u);
            atomic_mul_u64_dev(&prod[idx], (u64)t.p);
        } else {
            atomicOr(&flags[idx], 2u);
        }
    }
}

__global__ void mobius_reduce_kernel(u64 L,
                                     const u64* __restrict__ prod,
                                     const uint32_t* __restrict__ flags,
                                     const MobiusReduceTask* __restrict__ tasks,
                                     Three192* __restrict__ partial) {
    int ti = (int)blockIdx.x;

    extern __shared__ __align__(16) unsigned char smem[];
    Three192* sh = reinterpret_cast<Three192*>(smem);

    MobiusReduceTask t = tasks[ti];
    Three192 local = Z3();
    for (u64 idx = t.first_idx + (u64)threadIdx.x; idx < t.end_idx; idx += (u64)blockDim.x) {
        uint32_t f = flags[idx];
        if ((f & 2u) == 0u) {
            u64 x = L + idx;
            bool odd = (f & 1u) != 0u;
            u64 pr = prod[idx];

            if (pr != 0 && x / pr > 1) odd = !odd;
            I192 sm = WI64(odd ? -1 : 1);
            local.a += sm;
            I192 sx = sm.mul_u64(x);
            local.b += sx;
            local.c += sx.mul_u64(x);
        }
    }

    Three192 wsum = warp_reduce_sum3(local);
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int num_warps = ((int)blockDim.x + 31) >> 5;
    if (lane == 0) sh[warp] = wsum;
    __syncthreads();

    if (warp == 0) {
        Three192 bsum = Z3();
        if (lane < num_warps) bsum = sh[lane];
        bsum = warp_reduce_sum3(bsum);
        if (lane == 0) partial[ti] = bsum;
    }
}

template<typename T>
static void cuda_ensure_capacity(T*& ptr, size_t& cap, size_t need) {
    if (need <= cap) return;
    if (ptr) CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
    if (need) CUDA_CHECK(cudaMalloc((void**)&ptr, need * sizeof(T)));
    cap = need;
}

static inline u64 ceil_mul_u64(u64 x, u64 d) {
    u64 q = x / d;
    if (x % d) ++q;
    return q * d;
}

static vector<LayerInfo> build_layers_with_segmented_mobius_cuda(u64 n, double& sieve_sec) {
    vector<LayerRequest> req;
    req.reserve((size_t)(2 * isqrt_u64(n) + 8));

    for (u64 left = 1; left <= n; ) {
        u64 N = n / left;
        u64 right = n / N;
        if (N >= 2) req.push_back({N, isqrt_u64(N), right});
        left = right + 1;
    }

    vector<LayerInfo> layers;
    layers.reserve(req.size());
    if (req.empty()) {
        sieve_sec = 0.0;
        return layers;
    }

    const u64 max_needed = req.back().right;
    const u64 block_size = F1_BUILTIN_MOBIUS_BLOCK;
    const u64 prime_limit = isqrt_u64(max_needed);
    const int tpb = F1_BUILTIN_MOBIUS_CUDA_TPB;
    const u64 mark_chunk = F1_BUILTIN_MOBIUS_MARK_CHUNK;
    const u64 reduce_chunk = F1_BUILTIN_MOBIUS_REDUCE_CHUNK;

    auto t0 = std::chrono::steady_clock::now();
    vector<uint32_t> primes = simple_primes_upto(prime_limit);
    int progress_ms = F1_BUILTIN_PROGRESS_MS;
    auto next_progress = t0;
    if (g_progress) {
        print_stage_progress_line("mobius", 0, max_needed, t0, false);
        next_progress = t0 + std::chrono::milliseconds(progress_ms);
    }

    u64* d_prod = nullptr;
    uint32_t* d_flags = nullptr;
    MobiusMarkTask* d_mark = nullptr;
    MobiusReduceTask* d_reduce = nullptr;
    Three192* d_partial = nullptr;
    size_t prod_cap = 0, flags_cap = 0, mark_cap = 0, reduce_cap = 0, partial_cap = 0;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    I192 cur0, cur1, cur2;
    I192 prev0, prev1, prev2;
    size_t qi = 0;

    vector<MobiusMarkTask> mark_tasks;
    vector<MobiusReduceTask> reduce_tasks;
    vector<Three192> partials;
    vector<Three192> interval_sums;
    vector<size_t> interval_req;

    const size_t NO_REQ = std::numeric_limits<size_t>::max();

    for (u64 L = 1; L <= max_needed; ) {
        u64 R = L + block_size - 1;
        if (R < L || R > max_needed) R = max_needed;
        size_t len = (size_t)(R - L + 1);

        cuda_ensure_capacity(d_prod, prod_cap, len);
        cuda_ensure_capacity(d_flags, flags_cap, len);

        int init_blocks = (int)((len + (size_t)tpb - 1) / (size_t)tpb);
        if (init_blocks < 1) init_blocks = 1;
        if (init_blocks > 65535) init_blocks = 65535;
        mobius_init_kernel<<<init_blocks, tpb, 0, stream>>>(d_prod, d_flags, len);
        CUDA_CHECK(cudaGetLastError());

        mark_tasks.clear();
        mark_tasks.reserve(primes.size() * 2);
        for (uint32_t pp : primes) {
            u64 p = (u64)pp;
            if (p * p > R) break;

            u64 start = ceil_mul_u64(L, p);
            if (start <= R) {
                u64 total = (R - start) / p + 1;
                for (u64 off = 0; off < total; ) {
                    u64 cnt = total - off;
                    if (cnt > mark_chunk) cnt = mark_chunk;
                    MobiusMarkTask mt{};
                    mt.first_idx = (start - L) + off * p;
                    mt.stride = p;
                    mt.count = cnt;
                    mt.p = pp;
                    mt.kind = 0;
                    mark_tasks.push_back(mt);
                    off += cnt;
                }
            }

            u64 p2 = p * p;
            u64 start2 = ceil_mul_u64(L, p2);
            if (start2 <= R) {
                u64 total = (R - start2) / p2 + 1;
                for (u64 off = 0; off < total; ) {
                    u64 cnt = total - off;
                    if (cnt > mark_chunk) cnt = mark_chunk;
                    MobiusMarkTask mt{};
                    mt.first_idx = (start2 - L) + off * p2;
                    mt.stride = p2;
                    mt.count = cnt;
                    mt.p = pp;
                    mt.kind = 1;
                    mark_tasks.push_back(mt);
                    off += cnt;
                }
            }
        }

        if (!mark_tasks.empty()) {
            if (mark_tasks.size() > (size_t)std::numeric_limits<int>::max()) {
                std::cerr << "too many Mobius CUDA mark tasks: " << mark_tasks.size() << "\n";
                std::exit(1);
            }
            cuda_ensure_capacity(d_mark, mark_cap, mark_tasks.size());
            CUDA_CHECK(cudaMemcpyAsync(d_mark, mark_tasks.data(), mark_tasks.size() * sizeof(MobiusMarkTask), cudaMemcpyHostToDevice, stream));
            mobius_mark_kernel<<<(unsigned)mark_tasks.size(), tpb, 0, stream>>>(d_mark, d_prod, d_flags);
            CUDA_CHECK(cudaGetLastError());
        }

        interval_req.clear();
        reduce_tasks.clear();
        u64 segL = L;
        size_t qj = qi;
        while (qj < req.size() && req[qj].right <= R) {
            u64 segR = req[qj].right;
            if (segL <= segR) {
                if (interval_req.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
                    std::cerr << "too many Mobius intervals in one block: " << interval_req.size() << "\n";
                    std::exit(1);
                }
                uint32_t interval = (uint32_t)interval_req.size();
                interval_req.push_back(qj);
                u64 first = segL - L;
                u64 count = segR - segL + 1;
                for (u64 off = 0; off < count; ) {
                    u64 cnt = count - off;
                    if (cnt > reduce_chunk) cnt = reduce_chunk;
                    MobiusReduceTask rt{};
                    rt.first_idx = first + off;
                    rt.end_idx = first + off + cnt;
                    rt.interval = interval;
                    reduce_tasks.push_back(rt);
                    off += cnt;
                }
                segL = segR + 1;
            }
            ++qj;
        }
        if (segL <= R) {
            if (interval_req.size() > (size_t)std::numeric_limits<uint32_t>::max()) {
                std::cerr << "too many Mobius intervals in one block: " << interval_req.size() << "\n";
                std::exit(1);
            }
            uint32_t interval = (uint32_t)interval_req.size();
            interval_req.push_back(NO_REQ);
            u64 first = segL - L;
            u64 count = R - segL + 1;
            for (u64 off = 0; off < count; ) {
                u64 cnt = count - off;
                if (cnt > reduce_chunk) cnt = reduce_chunk;
                MobiusReduceTask rt{};
                rt.first_idx = first + off;
                rt.end_idx = first + off + cnt;
                rt.interval = interval;
                reduce_tasks.push_back(rt);
                off += cnt;
            }
        }

        if (!reduce_tasks.empty()) {
            if (reduce_tasks.size() > (size_t)std::numeric_limits<int>::max()) {
                std::cerr << "too many Mobius CUDA reduce tasks: " << reduce_tasks.size() << "\n";
                std::exit(1);
            }
            cuda_ensure_capacity(d_reduce, reduce_cap, reduce_tasks.size());
            cuda_ensure_capacity(d_partial, partial_cap, reduce_tasks.size());
            CUDA_CHECK(cudaMemcpyAsync(d_reduce, reduce_tasks.data(), reduce_tasks.size() * sizeof(MobiusReduceTask), cudaMemcpyHostToDevice, stream));
            size_t shared_bytes = (size_t)((tpb + 31) / 32) * sizeof(Three192);
            mobius_reduce_kernel<<<(unsigned)reduce_tasks.size(), tpb, shared_bytes, stream>>>(L, d_prod, d_flags, d_reduce, d_partial);
            CUDA_CHECK(cudaGetLastError());

            partials.resize(reduce_tasks.size());
            CUDA_CHECK(cudaMemcpyAsync(partials.data(), d_partial, partials.size() * sizeof(Three192), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            interval_sums.assign(interval_req.size(), Z3());
            for (size_t i = 0; i < partials.size(); ++i) {
                uint32_t iv = reduce_tasks[i].interval;
                interval_sums[iv].a += partials[i].a;
                interval_sums[iv].b += partials[i].b;
                interval_sums[iv].c += partials[i].c;
            }

            for (size_t iv = 0; iv < interval_req.size(); ++iv) {
                cur0 += interval_sums[iv].a;
                cur1 += interval_sums[iv].b;
                cur2 += interval_sums[iv].c;

                size_t rq = interval_req[iv];
                if (rq != NO_REQ) {
                    I192 ds0 = cur0 - prev0;
                    I192 ds1 = cur1 - prev1;
                    I192 ds2 = cur2 - prev2;

                    if (!ds0.is_zero() || !ds1.is_zero() || !ds2.is_zero()) {
                        layers.push_back({req[rq].N, req[rq].cap, ds0, ds1, ds2});
                    }
                    prev0 = cur0; prev1 = cur1; prev2 = cur2;
                    ++qi;
                }
            }
        } else {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        if (g_progress) {
            auto now = std::chrono::steady_clock::now();
            if (now >= next_progress || R >= max_needed) {
                print_stage_progress_line("mobius", R, max_needed, t0, R >= max_needed);
                next_progress = now + std::chrono::milliseconds(progress_ms);
            }
        }

        L = R + 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_prod) CUDA_CHECK(cudaFree(d_prod));
    if (d_flags) CUDA_CHECK(cudaFree(d_flags));
    if (d_mark) CUDA_CHECK(cudaFree(d_mark));
    if (d_reduce) CUDA_CHECK(cudaFree(d_reduce));
    if (d_partial) CUDA_CHECK(cudaFree(d_partial));

    auto t1 = std::chrono::steady_clock::now();
    sieve_sec = std::chrono::duration<double>(t1 - t0).count();
    return layers;
}

static I192 final_tail(u64 n) {
    u64 a = n, b = n - 1, c = n - 1, d = 2 * n - 1;
    if ((a & 1ull) == 0) a >>= 1; else b >>= 1;
    if (a % 3ull == 0) a /= 3ull;
    else if (b % 3ull == 0) b /= 3ull;
    else d /= 3ull;
    return I192::from_u64(a).mul_u64(b).mul_u64(c).mul_u64(d);
}

static inline u64 sat_add_u64(u64 a, u64 b) {
    u64 r = a + b;
    return r < a ? std::numeric_limits<u64>::max() : r;
}

static inline u64 sat_mul_u64(u64 a, u64 b) {
    if (a && b > std::numeric_limits<u64>::max() / a) return std::numeric_limits<u64>::max();
    return a * b;
}

static inline u64 diag_pair_count(u64 c, u64 cap) {
    u64 alo = c / 2 + 1;
    if (alo < 2) alo = 2;
    u64 ahi = c - 1;
    if (ahi > cap) ahi = cap;
    return ahi >= alo ? (ahi - alo + 1) : 0;
}

static inline u64 ab_diag_cost_est(u64 N, u64 cap, u64 c) {
    u64 pairs = diag_pair_count(c, cap);
    if (!pairs) return 0;
    u64 Y = N / c;

    u64 mult = fast_ab_pair_acc64_ok(c, Y) ? 1ULL : 4ULL;
    return sat_mul_u64(pairs, mult);
}

static inline u64 x_cost_est(u64 N, u64 x) {
    u64 M = N / x;
    if (M < 3) return 0;

    return fast_x_acc64_ok(x, M) ? 1ULL : 4ULL;
}

static void push_ab_diag_work(vector<WorkItem>& work, uint32_t layer_idx,
                              u64 c_lo, u64 c_hi, u64 a_lo, u64 a_hi, u64 cost) {
    WorkItem w{};
    w.layer = layer_idx;
    w.kind = 2;
    w.lo = c_lo;
    w.hi = c_hi;
    w.a_lo = a_lo;
    w.a_hi = a_hi;
    w.cost = cost ? cost : 1;
    work.push_back(w);
}

struct WorkSplitTargets {
    u64 ab = 2048;
    u64 x = 512;
};

static u64 clamp_target_u64(long double v, u64 minv) {
    if (!(v > 0.0L)) return minv;
    long double mx = (long double)std::numeric_limits<u64>::max();
    if (v >= mx) return std::numeric_limits<u64>::max();
    u64 r = (u64)(v + 0.5L);
    if (r < minv) r = minv;
    return r;
}

static inline bool should_split_ab_diag(u64 cost, u64 pairs, u64 target_ab);
static inline u64 ab_diag_split_step(u64 cost, u64 pairs, u64 target_ab);

static u64 fixed_scaled_target_for_n(u64 base_default, u64 minv, u64 n_global) {
    const u64 BASE_N = 1ULL << 30;
    u64 scale = 1;
    if (n_global > BASE_N) {
        scale = (n_global + BASE_N - 1) / BASE_N;
        if (scale < 1) scale = 1;
        if (scale > (1ULL << 24)) scale = (1ULL << 24);
    }
    u64 t = sat_mul_u64(base_default, scale);
    return t < minv ? minv : t;
}

static u64 diag_scaled_ab_target_for_layers(const vector<LayerInfo>& layers, u64 minv) {
    u64 max_cap = 0;
    for (const LayerInfo& L : layers) {
        if (L.cap > max_cap) max_cap = L.cap;
    }
    if (max_cap == 0) return minv;

    constexpr long double SQRT2 = 1.414213562373095048801688724209698L;
    return clamp_target_u64((long double)max_cap * SQRT2, minv);
}

static WorkSplitTargets choose_work_split_targets(const vector<LayerInfo>& layers, u64 n_global) {
    WorkSplitTargets t{};

    const u64 linear_ab = fixed_scaled_target_for_n(2048ULL, 64ULL, n_global);
    const u64 diag_ab   = diag_scaled_ab_target_for_layers(layers, 64ULL);
    t.ab = std::min(linear_ab, diag_ab);
    t.x  = fixed_scaled_target_for_n(512ULL,  16ULL, n_global);
    return t;
}

static inline bool should_split_ab_diag(u64 cost, u64 pairs, u64 target_ab) {
    if (!cost || !pairs) return false;
    return cost > target_ab;
}

static inline u64 ab_diag_split_step(u64 cost, u64 pairs, u64 target_ab) {
    u64 mult = cost / pairs;
    if (mult == 0) mult = 1;
    u64 a_step = target_ab / mult;
    if (a_step < 1) a_step = 1;
    return a_step;
}

static void append_layer_work_with_targets(vector<WorkItem>& work, uint32_t layer_idx,
                                           u64 N, u64 cap, u64 TARGET_AB, u64 TARGET_X) {
    if (cap >= 2) {
        const u64 c_begin = 3;
        const u64 c_end = 2 * cap;
        u64 c = c_begin;
        while (c < c_end) {
            u64 pairs = diag_pair_count(c, cap);
            if (!pairs) { ++c; continue; }
            u64 cost = ab_diag_cost_est(N, cap, c);

            if (should_split_ab_diag(cost, pairs, TARGET_AB)) {
                u64 alo = c / 2 + 1;
                if (alo < 2) alo = 2;
                u64 ahi = c - 1;
                if (ahi > cap) ahi = cap;
                u64 mult = cost / pairs;
                if (mult == 0) mult = 1;
                u64 a_step = ab_diag_split_step(cost, pairs, TARGET_AB);
                for (u64 a = alo; a <= ahi; ) {
                    u64 ae = a + a_step;
                    if (ae <= a || ae > ahi + 1) ae = ahi + 1;
                    u64 pc = ae - a;
                    push_ab_diag_work(work, layer_idx, c, c + 1, a, ae, sat_mul_u64(pc, mult));
                    a = ae;
                }
                ++c;
                continue;
            }

            u64 c0 = c;
            u64 total = 0;
            while (c < c_end) {
                u64 ccost = ab_diag_cost_est(N, cap, c);
                if (ccost == 0) { ++c; continue; }
                if (total && sat_add_u64(total, ccost) > TARGET_AB) break;
                total = sat_add_u64(total, ccost);
                ++c;
            }
            if (c == c0) ++c;
            push_ab_diag_work(work, layer_idx, c0, c, 0, 0, total);
        }
    }

    if (cap >= 1) {
        u64 x = 1;
        while (x <= cap) {
            u64 c0 = x;
            u64 total = 0;
            while (x <= cap) {
                u64 ccost = x_cost_est(N, x);
                if (ccost == 0) { ++x; continue; }
                if (total && sat_add_u64(total, ccost) > TARGET_X) break;
                total = sat_add_u64(total, ccost);
                ++x;
            }
            if (x == c0) ++x;
            WorkItem w{};
            w.layer = layer_idx;
            w.kind = 1;
            w.lo = c0;
            w.hi = x;
            w.cost = total ? total : 1;
            work.push_back(w);
        }
    }
}

static bool split_work_item_once(const WorkItem& w, const vector<LayerInfo>& layers,
                                 WorkItem& a, WorkItem& b) {
    a = w;
    b = w;

    if (w.kind == 1) {
        if (w.hi <= w.lo + 1) return false;
        u64 len = w.hi - w.lo;
        u64 mid = w.lo + len / 2;
        if (mid <= w.lo || mid >= w.hi) return false;
        a.hi = mid;
        b.lo = mid;
        u64 ac = w.cost / 2;
        if (ac < 1) ac = 1;
        if (ac >= w.cost && w.cost > 1) ac = w.cost - 1;
        a.cost = ac;
        b.cost = w.cost > ac ? w.cost - ac : 1;
        return true;
    }

    if (w.kind == 2) {
        if (w.hi > w.lo + 1) {
            u64 len = w.hi - w.lo;
            u64 mid = w.lo + len / 2;
            if (mid <= w.lo || mid >= w.hi) return false;
            a.hi = mid;
            b.lo = mid;
            a.a_lo = a.a_hi = 0;
            b.a_lo = b.a_hi = 0;
            u64 ac = w.cost / 2;
            if (ac < 1) ac = 1;
            if (ac >= w.cost && w.cost > 1) ac = w.cost - 1;
            a.cost = ac;
            b.cost = w.cost > ac ? w.cost - ac : 1;
            return true;
        }

        if (w.lo >= w.hi || w.layer >= layers.size()) return false;
        const LayerInfo& L = layers[w.layer];
        u64 c = w.lo;
        u64 alo = c / 2 + 1;
        if (alo < 2) alo = 2;
        u64 ahi = c - 1;
        if (ahi > L.cap) ahi = L.cap;
        if (w.a_hi != 0) {
            if (alo < w.a_lo) alo = w.a_lo;
            if (ahi + 1 > w.a_hi) ahi = w.a_hi - 1;
        }
        if (ahi < alo) return false;
        u64 cnt = ahi - alo + 1;
        if (cnt <= 1) return false;
        u64 mid = alo + cnt / 2;
        if (mid <= alo || mid > ahi) return false;

        a.lo = c; a.hi = c + 1;
        b.lo = c; b.hi = c + 1;
        a.a_lo = alo;
        a.a_hi = mid;
        b.a_lo = mid;
        b.a_hi = ahi + 1;

        u64 ac = w.cost / 2;
        if (ac < 1) ac = 1;
        if (ac >= w.cost && w.cost > 1) ac = w.cost - 1;
        a.cost = ac;
        b.cost = w.cost > ac ? w.cost - ac : 1;
        return true;
    }

    return false;
}

static size_t desired_gpu_items_for_launch(size_t launch_blocks) {
    size_t desired = launch_blocks * 8u;
    if (desired < launch_blocks) desired = launch_blocks;
    return desired;
}

static size_t max_gpu_work_items() {
    return 1048576;
}

static void expand_work_for_occupancy(vector<WorkItem>& work,
                                      const vector<LayerInfo>& layers,
                                      size_t desired_items) {
    const size_t max_items = max_gpu_work_items();
    if (desired_items > max_items) desired_items = max_items;
    if (work.size() >= desired_items || work.empty()) return;

    for (;;) {
        if (work.size() >= desired_items || work.size() >= max_items) break;

        std::sort(work.begin(), work.end(), [](const WorkItem& x, const WorkItem& y) {
            return x.cost > y.cost;
        });

        vector<WorkItem> next;
        size_t reserve_n = work.size() * 2;
        if (reserve_n < work.size() || reserve_n > max_items) reserve_n = max_items;
        next.reserve(reserve_n);
        bool changed = false;

        for (const WorkItem& w : work) {
            if (next.size() + 2 <= max_items && next.size() < desired_items) {
                WorkItem a{}, b{};
                if (split_work_item_once(w, layers, a, b)) {
                    next.push_back(a);
                    next.push_back(b);
                    changed = true;
                    continue;
                }
            }
            next.push_back(w);
        }

        work.swap(next);
        if (!changed) break;
    }
}

struct F1Profile {
    double kernel_seconds = 0.0;
};

static void print_progress_line(const GpuStats& st, size_t total_items, unsigned long long total_cost,
                                bool final_line, double elapsed_sec) {
    unsigned long long done_cost = st.done_cost;
    if (final_line || st.done_items >= (unsigned long long)total_items) done_cost = total_cost;
    double pct = total_cost ? (100.0 * (double)done_cost / (double)total_cost) : 100.0;
    if (pct > 100.0) pct = 100.0;

    char elapsed_total[64], bar_buf[32];
    format_progress_bar(pct, bar_buf, sizeof(bar_buf));
    format_elapsed_total(elapsed_sec, pct, final_line || pct >= 99.999, elapsed_total, sizeof(elapsed_total));
    std::cerr << "\r# gpu    "
              << fixed << setprecision(1) << pct << "% " << bar_buf
              << " time=" << elapsed_total
              << "        ";
    if (final_line) std::cerr << "\n";
}
#define F1_LAUNCH_CASE(TPBV, MINBV) \
    case MINBV: \
        f1_work_kernel_persistent<TPBV, MINBV><<<grid, block, shared_bytes, stream>>>( \
            d_layers, d_work, d_out, d_stats, work_count, track_progress_cost); \
        break

static void launch_f1_kernel(int tpb, int minb,
                             dim3 grid, dim3 block, size_t shared_bytes,
                             cudaStream_t stream,
                             const LayerInfo* d_layers, const WorkItem* d_work,
                             Three192* d_out, GpuStats* d_stats,
                             int work_count, int track_progress_cost) {
    switch (tpb) {
        case 64:
            switch (minb) { F1_LAUNCH_CASE(64, 1); F1_LAUNCH_CASE(64, 2); F1_LAUNCH_CASE(64, 4); default: F1_LAUNCH_CASE(64, 8); }
            break;
        case 128:
            switch (minb) { F1_LAUNCH_CASE(128, 1); F1_LAUNCH_CASE(128, 2); F1_LAUNCH_CASE(128, 4); default: F1_LAUNCH_CASE(128, 8); }
            break;
        case 256:
            switch (minb) { F1_LAUNCH_CASE(256, 1); F1_LAUNCH_CASE(256, 2); default: F1_LAUNCH_CASE(256, 4); }
            break;
        default:
            switch (minb) { F1_LAUNCH_CASE(512, 1); default: F1_LAUNCH_CASE(512, 2); }
            break;
    }
}

#define F1_OCC_CASE(TPBV, MINBV) \
    case MINBV: \
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, \
            f1_work_kernel_persistent<TPBV, MINBV>, TPBV, shared_bytes)); \
        break

static int occupancy_blocks_per_sm_for_kernel(int tpb, int minb, size_t shared_bytes) {
    int occ = 0;
    switch (tpb) {
        case 64:
            switch (minb) { F1_OCC_CASE(64, 1); F1_OCC_CASE(64, 2); F1_OCC_CASE(64, 4); default: F1_OCC_CASE(64, 8); }
            break;
        case 128:
            switch (minb) { F1_OCC_CASE(128, 1); F1_OCC_CASE(128, 2); F1_OCC_CASE(128, 4); default: F1_OCC_CASE(128, 8); }
            break;
        case 256:
            switch (minb) { F1_OCC_CASE(256, 1); F1_OCC_CASE(256, 2); default: F1_OCC_CASE(256, 4); }
            break;
        default:
            switch (minb) { F1_OCC_CASE(512, 1); default: F1_OCC_CASE(512, 2); }
            break;
    }
    return occ;
}

#undef F1_LAUNCH_CASE
#undef F1_OCC_CASE

static void validate_k45_layer_bounds_or_die(const vector<LayerInfo>& layers) {
    u64 max_cap = 0;
    for (const LayerInfo& L : layers) {
        if (L.cap > max_cap) max_cap = L.cap;
    }
    if (max_cap > (u64)std::numeric_limits<uint32_t>::max() / 2u) {
        std::cerr << "internal bound failure: layer cap=" << (unsigned long long)max_cap
                  << " is too large for compact uint32 HFrame fields. This build is intended for k<=45.\n";
        std::exit(1);
    }
}

static I192 f1_cuda_with_layers(u64 n, const vector<LayerInfo>& layers, F1Profile* prof) {
    if (n <= 1) return I192();

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int sm_total = prop.multiProcessorCount;

    int tpb = F1_BUILTIN_CUDA_TPB;
    int launch_minb = F1_BUILTIN_CUDA_LAUNCH_MINB;
    int requested_blocks_per_sm = F1_BUILTIN_CUDA_BLOCKS_PER_SM;
    size_t shared_bytes = (size_t)((tpb + 31) / 32) * sizeof(Three192);
    int occ_blocks_per_sm = occupancy_blocks_per_sm_for_kernel(tpb, launch_minb, shared_bytes);

    int resident_blocks_per_sm = requested_blocks_per_sm;
    if (occ_blocks_per_sm > 0 && resident_blocks_per_sm > occ_blocks_per_sm) {
        resident_blocks_per_sm = occ_blocks_per_sm;
    }
    if (resident_blocks_per_sm < 1) resident_blocks_per_sm = 1;
    size_t launch_blocks = (size_t)sm_total * (size_t)resident_blocks_per_sm;
    if (launch_blocks < 1) launch_blocks = 1;

    WorkSplitTargets split_targets = choose_work_split_targets(layers, n);

    vector<WorkItem> work;

    size_t reserve_n = layers.size() * 16 + 1024;
    if (n >= (1ULL << 30) && reserve_n < 2000000ULL) reserve_n = 2000000ULL;
    if (n >= (1ULL << 34) && reserve_n < 6000000ULL) reserve_n = 6000000ULL;
    if (reserve_n > 20000000ULL) reserve_n = 20000000ULL;
    work.reserve(reserve_n);

    for (uint32_t idx = 0; idx < layers.size(); ++idx) {
        append_layer_work_with_targets(work, idx, layers[idx].N, layers[idx].cap, split_targets.ab, split_targets.x);
    }

    expand_work_for_occupancy(work, layers, desired_gpu_items_for_launch(launch_blocks));

    std::sort(work.begin(), work.end(), [](const WorkItem& a, const WorkItem& b) {
        return a.cost > b.cost;
    });

    unsigned long long total_work_cost = 0;
    for (const auto& w : work) {
        unsigned long long c = (unsigned long long)w.cost;
        if (ULLONG_MAX - total_work_cost < c) { total_work_cost = ULLONG_MAX; break; }
        total_work_cost += c;
    }
    if (work.empty()) return final_tail(n);
    if (work.size() > (size_t)std::numeric_limits<int>::max()) {
        std::cerr << "too many CUDA work items: " << work.size() << "\n";
        std::exit(1);
    }

    LayerInfo* d_layers = nullptr;
    WorkItem* d_work = nullptr;
    Three192* d_out = nullptr;
    GpuStats* d_stats = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_layers, layers.size() * sizeof(LayerInfo)));
    CUDA_CHECK(cudaMalloc((void**)&d_work, work.size() * sizeof(WorkItem)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, work.size() * sizeof(Three192)));
    CUDA_CHECK(cudaMalloc((void**)&d_stats, sizeof(GpuStats)));
    CUDA_CHECK(cudaMemcpy(d_layers, layers.data(), layers.size() * sizeof(LayerInfo), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_work, work.data(), work.size() * sizeof(WorkItem), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(GpuStats)));

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 65536));

    dim3 block(tpb);
    dim3 grid((unsigned)launch_blocks);

    bool show_progress = g_progress;
    cudaStream_t compute_stream = nullptr, copy_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    if (show_progress) CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0, compute_stream));
    launch_f1_kernel(tpb, launch_minb, grid, block, shared_bytes, compute_stream,
                     d_layers, d_work, d_out, d_stats, (int)work.size(), show_progress ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev1, compute_stream));

    int progress_ms = F1_BUILTIN_PROGRESS_MS;
    auto kernel_wall_t0 = std::chrono::steady_clock::now();
    GpuStats last_stats{};
    if (show_progress) {
        auto next_print = std::chrono::steady_clock::now();
        for (;;) {
            cudaError_t q = cudaEventQuery(ev1);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) CUDA_CHECK(q);

            auto now = std::chrono::steady_clock::now();
            if (now >= next_print) {
                CUDA_CHECK(cudaMemcpyAsync(&last_stats, d_stats, sizeof(GpuStats), cudaMemcpyDeviceToHost, copy_stream));
                CUDA_CHECK(cudaStreamSynchronize(copy_stream));
                double elapsed_sec = std::chrono::duration<double>(now - kernel_wall_t0).count();
                print_progress_line(last_stats, work.size(), total_work_cost, false, elapsed_sec);
                next_print = now + std::chrono::milliseconds(progress_ms);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }

    CUDA_CHECK(cudaEventSynchronize(ev1));
    if (show_progress) {
        CUDA_CHECK(cudaMemcpyAsync(&last_stats, d_stats, sizeof(GpuStats), cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - kernel_wall_t0).count();
        print_progress_line(last_stats, work.size(), total_work_cost, true, elapsed_sec);
    }

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    if (prof) {
        prof->kernel_seconds = (double)ms / 1000.0;
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    if (copy_stream) CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));

    vector<Three192> partial(work.size());
    CUDA_CHECK(cudaMemcpy(partial.data(), d_out, partial.size() * sizeof(Three192), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_layers));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_stats));

    vector<Three192> layer_coeff(layers.size());
    for (size_t i = 0; i < work.size(); ++i) {
        uint32_t li = work[i].layer;
        layer_coeff[li].a += partial[i].a;
        layer_coeff[li].b += partial[i].b;
        layer_coeff[li].c += partial[i].c;
    }

    I192 ans;
    for (size_t i = 0; i < layers.size(); ++i) {
        const Three192& lc = layer_coeff[i];
        const LayerInfo& L = layers[i];
        ans += lc.a.mul_i128_low(L.s0).mul_u64(n).mul_u64(n);
        ans -= lc.b.mul_i128_low(L.s1).mul_u64(n);
        ans += lc.c.mul(L.s2);
    }
    return ans + final_tail(n);
}

static bool parse_u32_arg(const char* s, uint32_t& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    errno = 0;
    unsigned long v = std::strtoul(s, &end, 10);
    if (errno || *end || v > std::numeric_limits<uint32_t>::max()) return false;
    out = (uint32_t)v;
    return true;
}

static u64 pow2_u64(uint32_t k) { return 1ULL << k; }

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        cerr << "usage: " << argv[0] << " k1 k2 [cuda_device_id] [--progress]\n";
        cerr << "computes f1(2^k) for every k in [k1, k2] using CUDA.\n";
        return 2;
    }

    uint32_t k1=0, k2=0;
    if (!parse_u32_arg(argv[1], k1) || !parse_u32_arg(argv[2], k2) || k1 > k2) {
        cerr << "bad k1/k2: expected integers with 0 <= k1 <= k2.\n";
        return 2;
    }

    if (k2 > 45) {
        cerr << "this k45-safe I192 production build is limited to k<=45 (n<=2^45).\n";
        return 2;
    }
    int device = 0;
    for (int ai = 3; ai < argc; ++ai) {
        if (!std::strcmp(argv[ai], "--progress")) g_progress = true;
        else device = std::atoi(argv[ai]);
    }
    CUDA_CHECK(cudaSetDevice(device));

    cout << "k\t2**k\tsieve_seconds\tkernel_seconds\ttotal_seconds\tanswer\n";
    cout.flush();

    for (uint32_t k = k1; k <= k2; ++k) {
        auto total_t0 = std::chrono::steady_clock::now();
        u64 n = pow2_u64(k);
        double sieve_sec = 0.0;
        vector<LayerInfo> layers = build_layers_with_segmented_mobius_cuda(n, sieve_sec);
        validate_k45_layer_bounds_or_die(layers);
        F1Profile prof;
        I192 ans = f1_cuda_with_layers(n, layers, &prof);
        auto total_t1 = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(total_t1 - total_t0).count();
        cout << k << '\t' << n << '\t'
             << fixed << setprecision(9)
             << sieve_sec << '\t' << prof.kernel_seconds << '\t' << total_sec << '\t'
             << ans.str() << '\n';
        cout.flush();
    }
    return 0;
}
