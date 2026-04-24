#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

using u64 = std::uint64_t;
using i64 = std::int64_t;
using u128 = unsigned __int128;

std::string to_string_u128(u128 x) {
    if (x == 0) return "0";
    std::string s;
    while (x > 0) {
        int digit = static_cast<int>(x % 10);
        s.push_back(static_cast<char>('0' + digit));
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

u128 sum_h1w1(u64 h, u64 w, u64 s, u64 r) {
    u64 m1 = (h - 1) / s;
    u64 m2 = (w - 1) / r;
    u64 m = 1 + std::min(m1, m2);

    u128 mm = static_cast<u128>(m);
    u128 sk  = mm * (mm - 1) / 2;
    u128 sk2 = (mm - 1) * mm * (2 * mm - 1) / 6;

    return mm * h * w
         - (static_cast<u128>(h) * r + static_cast<u128>(w) * s) * sk
         + (static_cast<u128>(r) * s) * sk2;
}

u128 F_fast(u64 n) {
    if (n < 2) {
        return 0;
    }

    u128 nn = static_cast<u128>(n);
    u128 base = (nn - 1) * (nn - 1) * nn * (2 * nn - 1) / 6;
    u128 count = 0;

    u64 a = 0, b = 1, c = 1, d = n - 1;

    while (c <= n - 1) {
        u64 s = a;
        u64 r = b;

        if (r != 1 && r + s < n) {
            i64 h = static_cast<i64>(n) - static_cast<i64>(r) - static_cast<i64>(s);
            i64 w = h;

            while (h > 0 && w > 0) {
                count += sum_h1w1(static_cast<u64>(h), static_cast<u64>(w), s, r);
                h -= static_cast<i64>(r);
                w -= static_cast<i64>(s);
            }
        }

        u64 k = static_cast<u64>((static_cast<u128>(n - 1) + b) / d);
        u64 na = c;
        u64 nb = d;
        u64 nc = k * c - a;
        u64 nd = k * d - b;

        a = na;
        b = nb;
        c = nc;
        d = nd;
    }

    return base + 2 * count;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <k>\n";
        std::cerr << "Program computes F_fast(2^1), F_fast(2^2), ..., F_fast(2^k)\n";
        return 1;
    }

    int kmax = 0;
    try {
        kmax = std::stoi(argv[1]);
    } catch (...) {
        std::cerr << "Invalid k\n";
        return 1;
    }

    if (kmax < 1 || kmax > 62) {
        std::cerr << "k must be in range [1, 62]\n";
        return 1;
    }

    for (int k = 1; k <= kmax; ++k) {
        u64 n = 1ULL << k;

        auto t1 = std::chrono::steady_clock::now();
        u128 result = F_fast(n);
        auto t2 = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed = t2 - t1;

        std::cout << "k = " << k
                  << ", n = " << n
                  << ", F_fast(n) = " << to_string_u128(result)
                  << ", time = " << std::fixed << std::setprecision(6)
                  << elapsed.count() << " s\n";
        std::cout.flush();
    }

    return 0;
}