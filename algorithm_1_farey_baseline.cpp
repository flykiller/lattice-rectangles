#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>

using i64 = int64_t;
using i128 = unsigned __int128; 

std::string to_string_i128(i128 n) {
    if (n == 0) return "0";
    std::string s;
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

i128 count_all_rectangles(i64 n) {
    if (n <= 1) return 0;

    i128 un = n;
    i128 base = (un - 1) * ((un - 1) * un * (2 * un - 1) / 6);
    i128 count = 0;

    i64 limit = n - 1;
    i64 a = 0, b = 1, c = 1, d = limit;

    auto process = [&](i64 s, i64 r) __attribute__((always_inline)) {
        if (r == 1) return;
        
        i64 h = n - r - s;
        i64 w = h; 
        
        i128 rs = (i128)r * s; 
        
        while (h > 0 && w > 0) {
            i64 m1 = (h - 1) / s;
            i64 m2 = (w - 1) / r;
            
            i64 m = 1 + (m1 < m2 ? m1 : m2);
            
            i128 sk = (i128)m * (m - 1) / 2;
            i128 sk2 = (i128)(m - 1) * m * (2 * m - 1) / 6;
            
            count += (i128)m * h * w - ((i128)h * r + (i128)w * s) * sk + rs * sk2;
            
            h -= r;
            w -= s;
        }
    };

    process(a, b);
    while (c <= limit) {
        i64 k_val = (limit + b) / d; 
        i64 next_c = k_val * c - a;
        i64 next_d = k_val * d - b;
        
        a = c;
        b = d;
        c = next_c;
        d = next_d;
        
        process(a, b);
    }

    return base + 2 * count;
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

