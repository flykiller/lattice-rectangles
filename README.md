# Counting Lattice Rectangles

Code accompanying the paper **“Counting All Lattice Rectangles in the Square Grid in Near-Linear Time”** by Dmitry Babichev and Sergey Babichev.

The programs compute the number of lattice rectangles in the square grid and were used to produce the timing comparisons in the paper. The Python files are simple reference/prototype versions; the C++ files are the benchmark implementations.

## Files

| File | Description |
|---|---|
| `algorithm_1_farey_baseline.cpp` / `.py` | Baseline implementation using the Farey-sequence summation. |
| `algorithm_2_mobius_split.cpp` / `.py` | Improved implementation using a small/large split and Möbius inversion. |
| `algorithm_3_floor_moments.cpp` / `.py` | Faster implementation using floor-sum moment recurrences. |
| `algorithm_4_near_linear.cpp` / `.py` | Main near-linear implementation used for the fastest sequential timings. |
| `parallel_large_values.cpp` | Parallel C++ implementation used to compute the largest values reported in the paper. |

## Building

Compile the sequential C++ programs with:

```bash
g++ -O3 -std=c++17 algorithm_1_farey_baseline.cpp -o algorithm_1_farey_baseline
g++ -O3 -std=c++17 algorithm_2_mobius_split.cpp -o algorithm_2_mobius_split
g++ -O3 -std=c++17 algorithm_3_floor_moments.cpp -o algorithm_3_floor_moments
g++ -O3 -std=c++17 algorithm_4_near_linear.cpp -o algorithm_4_near_linear
```

Compile the parallel program with:

```bash
g++ -O3 -std=c++17 -pthread parallel_large_values.cpp -o parallel_large_values
```

## Running

The sequential programs benchmark all inputs `n = 2^1, 2^2, ..., 2^K`:

```bash
./algorithm_1_farey_baseline K
./algorithm_2_mobius_split K
./algorithm_3_floor_moments K
./algorithm_4_near_linear K
```

The parallel program computes a range of powers of two:

```bash
./parallel_large_values k1 k2 threads chunk_size
```

Example:

```bash
./parallel_large_values 8 12 16 64
```

## Notes

The C++ programs use 128-bit integer arithmetic, except for `parallel_large_values.cpp`, which includes a custom 256-bit integer type for larger outputs.
