# Counting Lattice Rectangles

Code accompanying the paper **“Counting All Lattice Rectangles in the Square Grid in Near-Linear Time”** by Dmitry Babichev and Sergey Babichev.

The programs compute the number of lattice rectangles in the square grid and were used to produce the timing comparisons in the paper. The Python files are simple reference/prototype versions; the C++ and CUDA files are the benchmark implementations.

## Files

| File | Description |
|---|---|
| `algorithm_1_farey_baseline.cpp` / `.py` | Baseline implementation using the Farey-sequence summation. |
| `algorithm_2_mobius_split.cpp` / `.py` | Improved implementation using a small/large split and Möbius inversion. |
| `algorithm_3_floor_moments.cpp` / `.py` | Faster implementation using floor-sum moment recurrences. |
| `algorithm_4_near_linear.cpp` / `.py` | Main near-linear implementation used for the fastest sequential timings. |
| `algorithm_5_divisor_layer.cpp` / `.py` | Implementation of the divisor-layer algorithm with running time `O(n log^2 n)`. |
| `all_values_bench.cpp` | Benchmark implementation of the all-values `O(N^{3/2})` algorithm. |
| `parallel_large_values.cpp` | Parallel C++ implementation used to compute large individual values. |
| `cuda_large_values.cu` | CUDA implementation used to compute the largest reported values, up to `n = 2^40`. |
| `absbp_xz_tool.cpp` | Encoder/decoder for the compressed precomputed table in the `v1.0-data` release. |

## Building

Compile the sequential C++ programs with:

```bash
g++ -O3 -std=c++17 algorithm_1_farey_baseline.cpp -o algorithm_1_farey_baseline
g++ -O3 -std=c++17 algorithm_2_mobius_split.cpp -o algorithm_2_mobius_split
g++ -O3 -std=c++17 algorithm_3_floor_moments.cpp -o algorithm_3_floor_moments
g++ -O3 -std=c++17 algorithm_4_near_linear.cpp -o algorithm_4_near_linear
g++ -O3 -std=c++17 algorithm_5_divisor_layer.cpp -o algorithm_5_divisor_layer
g++ -O3 -std=c++17 all_values_bench.cpp -o all_values_bench
```

Compile the parallel C++ program with:

```bash
g++ -O3 -std=c++17 -pthread parallel_large_values.cpp -o parallel_large_values
```

Compile the CUDA program with:

```bash
nvcc -O3 -std=c++17 -arch=sm_89 cuda_large_values.cu -o cuda_large_values
```

Change `sm_89` if your GPU requires a different CUDA architecture.

Compile the compressed-table tool with:

```bash
g++ -O2 -std=c++17 absbp_xz_tool.cpp -o absbp_xz_tool
```

The compressed-table tool requires the `xz` command-line utility.

## Running

The sequential programs benchmark all inputs `n = 2^1, 2^2, ..., 2^K`:

```bash
./algorithm_1_farey_baseline K
./algorithm_2_mobius_split K
./algorithm_3_floor_moments K
./algorithm_4_near_linear K
./algorithm_5_divisor_layer K
```

The all-values benchmark computes `F(N)` for `N = 2^1, 2^2, ..., 2^K`:

```bash
./all_values_bench K
```

The parallel C++ program computes a range of powers of two:

```bash
./parallel_large_values k1 k2 threads chunk_size
```

Example:

```bash
./parallel_large_values 8 12 16 64
```

The CUDA program computes a range of powers of two on a selected GPU:

```bash
./cuda_large_values k1 k2 [cuda_device_id] [--progress]
```

Example:

```bash
./cuda_large_values 35 40 0 --progress
```

## Precomputed values

Compressed precomputed values of `F(N)` for all `N <= 10^8` are available in the [`v1.0-data`](https://github.com/flykiller/lattice-rectangles/releases/tag/v1.0-data) release.

The unpacked table is large, so it is not stored directly in this git repository. The release contains an `.absbp.xz` archive; after decompression it becomes a plain text file with one integer value per line.

To decode the archive, build `absbp_xz_tool.cpp` and run:

```bash
./absbp_xz_tool decompress values_up_to_1e8.absbp.xz values_up_to_1e8.txt
```

The tool also supports compression and verification:

```bash
./absbp_xz_tool compress   values_up_to_1e8.txt values_up_to_1e8.absbp.xz
./absbp_xz_tool verify     values_up_to_1e8.txt values_up_to_1e8.absbp.xz
```

The compressed format stores the first two values explicitly, then stores signed second differences using ZigZag coding, packs them by bit-planes, and finally compresses the binary stream with `xz`.

## Values at powers of two

The following table gives the exact values used for the large-value checks in the paper. The entries up to `2^40` were produced with the CUDA implementation.

| k | F(2^k) | k | F(2^k) |
|---:|---:|---:|---:|
| 1 | `1` | 21 | `48931868439876126051425552` |
| 2 | `44` | 22 | `821437615651793675198669752` |
| 3 | `1192` | 23 | `13759445380252558103053449112` |
| 4 | `27128` | 24 | `230014222561387209679445816240` |
| 5 | `564120` | 25 | `3838037104619867210112196814232` |
| 6 | `11114080` | 26 | `63933546372113490066412405897360` |
| 7 | `211224480` | 27 | `1063335985124949941305863686097296` |
| 8 | `3914221216` | 28 | `17659763652737469299382592232330696` |
| 9 | `71182606216` | 29 | `292898424695610564494215857912343064` |
| 10 | `1275797150128` | 30 | `4851850095158746095561485451592336296` |
| 11 | `22602804487208` | 31 | `80277206323003614389748671287223855080` |
| 12 | `396685572297544` | 32 | `1326796977975476403092689286862986516504` |
| 13 | `6907621416632376` | 33 | `21906538476526319541299023010218991588136` |
| 14 | `119492377263166968` | 34 | `361349204887120272089523042249821840571528` |
| 15 | `2055404973525169560` | 35 | `5955100706397110811260922659812491131662432` |
| 16 | `35182910663019639384` | 36 | `98057826153604756744005601368029402514221504` |
| 17 | `599669468453524178752` | 37 | `1613344656077691850026984888873116366804460232` |
| 18 | `10182597857710132553464` | 38 | `26524225499163321460061315970545176007812869616` |
| 19 | `172327747508964813792096` | 39 | `435758984017337173124103405065600778830350047408` |
| 20 | `2907742868855598433202344` | 40 | `7154085760768979246024995359851578213153827420872` |

## Notes

The C++ programs use 128-bit integer arithmetic, except for `parallel_large_values.cpp`, which includes a custom 256-bit integer type for larger outputs. The CUDA implementation uses exact multi-limb integer arithmetic on the GPU and was used for the largest computations reported in the paper.
