// absbp_xz_tool_readable.cpp
//
// A small lossless compressor for a text file that contains exactly one integer
// per line, for example:
//
//     0
//     1
//     10
//     44
//     ...
//
// Format idea:
//   1. Store N, EOL style, and the first two numbers in a small header.
//   2. For every following number store the second difference:
//
//          d2[i] = value[i] - 2 * value[i - 1] + value[i - 2]
//
//   3. ZigZag-encode signed second differences into unsigned integers.
//   4. Write those unsigned integers by bit-planes.
//   5. Compress the resulting binary stream with external `xz`.
//
// Commands:
//
//     ./absbp_xz_tool_readable compress   values.txt values.absbp.xz
//     ./absbp_xz_tool_readable decompress values.absbp.xz restored.txt
//     ./absbp_xz_tool_readable verify     values.txt values.absbp.xz
//
// Build:
//
//     g++ -O2 -std=c++17 absbp_xz_tool_readable.cpp -o absbp_xz_tool_readable
//
// Notes:
//   * Requires the `xz` command line utility.
//   * Uses signed __int128, so every input number must fit into signed 128-bit.
//   * The compressed file is compatible with the previous ABP2 .absbp.xz format.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using i128 = __int128_t;
using u128 = __uint128_t;

namespace {

// -----------------------------------------------------------------------------
// Constants and small helpers
// -----------------------------------------------------------------------------

constexpr char kMagic[4] = {'A', 'B', 'P', '2'};
constexpr std::size_t kIoBufferSize = 1 << 20;

struct Header {
    std::uint64_t row_count = 0;
    std::uint16_t bit_width = 0;
    std::uint8_t has_final_eol = 1;
    std::string eol = "\n";
    i128 first_value = 0;
    i128 second_value = 0;
};

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string ShellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::string WithCommas(std::uint64_t value) {
    std::string s = std::to_string(value);
    std::string out;
    int group = 0;

    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (group == 3) {
            out.push_back(',');
            group = 0;
        }
        out.push_back(*it);
        ++group;
    }

    std::reverse(out.begin(), out.end());
    return out;
}

void PrintProgress(const char* phase, std::uint64_t rows, bool force = false) {
    static auto last_print = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();

    if (force || now - last_print > std::chrono::seconds(2)) {
        std::cerr << "\r" << phase << ": " << WithCommas(rows) << " rows" << std::flush;
        last_print = now;

        if (force) {
            std::cerr << "\n";
        }
    }
}

void SetLargeInputBuffer(std::ifstream& stream, std::vector<char>& buffer) {
    buffer.resize(kIoBufferSize);
    stream.rdbuf()->pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
}

// -----------------------------------------------------------------------------
// Decimal <-> signed 128-bit
// -----------------------------------------------------------------------------

std::string Int128ToString(i128 value) {
    if (value == 0) {
        return "0";
    }

    const bool is_negative = value < 0;
    u128 magnitude = is_negative ? (u128(0) - u128(value)) : u128(value);

    std::string digits;
    while (magnitude != 0) {
        const unsigned digit = static_cast<unsigned>(magnitude % 10);
        digits.push_back(static_cast<char>('0' + digit));
        magnitude /= 10;
    }

    if (is_negative) {
        digits.push_back('-');
    }

    std::reverse(digits.begin(), digits.end());
    return digits;
}

i128 ParseInt128(std::string line, std::uint64_t line_number) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    if (line.empty()) {
        Fail("empty line at " + std::to_string(line_number));
    }

    bool is_negative = false;
    std::size_t pos = 0;

    if (line[0] == '-') {
        is_negative = true;
        pos = 1;
    }

    if (pos == line.size()) {
        Fail("bad integer at line " + std::to_string(line_number));
    }

    const u128 positive_limit = (u128(1) << 127) - 1;
    const u128 negative_limit = (u128(1) << 127);
    const u128 limit = is_negative ? negative_limit : positive_limit;

    u128 value = 0;
    for (; pos < line.size(); ++pos) {
        const char c = line[pos];

        if (c == ' ' || c == '\t') {
            Fail("line " + std::to_string(line_number) +
                 " has spaces/tabs; expected exactly one integer per line");
        }

        if (c < '0' || c > '9') {
            Fail("bad character at line " + std::to_string(line_number));
        }

        const unsigned digit = static_cast<unsigned>(c - '0');
        if (value > (limit - digit) / 10) {
            Fail("integer exceeds signed 128-bit range at line " + std::to_string(line_number));
        }

        value = value * 10 + digit;
    }

    return is_negative ? -i128(value) : i128(value);
}

// -----------------------------------------------------------------------------
// ZigZag coding for signed second differences
// -----------------------------------------------------------------------------

u128 ZigZagEncode(i128 value) {
    if (value >= 0) {
        return u128(value) << 1;
    }

    const u128 magnitude = u128(0) - u128(value);
    return (magnitude << 1) - 1;
}

i128 ZigZagDecode(u128 code) {
    if (code & 1) {
        return -i128((code >> 1) + 1);
    }
    return i128(code >> 1);
}

std::uint16_t BitLength(u128 value) {
    if (value == 0) {
        return 0;
    }

    const std::uint64_t high = static_cast<std::uint64_t>(value >> 64);
    if (high != 0) {
        return static_cast<std::uint16_t>(64 + (64 - __builtin_clzll(high)));
    }

    const std::uint64_t low = static_cast<std::uint64_t>(value);
    return static_cast<std::uint16_t>(64 - __builtin_clzll(low));
}

// -----------------------------------------------------------------------------
// Tiny binary encoding helpers for the ABP2 header
// -----------------------------------------------------------------------------

void PutU16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void PutU64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

std::uint16_t GetU16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (std::uint16_t(p[1]) << 8));
}

std::uint64_t GetU64(const std::uint8_t* p) {
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        value |= std::uint64_t(*p++) << shift;
    }
    return value;
}

void PutVarUInt(std::vector<std::uint8_t>& out, std::uint64_t value) {
    while (true) {
        std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7f);
        value >>= 7;

        if (value != 0) {
            out.push_back(byte | 0x80);
        } else {
            out.push_back(byte);
            return;
        }
    }
}

void PutString(std::vector<std::uint8_t>& out, const std::string& s) {
    PutVarUInt(out, static_cast<std::uint64_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

std::uint64_t ReadVarUInt(FILE* file) {
    std::uint64_t value = 0;
    unsigned shift = 0;

    while (true) {
        const int ch = std::fgetc(file);
        if (ch == EOF) {
            Fail("unexpected EOF while reading varuint");
        }

        const auto byte = static_cast<std::uint8_t>(ch);
        value |= std::uint64_t(byte & 0x7f) << shift;

        if ((byte & 0x80) == 0) {
            return value;
        }

        shift += 7;
        if (shift >= 64) {
            Fail("bad varuint");
        }
    }
}

std::string ReadString(FILE* file) {
    const std::uint64_t size = ReadVarUInt(file);

    std::string s(static_cast<std::size_t>(size), '\0');
    if (size != 0 && std::fread(s.data(), 1, static_cast<std::size_t>(size), file) != size) {
        Fail("unexpected EOF while reading string");
    }

    return s;
}

std::vector<std::uint8_t> EncodeHeader(const Header& header) {
    std::vector<std::uint8_t> out;

    out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
    PutU64(out, header.row_count);
    PutU16(out, header.bit_width);
    out.push_back(header.has_final_eol);
    PutString(out, header.eol);

    if (header.row_count >= 1) {
        PutString(out, Int128ToString(header.first_value));
    }
    if (header.row_count >= 2) {
        PutString(out, Int128ToString(header.second_value));
    }

    return out;
}

Header DecodeHeader(FILE* file) {
    std::uint8_t fixed[15];

    if (std::fread(fixed, 1, sizeof(fixed), file) != sizeof(fixed)) {
        Fail("unexpected EOF while reading header");
    }

    if (std::memcmp(fixed, kMagic, sizeof(kMagic)) != 0) {
        Fail("bad magic; not an ABP2 file");
    }

    Header header;
    header.row_count = GetU64(fixed + 4);
    header.bit_width = GetU16(fixed + 12);
    header.has_final_eol = fixed[14];
    header.eol = ReadString(file);

    if (header.eol != "\n" && header.eol != "\r\n") {
        Fail("unsupported EOL style in compressed file");
    }

    if (header.bit_width > 128) {
        Fail("bit width > 128 is unsupported by this C++ build");
    }

    if (header.row_count >= 1) {
        header.first_value = ParseInt128(ReadString(file), 0);
    }
    if (header.row_count >= 2) {
        header.second_value = ParseInt128(ReadString(file), 0);
    }

    return header;
}

// -----------------------------------------------------------------------------
// Detect input text details
// -----------------------------------------------------------------------------

void DetectEolAndFinalNewline(const std::string& path, Header& header) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Fail("cannot open input: " + path);
    }

    char previous = 0;
    char current = 0;
    bool saw_newline = false;

    while (input.get(current)) {
        if (current == '\n') {
            header.eol = (previous == '\r') ? "\r\n" : "\n";
            saw_newline = true;
            break;
        }
        previous = current;
    }

    input.clear();
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();

    if (size <= 0) {
        header.has_final_eol = 1;
        return;
    }

    input.seekg(-1, std::ios::end);
    char last = 0;
    input.get(last);
    header.has_final_eol = (last == '\n') ? 1 : 0;

    if (!saw_newline) {
        header.eol = "\n";
    }
}

// -----------------------------------------------------------------------------
// First pass: count rows and find the maximum ZigZag-encoded second difference
// -----------------------------------------------------------------------------

Header ScanInputFile(const std::string& path) {
    Header header;
    DetectEolAndFinalNewline(path, header);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Fail("cannot open input: " + path);
    }

    std::vector<char> buffer;
    SetLargeInputBuffer(input, buffer);

    std::string line;
    i128 value_before_previous = 0;
    i128 previous_value = 0;
    u128 max_code = 0;
    std::uint64_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const i128 value = ParseInt128(line, line_number);

        ++header.row_count;
        if (header.row_count == 1) {
            header.first_value = value;
        } else if (header.row_count == 2) {
            header.second_value = value;
        } else {
            const i128 second_difference = (value - previous_value) -
                                           (previous_value - value_before_previous);
            const u128 code = ZigZagEncode(second_difference);
            if (code > max_code) {
                max_code = code;
            }
        }

        value_before_previous = previous_value;
        previous_value = value;

        if ((header.row_count & 0x3ffff) == 0) {
            PrintProgress("scan", header.row_count);
        }
    }

    if (!input.eof()) {
        Fail("read error while scanning input");
    }

    header.bit_width = BitLength(max_code);
    PrintProgress("scan", header.row_count, true);
    return header;
}

// -----------------------------------------------------------------------------
// Second pass: write ZigZag-coded second differences by bit-planes
// -----------------------------------------------------------------------------

std::size_t BytesPerBitPlane(std::uint64_t value_count) {
    return static_cast<std::size_t>((value_count + 7) / 8);
}

std::vector<std::uint8_t> PackBitPlanes(const std::string& path, const Header& header) {
    const std::uint64_t code_count = (header.row_count >= 2) ? header.row_count - 2 : 0;
    const std::size_t plane_size = BytesPerBitPlane(code_count);

    std::vector<std::uint8_t> planes(std::size_t(header.bit_width) * plane_size, 0);
    if (code_count == 0 || header.bit_width == 0) {
        return planes;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Fail("cannot open input: " + path);
    }

    std::vector<char> buffer;
    SetLargeInputBuffer(input, buffer);

    std::string line;
    i128 value_before_previous = 0;
    i128 previous_value = 0;
    std::uint64_t row = 0;
    std::uint64_t code_index = 0;

    while (std::getline(input, line)) {
        ++row;
        const i128 value = ParseInt128(line, row);

        if (row >= 3) {
            const i128 second_difference = (value - previous_value) -
                                           (previous_value - value_before_previous);
            u128 code = ZigZagEncode(second_difference);

            const std::size_t byte_index = static_cast<std::size_t>(code_index >> 3);
            const std::uint8_t mask = static_cast<std::uint8_t>(1u << (code_index & 7));

            std::uint16_t bit = 0;
            while (code != 0) {
                if (code & 1) {
                    planes[std::size_t(bit) * plane_size + byte_index] |= mask;
                }
                code >>= 1;
                ++bit;
            }

            ++code_index;
        }

        value_before_previous = previous_value;
        previous_value = value;

        if ((row & 0x3ffff) == 0) {
            PrintProgress("pack", row);
        }
    }

    if (code_index != code_count) {
        Fail("row count mismatch during packing");
    }

    PrintProgress("pack", row, true);
    return planes;
}

u128 ReadCodeFromBitPlanes(
    const std::vector<std::uint8_t>& planes,
    std::uint64_t code_index,
    std::uint16_t bit_width,
    std::size_t plane_size
) {
    const std::size_t byte_index = static_cast<std::size_t>(code_index >> 3);
    const std::uint8_t mask = static_cast<std::uint8_t>(1u << (code_index & 7));

    u128 code = 0;
    for (std::uint16_t bit = 0; bit < bit_width; ++bit) {
        if (planes[std::size_t(bit) * plane_size + byte_index] & mask) {
            code |= u128(1) << bit;
        }
    }

    return code;
}

// -----------------------------------------------------------------------------
// External xz process helpers
// -----------------------------------------------------------------------------

FILE* OpenXzForWriting(const std::string& output_path, int preset, bool extreme) {
    const std::string level = "-" + std::to_string(preset) + (extreme ? "e" : "");
    const std::string command = "xz " + level + " -c > " + ShellQuote(output_path);

    FILE* file = popen(command.c_str(), "w");
    if (!file) {
        Fail("cannot start: " + command);
    }
    return file;
}

FILE* OpenXzForReading(const std::string& input_path) {
    const std::string command = "xz -dc -- " + ShellQuote(input_path);

    FILE* file = popen(command.c_str(), "r");
    if (!file) {
        Fail("cannot start: " + command);
    }
    return file;
}

void WriteExact(FILE* file, const void* data, std::size_t size) {
    const auto* p = static_cast<const std::uint8_t*>(data);

    while (size != 0) {
        const std::size_t written = std::fwrite(p, 1, size, file);
        if (written == 0) {
            Fail("write to xz failed");
        }
        p += written;
        size -= written;
    }
}

void ReadExact(FILE* file, void* data, std::size_t size) {
    auto* p = static_cast<std::uint8_t*>(data);

    while (size != 0) {
        const std::size_t read = std::fread(p, 1, size, file);
        if (read == 0) {
            Fail("unexpected EOF");
        }
        p += read;
        size -= read;
    }
}

std::vector<std::uint8_t> ReadBitPlanes(FILE* file, const Header& header) {
    const std::uint64_t code_count = (header.row_count >= 2) ? header.row_count - 2 : 0;
    const std::size_t total_size = std::size_t(header.bit_width) * BytesPerBitPlane(code_count);

    std::vector<std::uint8_t> planes(total_size);
    if (total_size != 0) {
        ReadExact(file, planes.data(), planes.size());
    }
    return planes;
}

// -----------------------------------------------------------------------------
// Output sinks used by decompression and verification
// -----------------------------------------------------------------------------

struct FileOutputSink {
    explicit FileOutputSink(const std::string& path) : output(path, std::ios::binary) {
        if (!output) {
            Fail("cannot open output: " + path);
        }
    }

    void Write(const char* data, std::size_t size) {
        output.write(data, static_cast<std::streamsize>(size));
        if (!output) {
            Fail("write failed");
        }
    }

    std::ofstream output;
};

struct CompareWithFileSink {
    explicit CompareWithFileSink(const std::string& path)
        : input(path, std::ios::binary), buffer(kIoBufferSize) {
        if (!input) {
            Fail("cannot open original: " + path);
        }
    }

    void Write(const char* decoded_data, std::size_t size) {
        std::size_t position = 0;

        while (position < size) {
            const std::size_t chunk_size = std::min(buffer.size(), size - position);
            input.read(buffer.data(), static_cast<std::streamsize>(chunk_size));

            if (input.gcount() != static_cast<std::streamsize>(chunk_size)) {
                Fail("original ended early at byte " + std::to_string(offset + input.gcount()));
            }

            if (std::memcmp(buffer.data(), decoded_data + position, chunk_size) != 0) {
                std::size_t mismatch = 0;
                while (mismatch < chunk_size && buffer[mismatch] == decoded_data[position + mismatch]) {
                    ++mismatch;
                }
                Fail("byte mismatch at offset " + std::to_string(offset + mismatch));
            }

            position += chunk_size;
            offset += chunk_size;
        }
    }

    void Finish() {
        char extra = 0;
        if (input.get(extra)) {
            Fail("decoded output ended early at byte " + std::to_string(offset));
        }
    }

    std::ifstream input;
    std::uint64_t offset = 0;
    std::vector<char> buffer;
};

// -----------------------------------------------------------------------------
// Reconstruction from header + bit-planes
// -----------------------------------------------------------------------------

template <class Sink>
void ReconstructText(const Header& header, const std::vector<std::uint8_t>& planes, Sink& sink) {
    const std::uint64_t code_count = (header.row_count >= 2) ? header.row_count - 2 : 0;
    const std::size_t plane_size = BytesPerBitPlane(code_count);

    auto emit_value = [&](std::uint64_t row_index, i128 value) {
        const std::string text = Int128ToString(value);
        sink.Write(text.data(), text.size());

        const bool is_last_row = (row_index == header.row_count - 1);
        if (!(is_last_row && !header.has_final_eol)) {
            sink.Write(header.eol.data(), header.eol.size());
        }
    };

    if (header.row_count == 0) {
        return;
    }

    emit_value(0, header.first_value);

    if (header.row_count >= 2) {
        emit_value(1, header.second_value);
    }

    i128 value_before_previous = header.first_value;
    i128 previous_value = header.second_value;

    for (std::uint64_t code_index = 0; code_index < code_count; ++code_index) {
        const u128 code = ReadCodeFromBitPlanes(
            planes,
            code_index,
            header.bit_width,
            plane_size
        );

        const i128 second_difference = ZigZagDecode(code);
        const i128 value = previous_value + (previous_value - value_before_previous) + second_difference;

        emit_value(code_index + 2, value);

        value_before_previous = previous_value;
        previous_value = value;

        if ((code_index & 0x3ffff) == 0) {
            PrintProgress("decode", code_index + 2);
        }
    }

    PrintProgress("decode", header.row_count, true);
}

// -----------------------------------------------------------------------------
// Commands
// -----------------------------------------------------------------------------

int CompressCommand(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " compress input.txt output.absbp.xz [preset 0..9] [--no-extreme]\n";
        return 2;
    }

    int preset = 9;
    bool extreme = true;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-extreme") {
            extreme = false;
        } else {
            preset = std::stoi(arg);
        }
    }

    if (preset < 0 || preset > 9) {
        Fail("xz preset must be in range 0..9");
    }

    const std::string input_path = argv[2];
    const std::string output_path = argv[3];

    const Header header = ScanInputFile(input_path);
    std::cerr << "rows: " << WithCommas(header.row_count) << "\n";
    std::cerr << "bit width: " << header.bit_width << "\n";

    const std::vector<std::uint8_t> bit_planes = PackBitPlanes(input_path, header);
    std::cerr << "raw bitplanes: " << WithCommas(bit_planes.size()) << " bytes\n";

    FILE* xz = OpenXzForWriting(output_path, preset, extreme);

    const std::vector<std::uint8_t> encoded_header = EncodeHeader(header);
    WriteExact(xz, encoded_header.data(), encoded_header.size());

    if (!bit_planes.empty()) {
        WriteExact(xz, bit_planes.data(), bit_planes.size());
    }

    const int close_code = pclose(xz);
    if (close_code != 0) {
        Fail("xz failed");
    }

    std::cerr << "written: " << output_path << "\n";
    return 0;
}

int DecompressCommand(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " decompress input.absbp.xz output.txt\n";
        return 2;
    }

    const std::string input_path = argv[2];
    const std::string output_path = argv[3];

    FILE* xz = OpenXzForReading(input_path);
    const Header header = DecodeHeader(xz);
    const std::vector<std::uint8_t> bit_planes = ReadBitPlanes(xz, header);

    const int close_code = pclose(xz);
    if (close_code != 0) {
        Fail("xz failed");
    }

    FileOutputSink output(output_path);
    ReconstructText(header, bit_planes, output);
    return 0;
}

int VerifyCommand(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " verify original.txt compressed.absbp.xz\n";
        return 2;
    }

    const std::string original_path = argv[2];
    const std::string compressed_path = argv[3];

    FILE* xz = OpenXzForReading(compressed_path);
    const Header header = DecodeHeader(xz);
    const std::vector<std::uint8_t> bit_planes = ReadBitPlanes(xz, header);

    const int close_code = pclose(xz);
    if (close_code != 0) {
        Fail("xz failed");
    }

    CompareWithFileSink compare(original_path);
    ReconstructText(header, bit_planes, compare);
    compare.Finish();

    std::cout << "OK: restored file is byte-for-byte identical\n";
    return 0;
}

void PrintUsage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " compress   input.txt output.absbp.xz [preset 0..9] [--no-extreme]\n"
        << "  " << program_name << " decompress input.absbp.xz output.txt\n"
        << "  " << program_name << " verify     original.txt compressed.absbp.xz\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            PrintUsage(argv[0]);
            return 2;
        }

        const std::string command = argv[1];

        if (command == "compress") {
            return CompressCommand(argc, argv);
        }

        if (command == "decompress") {
            return DecompressCommand(argc, argv);
        }

        if (command == "verify") {
            return VerifyCommand(argc, argv);
        }

        std::cerr << "unknown command: " << command << "\n";
        PrintUsage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
