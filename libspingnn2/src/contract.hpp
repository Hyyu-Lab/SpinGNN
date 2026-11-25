#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

namespace sp2 {

using Json = nlohmann::json;

std::string sha256_bytes(const std::vector<unsigned char>& bytes);

// Host-only reader for the immutable arrays referenced by M1's JSON model.
class Contract {
public:
    explicit Contract(const std::filesystem::path& manifest);
    Json model;
    const std::string& digest() const noexcept { return digest_; }

    const std::vector<std::size_t>& shape(const Json& ref) const {
        return descriptor(ref).shape;
    }

    template<class T>
    std::vector<T> array(const Json& ref) const {
        static_assert(std::is_arithmetic_v<T>, "Contract readers require a numeric type");
        const auto& desc = descriptor(ref);
        std::vector<T> result(desc.count);

        // The file is always little-endian, independent of the host. Decode
        // with memcpy rather than giving vector<unsigned char> a wider alias.
        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto* data = blob_.data() + desc.offset + i * desc.width;
            switch (desc.dtype) {
                case Dtype::f64: result[i] = convert<T>(little<float64_bits, double>(data)); break;
                case Dtype::f32: result[i] = convert<T>(little<float32_bits, float>(data)); break;
                case Dtype::i64: result[i] = convert<T>(little<std::uint64_t, std::int64_t>(data)); break;
                case Dtype::i32: result[i] = convert<T>(little<std::uint32_t, std::int32_t>(data)); break;
                case Dtype::i16: result[i] = convert<T>(little<std::uint16_t, std::int16_t>(data)); break;
                case Dtype::i8: result[i] = convert<T>(little<std::uint8_t, std::int8_t>(data)); break;
                case Dtype::u8: result[i] = convert<T>(*data); break;
                case Dtype::boolean: result[i] = convert<T>(*data != 0); break;
            }
        }
        return result;
    }

private:
    using float64_bits = std::uint64_t;
    using float32_bits = std::uint32_t;

    enum class Dtype { f64, f32, i64, i32, i16, i8, u8, boolean };

    struct Descriptor {
        std::size_t offset;
        std::vector<std::size_t> shape;
        std::size_t count;
        std::size_t width;
        Dtype dtype;
    };

    const Descriptor& descriptor(const Json& ref) const;

    template<class Bits, class Value>
    static Value little(const unsigned char* bytes) {
        static_assert(sizeof(Bits) == sizeof(Value), "M1 wire type width mismatch");
        Bits pattern = 0;
        for (std::size_t byte = 0; byte < sizeof(Bits); ++byte) {
            pattern |= static_cast<Bits>(bytes[byte]) << (8 * byte);
        }
        Value value;
        std::memcpy(&value, &pattern, sizeof(Value));
        return value;
    }

    template<class Out, class In>
    static Out convert(In value) {
        if constexpr (std::is_integral_v<Out> && !std::is_same_v<Out, bool>) {
            const auto wide = static_cast<long double>(value);
            if (!(wide >= std::numeric_limits<Out>::lowest()
                  && wide <= std::numeric_limits<Out>::max())
                || wide != std::trunc(wide)) {
                throw std::runtime_error("Contract array value is outside the requested integer type");
            }
        }
        return static_cast<Out>(value);
    }

    std::string digest_;
    std::vector<Descriptor> arrays_;
    std::vector<unsigned char> blob_;
};

}  // namespace sp2
