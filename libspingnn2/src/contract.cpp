#include "contract.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/file.h>
#include <unistd.h>

namespace sp2 {
namespace {

class DirectoryLock {
public:
    explicit DirectoryLock(const std::filesystem::path& directory) {
        fd_ = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(), "Open contract directory");
        }
        int status;
        do {
            status = ::flock(fd_, LOCK_SH);
        } while (status < 0 && errno == EINTR);
        if (status < 0) {
            const auto error = errno;
            ::close(fd_);
            throw std::system_error(error, std::generic_category(), "Lock contract directory");
        }
    }

    ~DirectoryLock() { ::close(fd_); }
    DirectoryLock(const DirectoryLock&) = delete;
    DirectoryLock& operator=(const DirectoryLock&) = delete;

private:
    int fd_;
};

std::vector<unsigned char> read_blob(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::uintmax_t>(size) > std::numeric_limits<std::size_t>::max()
        || size > std::numeric_limits<std::streamsize>::max()) {
        throw std::runtime_error("Cannot open or size contract binary " + path.string());
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (size > 0 && !stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Cannot read contract binary " + path.string());
    }
    return bytes;
}

std::size_t index(const Json& value, const char* field) {
    if (!value.is_number_integer()
        || (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)
        || value.get<std::uint64_t>() > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("Invalid contract ") + field);
    }
    return value.get<std::size_t>();
}

}  // namespace

std::string sha256_bytes(const std::vector<unsigned char>& bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), digest, &size, EVP_sha256(), nullptr) != 1
        || size != 32) {
        throw std::runtime_error("Cannot calculate contract SHA-256");
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (unsigned int i = 0; i < size; ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 15];
    }
    return result;
}

Contract::Contract(const std::filesystem::path& manifest) {
    Json header;
    const auto directory = manifest.has_parent_path() ? manifest.parent_path() : ".";
    {
        // The exporter replaces the two conventional filenames under an
        // exclusive flock. Open both while holding its shared counterpart.
        const DirectoryLock lock(directory);
        const auto bytes = read_blob(manifest);
        digest_ = sha256_bytes(bytes);
        header = Json::parse(bytes);
        if (header.at("format").get<std::string>() != "spingnn2.contract.v1"
            || index(header.at("format_version"), "version") != 1) {
            throw std::runtime_error("Unsupported SpinGNN2 contract format or version");
        }
        const std::filesystem::path filename = header.at("binary").get<std::string>();
        if (filename.empty() || filename != filename.filename() || filename == "." || filename == "..") {
            throw std::runtime_error("Invalid contract binary filename");
        }
        blob_ = read_blob(directory / filename);
    }
    if (sha256_bytes(blob_) != header.at("sha256").get<std::string>()) {
        throw std::runtime_error("SpinGNN2 binary SHA-256 mismatch");
    }

    struct TypeRecord { const char* code; Dtype dtype; std::size_t width; };
    constexpr std::array<TypeRecord, 8> types{{
        {"<f8", Dtype::f64, 8}, {"<f4", Dtype::f32, 4},
        {"<i8", Dtype::i64, 8}, {"<i4", Dtype::i32, 4},
        {"<i2", Dtype::i16, 2}, {"|i1", Dtype::i8, 1},
        {"|u1", Dtype::u8, 1}, {"|b1", Dtype::boolean, 1},
    }};
    const auto& arrays = header.at("arrays");
    if (!arrays.is_array()) {
        throw std::runtime_error("Invalid contract array descriptor list");
    }
    arrays_.reserve(arrays.size());
    for (const auto& item : arrays) {
        const auto code = item.at("dtype").get<std::string>();
        const auto type = std::find_if(types.begin(), types.end(), [&](const auto& entry) { return code == entry.code; });
        const auto offset = index(item.at("offset"), "array offset");
        const auto& dims = item.at("shape");
        if (type == types.end() || offset % 8 || !dims.is_array()) {
            throw std::runtime_error("Invalid contract array descriptor");
        }
        std::vector<std::size_t> shape;
        shape.reserve(dims.size());
        std::size_t count = 1;
        for (const auto& dim : dims) {
            shape.push_back(index(dim, "array dimension"));
        }
        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
            count = 0;
        } else {
            for (const auto dimension : shape) {
                if (dimension > std::numeric_limits<std::size_t>::max() / count) {
                    throw std::runtime_error("Contract array dimensions overflow");
                }
                count *= dimension;
            }
        }
        if (offset > blob_.size() || count > (blob_.size() - offset) / type->width) {
            throw std::runtime_error("Contract array exceeds binary bounds");
        }
        arrays_.push_back({offset, std::move(shape), count, type->width, type->dtype});
    }

    model = std::move(header.at("model"));
    const auto family = model.at("family").get<std::string>();
    if (family != "Allegro_nosoc" && family != "Allegro_soc" && family != "Allegro_born"
        && family != "Allegro_e2IP" && family != "Allegro_soc_e2IP") {
        throw std::runtime_error("Unsupported model family " + family);
    }
    const auto precision = model.at("dtype").get<std::string>();
    if (precision != "float32" && precision != "float64") {
        throw std::runtime_error("Unsupported contract model precision " + precision);
    }
}

const Contract::Descriptor& Contract::descriptor(const Json& ref) const {
    if (!ref.is_object() || ref.size() != 1 || !ref.contains("$array")) {
        throw std::runtime_error("Invalid contract array reference");
    }
    const auto id = index(ref.at("$array"), "array reference");
    if (id >= arrays_.size()) {
        throw std::runtime_error("Invalid contract array reference");
    }
    return arrays_[id];
}

}  // namespace sp2
