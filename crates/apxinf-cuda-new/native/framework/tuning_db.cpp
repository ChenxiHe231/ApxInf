// Best-effort persistent storage for exact-key recipes.
//
// The file name is an FNV-1a hash of the caller-provided exact key. The file
// stores the full key on its first line and the opaque serialized recipe on its
// second, so reads still reject hash collisions and truncated content. Writes
// use a unique temporary file, fsync, rename and a directory fsync. Empty directories,
// missing or malformed files, and I/O failures behave as cache misses; callers
// own recipe serialization, parsing, and implementation validation.
#include "runtime_internal.h"

#include <filesystem>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace apxinf::framework {
namespace {

std::string path(const std::string& directory, const std::string& key) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char character : key) {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << directory << '/' << std::hex << hash << ".recipe";
  return output.str();
}

}  // namespace

std::string read_recipe(const std::string& directory, const std::string& key) {
  if (directory.empty()) return {};
  std::ifstream input(path(directory, key));
  std::string stored_key;
  std::string recipe;
  if (!std::getline(input, stored_key) || stored_key != key ||
      !std::getline(input, recipe)) {
    return {};
  }
  return recipe;
}

void write_recipe(const std::string& directory, const std::string& key,
                  const std::string& recipe) {
  if (directory.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) return;
  const auto destination = path(directory, key);
  std::string temporary = destination + ".XXXXXX";
  const int fd = mkstemp(temporary.data());
  if (fd < 0) return;
  const std::string contents = key + '\n' + recipe + '\n';
  size_t offset = 0;
  bool complete = true;
  while (offset < contents.size()) {
    const auto count = ::write(fd, contents.data() + offset,
                               contents.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      complete = false;
      break;
    }
    offset += static_cast<size_t>(count);
  }
  if (complete) complete = ::fsync(fd) == 0;
  if (::close(fd) != 0) complete = false;
  if (!complete) {
    ::unlink(temporary.c_str());
    return;
  }
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    ::unlink(temporary.c_str());
    return;
  }
  const int directory_fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
  if (directory_fd >= 0) {
    ::fsync(directory_fd);
    ::close(directory_fd);
  }
}

}  // namespace apxinf::framework
