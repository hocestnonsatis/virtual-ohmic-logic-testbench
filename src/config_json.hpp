#pragma once

#include "config.hpp"

#include <string>
#include <string_view>

namespace volt {

/// Parses a JSON object (string keys, numeric values) and updates [base] in place.
/// Unknown keys are ignored. On failure, [err] is set and [base] may be partially updated.
bool load_config_from_json(std::string_view text, Config& base, std::string& err);

bool load_config_from_json_file(const std::string& path, Config& base, std::string& err);

}  // namespace volt
