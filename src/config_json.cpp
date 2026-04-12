#include "config_json.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace volt {

namespace {

void skip_ws(std::string_view s, std::size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

bool parse_string(std::string_view s, std::size_t& i, std::string& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') {
        return false;
    }
    ++i;
    const std::size_t start = i;
    while (i < s.size() && s[i] != '"') {
        ++i;
    }
    if (i >= s.size()) {
        return false;
    }
    out.assign(s.substr(start, i - start));
    ++i;
    return true;
}

bool parse_number(std::string_view s, std::size_t& i, double& out) {
    skip_ws(s, i);
    if (i >= s.size()) {
        return false;
    }
    const std::string tmp(s.substr(i));
    char* end = nullptr;
    const double v = std::strtod(tmp.c_str(), &end);
    if (end == tmp.c_str()) {
        return false;
    }
    out = v;
    i += static_cast<std::size_t>(end - tmp.c_str());
    return true;
}

void apply_key(std::string_view key, double v, Config& c) {
    if (key == "G_min") {
        c.G_min = static_cast<float>(v);
    } else if (key == "G_max") {
        c.G_max = static_cast<float>(v);
    } else if (key == "V_min") {
        c.V_min = static_cast<float>(v);
    } else if (key == "V_max") {
        c.V_max = static_cast<float>(v);
    } else if (key == "I_min") {
        c.I_min = static_cast<float>(v);
    } else if (key == "I_range") {
        c.I_range = static_cast<float>(v);
    } else if (key == "n_bits_adc") {
        c.n_bits_adc = static_cast<int>(std::lround(v));
    } else if (key == "noise_stddev") {
        c.noise_stddev = static_cast<float>(v);
    } else if (key == "disturb_ratio") {
        c.disturb_ratio = static_cast<float>(v);
    } else if (key == "disturb_alpha") {
        c.disturb_alpha = static_cast<float>(v);
    } else if (key == "noise_seed") {
        c.noise_seed = static_cast<unsigned int>(std::llround(v));
    } else if (key == "activation_sigmoid_steepness") {
        c.activation_sigmoid_steepness = static_cast<float>(v);
    } else if (key == "write_endurance_lambda") {
        c.write_endurance_lambda = static_cast<float>(v);
    }
}

bool parse_object(std::string_view s, std::size_t& i, Config& c, std::string& err) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{') {
        err = "JSON: expected '{'";
        return false;
    }
    ++i;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}') {
        ++i;
        return true;
    }
    for (;;) {
        std::string key;
        if (!parse_string(s, i, key)) {
            err = "JSON: expected string key";
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') {
            err = "JSON: expected ':' after key";
            return false;
        }
        ++i;
        double num = 0.0;
        if (!parse_number(s, i, num)) {
            err = "JSON: expected number for key \"" + key + "\"";
            return false;
        }
        apply_key(key, num, c);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}') {
            ++i;
            return true;
        }
        err = "JSON: expected ',' or '}'";
        return false;
    }
}

}  // namespace

bool load_config_from_json(std::string_view text, Config& base, std::string& err) {
    std::size_t i = 0;
    if (!parse_object(text, i, base, err)) {
        return false;
    }
    skip_ws(text, i);
    if (i != text.size()) {
        err = "JSON: trailing data after object";
        return false;
    }
    return true;
}

bool load_config_from_json_file(const std::string& path, Config& base, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open config file: " + path;
        return false;
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    if (!f && !f.eof()) {
        err = "read error: " + path;
        return false;
    }
    if (f.bad()) {
        err = "read error: " + path;
        return false;
    }
    const std::string content = buf.str();
    return load_config_from_json(content, base, err);
}

}  // namespace volt
