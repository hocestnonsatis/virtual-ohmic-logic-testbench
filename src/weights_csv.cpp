#include "weights_csv.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

namespace volt {

namespace {

void trim_inplace(std::string& s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
}

bool parse_line_row(const std::string& line, std::vector<double>& row, std::string& err) {
    row.clear();
    std::size_t pos = 0;
    while (pos < line.size()) {
        std::size_t comma = line.find(',', pos);
        const std::size_t end = (comma == std::string::npos) ? line.size() : comma;
        std::string tok = line.substr(pos, end - pos);
        trim_inplace(tok);
        if (tok.empty()) {
            err = "CSV: empty field in row";
            return false;
        }
        char* p_end = nullptr;
        const double v = std::strtod(tok.c_str(), &p_end);
        if (p_end == tok.c_str()) {
            err = "CSV: invalid number: " + tok;
            return false;
        }
        row.push_back(v);
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return !row.empty();
}

}  // namespace

bool load_weights_csv_file(const std::string& path, std::vector<std::vector<double>>& out,
                           std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open weights file: " + path;
        return false;
    }
    out.clear();
    std::string line;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        trim_inplace(line);
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::vector<double> row;
        if (!parse_line_row(line, row, err)) {
            err = "line " + std::to_string(line_no) + ": " + err;
            return false;
        }
        if (!out.empty() && row.size() != out.front().size()) {
            err = "CSV: row width mismatch at line " + std::to_string(line_no);
            return false;
        }
        out.push_back(std::move(row));
    }
    if (out.empty()) {
        err = "CSV: no data rows";
        return false;
    }
    const std::size_t n = out.size();
    if (n != out[0].size()) {
        err = "CSV: matrix must be square";
        return false;
    }
    if (n > static_cast<std::size_t>(k_max_weights_dim)) {
        err = "CSV: matrix dimension exceeds k_max_weights_dim";
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double& w = out[i][j];
            if (w < -1.0 || w > 1.0) {
                std::cerr << "[weights_csv] warning: entry (" << i << "," << j << ") = " << w
                          << " outside [-1,1]; clamping\n";
                w = std::max(-1.0, std::min(1.0, w));
            }
        }
    }
    return true;
}

bool load_inputs_csv_file(const std::string& path, int expected_n, std::vector<float>& out,
                          std::string& err) {
    if (expected_n < 1) {
        err = "inputs: expected_n must be positive";
        return false;
    }
    std::ifstream f(path);
    if (!f) {
        err = "cannot open inputs file: " + path;
        return false;
    }
    std::vector<double> acc;
    std::string line;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        trim_inplace(line);
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::vector<double> row;
        if (!parse_line_row(line, row, err)) {
            err = "line " + std::to_string(line_no) + ": " + err;
            return false;
        }
        for (double x : row) {
            acc.push_back(x);
        }
    }
    if (acc.size() != static_cast<std::size_t>(expected_n)) {
        err = "inputs: expected " + std::to_string(expected_n) + " values, got " +
              std::to_string(acc.size());
        return false;
    }
    out.resize(static_cast<std::size_t>(expected_n));
    for (int i = 0; i < expected_n; ++i) {
        float v = static_cast<float>(acc[static_cast<std::size_t>(i)]);
        if (v < 0.0f || v > 1.0f) {
            std::cerr << "[inputs_csv] warning: input " << i << " = " << v
                      << " outside [0,1]; clamping\n";
            v = std::max(0.0f, std::min(1.0f, v));
        }
        out[static_cast<std::size_t>(i)] = v;
    }
    return true;
}

}  // namespace volt
