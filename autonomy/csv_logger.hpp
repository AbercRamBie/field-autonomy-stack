#pragma once

#include "autonomy/types.hpp"

#include <fstream>
#include <string>

namespace autonomy {

class CsvLogger {
public:
    explicit CsvLogger(const std::string& output_path);

    void write(const TelemetryRecord& record);

private:
    std::ofstream output_;
};

}  // namespace autonomy