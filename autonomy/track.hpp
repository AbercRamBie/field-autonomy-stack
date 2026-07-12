#pragma once

namespace autonomy {

class Track {
public:
    [[nodiscard]] double curvatureAt(double distance_m) const;
    [[nodiscard]] double laneWidthM() const noexcept;

private:
    double lane_width_m_{3.5};
};

}  // namespace autonomy