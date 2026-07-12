// Placeholder
#include "autonomy/track.hpp"

#include <cmath>

namespace autonomy {

double Track::curvatureAt(const double distance_m) const {
    const double long_curve = 0.012 * std::sin(distance_m / 24.0);
    const double short_curve = 0.004 * std::sin(distance_m / 7.0);

    return long_curve + short_curve;
}

double Track::laneWidthM() const noexcept {
    return lane_width_m_;
}

}  // namespace autonomy