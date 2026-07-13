#pragma once

#include "autonomy/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace autonomy {

struct RenderConfig {
    int width_px{640};
    int height_px{480};
    int horizon_y_px{150};

    double pixels_per_metre{75.0};
    double heading_pixels_per_rad{280.0};
};

class SceneRenderer {
public:
    explicit SceneRenderer(RenderConfig config = {});

    [[nodiscard]] cv::Mat render(
        const VehicleState& state,
        double lane_width_m,
        const FaultState& faults
    );

private:
    RenderConfig config_;
};

}  // namespace autonomy