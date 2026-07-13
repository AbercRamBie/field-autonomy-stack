#include "autonomy/types.hpp"

#include <opencv2/core.hpp>

namespace autonomy {

class LaneDetector {
public:
    LaneDetector(
        double pixels_per_metre = 75.0,
        int horizon_y_px = 150
    );

    [[nodiscard]] LaneObservation detect(
        const cv::Mat& frame,
        cv::Mat* debug_frame = nullptr
    ) const;

private:
    double pixels_per_metre_;
    int horizon_y_px_;
};

}  // namespace autonomy