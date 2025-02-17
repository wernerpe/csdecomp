#include <Eigen/Sparse>

#include "minimal_plant.h"

namespace csdecomp {
Eigen::SparseMatrix<uint8_t> VisibilityGraph(
    const Eigen::MatrixXf& configurations, const MinimalPlant& mplant,
    const float step_size, const uint64_t max_configs_per_batch = 10000000);
}
