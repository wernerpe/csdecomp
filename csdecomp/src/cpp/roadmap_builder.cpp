// Copyright 2024 Toyota Research Institute.  All rights reserved.
#include "roadmap_builder.h"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "csdecomp/src/cuda/cuda_collision_checker.h"
#include "csdecomp/src/cuda/cuda_utilities.h"
#include "csdecomp/src/cuda/drm_cuda_utils.h"

namespace {
bool SortPairByFirst(const std::pair<double, int32_t>& a,
                     const std::pair<double, int32_t>& b) {
  return (a.first < b.first);
}

size_t CountEdgesInAdjacencyMap(
    const std::unordered_map<int32_t, std::vector<int32_t>>& adjacency_map) {
  size_t total = 0;
  for (const auto& [_, neighbors] : adjacency_map) {
    total += neighbors.size();
  }
  return total;
}

}  // namespace

namespace csdecomp {
RoadmapBuilder::RoadmapBuilder(const Plant& plant,
                               const std::string& target_link_name,
                               const RoadmapOptions& options) {
  tree_ = plant.getKinematicTree();
  mplant_ = plant.getMinimalPlant();
  inspector_ = plant.getSceneInspector();
  domain_.MakeBox(tree_.getPositionLowerLimits(),
                  tree_.getPositionUpperLimits());
  configuration_dimension_ = tree_.numConfigurationVariables();

  target_link_index_ = tree_.getLinkIndexByName(target_link_name);
  drm_.target_link_name_ = target_link_name;
  drm_.target_link_index_ = target_link_index_;
  drm_.options = options;
}

void RoadmapBuilder::GetRandomSamples(const int num_samples) {
  constexpr int kMixingSteps = 50;
  const Eigen::VectorXf initial_sample = domain_.ChebyshevCenter();
  int32_t node_counter = 1;
  for (int ii = 0; ii < num_samples; ++ii) {
    Eigen::VectorXf random_sample =
        domain_.UniformSample(initial_sample, kMixingSteps);
    if (checkCollisionFree(random_sample, mplant_)) {
      raw_node_data_.insert({node_counter, random_sample});
      ++node_counter;
    }
  }
  std::cout << "Generated " << raw_node_data_.size() << " random samples."
            << std::endl;

  // TODO(richard.cheng): Add heuristics for pruning out "bad" samples (e.g. tip
  // pose pointing backwards).
}

int RoadmapBuilder::AddNodesManual(const Eigen::MatrixXf& nodes_to_add) {
  assert(tree_.numConfigurationVariables() == nodes_to_add.rows());
  int num_added = 0;
  const int num_nodes_in_roadmap = raw_node_data_.size();
  for (int i = 0; i < nodes_to_add.cols(); ++i) {
    if (!domain_.PointInSet(nodes_to_add.col(i))) {
      std::cout << fmt::format(
          "sample {} is not in the domain and will be ignored!\n", i);
    } else if (checkCollisionFree(nodes_to_add.col(i), mplant_)) {
      num_added++;
      raw_node_data_.insert(
          {num_nodes_in_roadmap + num_added, nodes_to_add.col(i)});
    }
  }
  std::cout << fmt::format(
      "Added {} collision-free nodes to the existing {} nodes total is {}\n",
      num_added, num_nodes_in_roadmap, raw_node_data_.size());

  // TODO(richard.cheng): Add heuristics for pruning out "bad" samples (e.g. tip
  // pose pointing backwards).
  return num_added;
}

void RoadmapBuilder::Write(const std::string& file_path) {
  drm_.Write(file_path);
}

void RoadmapBuilder::Reset() {
  drm_.Clear();
  drm_.options = RoadmapOptions();
}

void RoadmapBuilder::Read(const std::string& file_path) {
  drm_.Read(file_path);
}

void RoadmapBuilder::BuildRoadmap(int32_t max_neighbors) {
  const size_t num_nodes = raw_node_data_.size();
  std::setbuf(stdout, nullptr);  // Disable buffering for stdout
  std::string tmp =
      fmt::format(
          "Building roadmap with {} nodes, limiting to {} max neighbors\n",
          num_nodes, max_neighbors) +
      "\n";
  printf(tmp.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  //   std::cout
  //       << fmt::format(
  //              "Building roadmap with {} nodes, limiting to {} max
  //              neighbors\n", num_nodes, max_neighbors)
  //       << std::endl
  //       << std::flush;
  // Allocate space and copy joints to GPU for checking joint distances.

  // Initialize joint IDs.
  std::vector<float> configurations_cpu_float;
  std::vector<int> node_id_cpu;
  for (const auto& element : raw_node_data_) {
    node_id_cpu.push_back(element.first);
    for (int ii = 0; ii < configuration_dimension_; ++ii) {
      configurations_cpu_float.push_back(element.second[ii]);
    }
  }
  // Copy joint info to GPU.
  CudaPtr<const float> all_configs_ptr(configurations_cpu_float.data(),
                                       configurations_cpu_float.size());
  all_configs_ptr.copyHostToDevice();

  // Allocate memory for joint distances.
  std::vector<float> joint_distance_allocation(num_nodes);
  CudaPtr<float> joint_distances(joint_distance_allocation.data(), num_nodes);

  int32_t node_counter = 0;

  for (const auto& element : raw_node_data_) {
    // Compute joint distances between current joint and every other joint
    // in node dataset, and then sort joints by distance.

    // Copy start joint to GPU for comparison.
    std::vector<float> start_configs_float(configuration_dimension_);
    for (int ii = 0; ii < configuration_dimension_; ++ii) {
      start_configs_float[ii] = static_cast<float>(element.second[ii]);
    }
    CudaPtr<float> start_joints_ptr(start_configs_float.data(),
                                    configuration_dimension_);
    start_joints_ptr.copyHostToDevice();

    // Compute joint distances between all joints and start joint.
    ComputeJointDistances(all_configs_ptr.device, start_joints_ptr.device,
                          configuration_dimension_, static_cast<int>(num_nodes),
                          joint_distances.device);
    joint_distances.copyDeviceToHost();

    std::vector<std::pair<double, int32_t>> neighbors;
    for (size_t ii = 0; ii < num_nodes; ++ii) {
      if ((joint_distances.host)[ii] > std::numeric_limits<float>::epsilon() &&
          (joint_distances.host)[ii] <
              drm_.options.max_configuration_distance_between_nodes) {
        neighbors.push_back(
            std::make_pair((joint_distances.host)[ii], node_id_cpu[ii]));
      }
    }
    std::sort(neighbors.begin(), neighbors.end(), SortPairByFirst);

    // Get world_T_link transforms for start configuration.
    std::vector<Eigen::Matrix4f> world_T_link_frames_1 =
        tree_.computeLinkFrameToWorldTransforms(element.second);

    // Iterate through potential neighbors by distance, until we have
    // successfully connected max_neighbors nodes to the current joint.
    std::vector<int32_t> nearest_neighbors;
    for (int ii = 0; ii < static_cast<int>(neighbors.size()); ++ii) {
      if (static_cast<int>(nearest_neighbors.size()) >= max_neighbors) {
        break;
      }

      // Skip edge connection if translation between any link frames is too
      // large.
      Eigen::VectorXf neighbor_joints =
          raw_node_data_.at(std::get<1>(neighbors[ii]));
      std::vector<Eigen::Matrix4f> world_T_link_frames_2 =
          tree_.computeLinkFrameToWorldTransforms(neighbor_joints);
      if (world_T_link_frames_1.size() != world_T_link_frames_2.size()) {
        std::cout << "ERROR: Frame sizes do not match." << std::endl;
      }

      bool distance_between_nodes_too_large = false;
      for (size_t frame_idx = 0; frame_idx < world_T_link_frames_1.size();
           ++frame_idx) {
        if ((world_T_link_frames_1[frame_idx].block(0, 3, 3, 1) -
             world_T_link_frames_2[frame_idx].block(0, 3, 3, 1))
                .norm() > drm_.options.max_task_space_distance_between_nodes) {
          distance_between_nodes_too_large = true;
          break;
        }
      }
      if (distance_between_nodes_too_large) {
        continue;
      }

      // Skip edge connection if joint distance between two nodes is too large.
      float config_dist = (element.second - neighbor_joints).norm();

      if (config_dist > drm_.options.max_configuration_distance_between_nodes) {
        continue;
      }

      // only add nodes if edge is collision free
      // know endpoints are collision free!
      int t_steps = floor(config_dist / drm_.options.edge_step_size) - 2;
      if (t_steps > 0) {
        Eigen::MatrixXf edge_check_work(configuration_dimension_, t_steps);
        for (int step_idx = 0; step_idx < t_steps; ++step_idx) {
          float t = (step_idx + 1) / (1.0 * t_steps);
          edge_check_work.col(step_idx) =
              t * neighbor_joints + (1 - t) * element.second;
        }
        std::vector<uint8_t> result =
            checkCollisionFreeCuda(&edge_check_work, &mplant_);
        bool any_collision =
            std::any_of(result.begin(), result.end(), [](uint8_t value) {
              return value == 0;  // Assuming 0 represents false
            });
        if (!any_collision) {
          nearest_neighbors.push_back(std::get<1>(neighbors[ii]));
        }
      } else {
        // drm_.node_adjacency_map[(*it_1).first].push_back(std::get<1>(neighbors[ii]));
        // drm_.id_to_node_map[(*it_1).first] = (*it_1).second;

        nearest_neighbors.push_back(std::get<1>(neighbors[ii]));
      }
    }

    assert(static_cast<int>(nearest_neighbors.size()) <= max_neighbors);
    if (!nearest_neighbors.empty()) {
      ++node_counter;
      drm_.id_to_node_map.insert({element.first, element.second});
      drm_.node_adjacency_map.insert({element.first, nearest_neighbors});
    }

    if (node_counter % drm_.options.nodes_processed_before_debug_statement ==
        0) {
      std::cout << "[CSD:RoadmapBuilder] Nodes processed so far: "
                << node_counter << std::endl
                << std::flush;
    }
  }
  // make adjacency map symmetric
  std::vector<std::pair<int32_t, int32_t>> missing_edges;
  for (const auto& item : drm_.node_adjacency_map) {
    const int32_t id = item.first;
    for (const auto& neighbor : item.second) {
      auto it = std::find(drm_.node_adjacency_map.at(neighbor).begin(),
                          drm_.node_adjacency_map.at(neighbor).end(), id);
      if (it == drm_.node_adjacency_map.at(neighbor).end())
        missing_edges.push_back({neighbor, id});
    }
  }
  std::cout << fmt::format(" Correcting {} edges\n", missing_edges.size());

  for (const auto& e_corr : missing_edges) {
    if (drm_.node_adjacency_map.find(e_corr.first) ==
        drm_.node_adjacency_map.end()) {
      drm_.node_adjacency_map[e_corr.first] = std::vector<int32_t>();
    }

    drm_.node_adjacency_map.at(e_corr.first).push_back(e_corr.second);
  }
  std::cout << fmt::format("Finished correcting edges\n");
}

void RoadmapBuilder::BuildPoseMap() {
  drm_.id_to_pose_map.clear();

  for (const auto& element : drm_.id_to_node_map) {
    std::vector<Eigen::Matrix4f> world_T_link_frames =
        tree_.computeLinkFrameToWorldTransforms(element.second);
    drm_.id_to_pose_map.insert(
        {element.first, world_T_link_frames[target_link_index_]});
  }
  std::cout << "Pose map has size " << drm_.id_to_pose_map.size() << std::endl
            << std::flush;
}

void RoadmapBuilder::BuildCollisionMap() {
  assert(drm_.options.offline_voxel_resolution >
         std::numeric_limits<double>::epsilon());
  auto robot_geometry_ids = inspector_.robot_geometry_ids;

  // Populate voxels throughout workspace with corresponding voxel IDs for
  // building collision map.
  std::vector<Eigen::Vector3f> root_p_voxels;
  std::vector<int> root_p_voxel_ids;
  int nx = ceil(drm_.options.robot_map_size_x /
                (2 * drm_.options.offline_voxel_resolution));
  int ny = ceil(drm_.options.robot_map_size_y /
                (2 * drm_.options.offline_voxel_resolution));
  int nz = ceil(drm_.options.robot_map_size_z /
                (2 * drm_.options.offline_voxel_resolution));
  int num_vox_tot = nx * ny * nz;
  for (int idx = 0; idx < num_vox_tot; ++idx) {
    root_p_voxels.push_back(GetCollisionVoxelCenter(idx, drm_.options));
    root_p_voxel_ids.push_back(idx);
  }

  // Copy voxels to format for CUDA collision checking.
  float voxel_radius = static_cast<float>(drm_.options.GetOfflineVoxelRadius());
  Voxels voxels(3, root_p_voxels.size());
  for (size_t ii = 0; ii < root_p_voxels.size(); ++ii) {
    voxels.col(ii) = root_p_voxels[ii];
  }
  std::cout << fmt::format("num voxels to check against {}",
                           root_p_voxels.size());
  // Copy configurations to format for CUDA collision checking.
  const size_t num_configurations = drm_.id_to_node_map.size();
  std::unordered_map<int32_t, int32_t> col_to_joint_id;
  Eigen::MatrixXf qfree(configuration_dimension_, num_configurations);
  int node_id = 0;
  for (const auto& element : drm_.id_to_node_map) {
    qfree.col(node_id) = element.second;
    col_to_joint_id.insert({node_id, element.first});
    ++node_id;
  }

  // load balancing
  const uint64_t computation_chunk_min_size =
      robot_geometry_ids.size() * voxels.cols();
  const uint64_t budget = 100000000;
  const uint64_t required_computation =
      num_configurations * computation_chunk_min_size;
  uint64_t num_config_chunks = required_computation / budget;

  // Collision results are stored as |config_0 - voxel_0 | config_0 - voxel_1
  // |
  // ... | config_0 - voxel_n | config_1 - voxel_0 | ... | config_n - voxel_n|
  std::vector<uint8_t> collision_free_results;
  if (num_config_chunks == 0) {
    collision_free_results = checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
        &qfree, &voxels, voxel_radius, &mplant_, robot_geometry_ids);
  } else {
    uint64_t num_configs_per_chunk = num_configurations / num_config_chunks;
    if (num_configs_per_chunk == 0) {
      throw std::runtime_error(
          " Not enough memory budget allocated. Try increasing. Warning this "
          "may "
          "cause Malloc errors if chosen too high.");
    }

    std::cout << fmt::format(
        "Min chunk size: {} \n num configs: {}\n allocated budget: {}\n num "
        "config chunks: {}\n num configs per chunk:{}\n",
        computation_chunk_min_size, num_configurations, budget,
        num_config_chunks, num_configs_per_chunk);
    std::vector<uint8_t> tmp_results;
    int num_configs_covered = 0;
    for (int batch_idx = 0; batch_idx < num_config_chunks; ++batch_idx) {
      Eigen::MatrixXf config_batch =
          qfree
              .block(0, batch_idx * num_configs_per_chunk,
                     configuration_dimension_, num_configs_per_chunk)
              .eval();
      tmp_results = checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
          &config_batch, &voxels, voxel_radius, &mplant_, robot_geometry_ids);
      for (const auto& t : tmp_results) {
        collision_free_results.push_back(t);
      }
      num_configs_covered += num_configs_per_chunk;
      std::cout
          << fmt::format(
                 "[DrmBuilder] Collision map built for {}/{} configurations\n",
                 num_configs_covered, num_configurations)
          << std::flush;
    }
    if (num_configurations - num_configs_covered) {
      // cover the remainder if there is one

      Eigen::MatrixXf config_batch_final =
          qfree
              .block(0, num_config_chunks * num_configs_per_chunk,
                     configuration_dimension_,
                     num_configurations - num_configs_covered)
              .eval();
      tmp_results = checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
          &config_batch_final, &voxels, voxel_radius, &mplant_,
          robot_geometry_ids);
      for (const auto& t : tmp_results) {
        collision_free_results.push_back(t);
      }
    }
    std::cout << fmt::format(
        "[DrmBuilder] Finished collision checking work \n");
  }

  std::unordered_map<int32_t, std::unordered_set<int32_t>> collision_map;
  for (size_t voxel_index = 0; voxel_index < root_p_voxels.size();
       ++voxel_index) {
    int counter = 0;
    int noncounter = 0;
    for (size_t configuration_index = 0;
         configuration_index < num_configurations; ++configuration_index) {
      if (!collision_free_results.at(
              configuration_index * root_p_voxels.size() + voxel_index)) {
        ++counter;
        const int32_t& voxel_id = root_p_voxel_ids[voxel_index];
        const int32_t& joint_id = col_to_joint_id.at(configuration_index);
        if (collision_map.find(voxel_id) == collision_map.end()) {
          std::unordered_set<int32_t> nodes_in_collision;
          nodes_in_collision.insert(joint_id);
          collision_map.insert({voxel_id, nodes_in_collision});
        } else {
          collision_map.at(voxel_id).insert(joint_id);
        }
      } else {
        ++noncounter;
      }
    }
  }

  // Erase voxels in collision with all nodes, because they provide no value.
  std::vector<int32_t> elements_to_erase;
  for (const auto& element : collision_map) {
    if (element.second.size() >= drm_.id_to_node_map.size()) {
      elements_to_erase.push_back(element.first);
      //   std::cout << fmt::format("ID to erase {} \n", element.first);
    }
  }

  std::cout << "Erasing " << elements_to_erase.size() << " elements."
            << std::endl;
  for (const int32_t element_to_erase : elements_to_erase) {
    collision_map.erase(element_to_erase);
  }

  // Populate collision_map in PRM.
  drm_.collision_map.clear();
  for (const auto& element : collision_map) {
    std::vector<int32_t> nodes_in_collision;
    for (int32_t node_id : element.second) {
      nodes_in_collision.push_back(node_id);
    }
    drm_.collision_map.insert({element.first, nodes_in_collision});
  }
  std::cout << "Collision map has size " << drm_.collision_map.size()
            << std::endl;
}

void RoadmapBuilder::BuildEdgeCollisionMap(
    const float step_size_edge_collision_map) {
  assert(drm_.options.offline_voxel_resolution >
         std::numeric_limits<double>::epsilon());
  auto robot_geometry_ids = inspector_.robot_geometry_ids;
  // Populate voxels throughout workspace with corresponding voxel IDs for
  // building collision map.
  std::vector<Eigen::Vector3f> root_p_voxels;
  std::vector<int> root_p_voxel_ids;
  int nx = ceil(drm_.options.robot_map_size_x /
                (2 * drm_.options.offline_voxel_resolution));
  int ny = ceil(drm_.options.robot_map_size_y /
                (2 * drm_.options.offline_voxel_resolution));
  int nz = ceil(drm_.options.robot_map_size_z /
                (2 * drm_.options.offline_voxel_resolution));
  int num_vox_tot = nx * ny * nz;
  for (int idx = 0; idx < num_vox_tot; ++idx) {
    root_p_voxels.push_back(GetCollisionVoxelCenter(idx, drm_.options));
    root_p_voxel_ids.push_back(idx);
  }

  // Copy voxels to format for CUDA collision checking.
  float voxel_radius = static_cast<float>(drm_.options.GetOfflineVoxelRadius());
  Voxels voxels(3, root_p_voxels.size());
  for (size_t ii = 0; ii < root_p_voxels.size(); ++ii) {
    voxels.col(ii) = root_p_voxels[ii];
  }
  std::cout << fmt::format("num voxels to check against {}",
                           root_p_voxels.size());

  std::unordered_map<int32_t, std::unordered_set<std::vector<int32_t>,
                                                 roadmaputils::VectorHash>>
      edge_collision_map;
  // implement a first stupid version, then make fast.
  int ids_checked = 0;
  float time_cc = 0;
  float time_postprocess = 0;
  for (const auto& [id1, adjacent_ids] : drm_.node_adjacency_map) {
    if (ids_checked % 50 == 0)
      std::cout << fmt::format(
                       "[CSD:RoadmapBuilder] Outgoing edges of NodeIds checked "
                       "{}/{}.\n",
                       ids_checked, drm_.id_to_node_map.size())
                << std::flush;
    auto start_cc = std::chrono::high_resolution_clock::now();
    // set up the edge check work and determine batching
    std::vector<int32_t> num_configs_to_check_for_neighbor;
    for (const auto& id2 : adjacent_ids) {
      if (id1 < id2) {
        Eigen::VectorXf node1 = drm_.id_to_node_map[id1];
        Eigen::VectorXf node2 = drm_.id_to_node_map[id2];
        float config_dist = (node2 - node1).norm();
        int t_steps = std::max(
            0.0f, floor(config_dist / step_size_edge_collision_map) - 2);
        num_configs_to_check_for_neighbor.push_back(t_steps);
      } else {
        num_configs_to_check_for_neighbor.push_back(0);
      }
    }

    int32_t num_configs_edge_work = std::accumulate(
        num_configs_to_check_for_neighbor.begin(),
        num_configs_to_check_for_neighbor.end(), static_cast<int32_t>(0));

    if (num_configs_edge_work == 0) continue;

    Eigen::MatrixXf edge_check_work(configuration_dimension_,
                                    num_configs_edge_work);
    int cols_written = 0;
    for (const auto& id2 : adjacent_ids) {
      if (id1 < id2) {
        // interpolate edge
        Eigen::VectorXf node1 = drm_.id_to_node_map[id1];
        Eigen::VectorXf node2 = drm_.id_to_node_map[id2];
        float config_dist = (node2 - node1).norm();
        int t_steps = std::max(
            0.0f, floor(config_dist / step_size_edge_collision_map) - 2);
        if (t_steps > 0) {
          for (int step_idx = 0; step_idx < t_steps; ++step_idx) {
            float t = (step_idx + 1) / (1.0 * t_steps);
            edge_check_work.col(cols_written + step_idx) =
                t * node2 + (1 - t) * node1;
          }
          cols_written += t_steps;
        }
      }
    }

    // break up edge check work into batches if needed and run the collision
    // checker.
    std::vector<uint8_t> results_all_outgoing_edges;
    const int max_num_configs_per_batch = 500;
    int num_batches = num_configs_edge_work / max_num_configs_per_batch;

    int remainder_size =
        num_configs_edge_work - num_batches * max_num_configs_per_batch;
    // std::cout << "[CSD:RoadmapBuilder] Build edge collision set num batches "
    //           << num_batches << " num configs " << num_configs_edge_work
    //           << std::endl;
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      // get configs of batch
      Eigen::MatrixXf batch_configs = Eigen::Map<Eigen::MatrixXf>(
          edge_check_work.data() +
              configuration_dimension_ * batch_idx * max_num_configs_per_batch,
          configuration_dimension_, max_num_configs_per_batch);
      std::vector<uint8_t> collision_free_results;
      // run collision checker
      collision_free_results = checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
          &batch_configs, &voxels, voxel_radius, &mplant_, robot_geometry_ids);
      // fill in results
      for (const auto& c : collision_free_results) {
        results_all_outgoing_edges.push_back(c);
      }
    }

    // handle remainder batch
    if (remainder_size > 0) {
      Eigen::MatrixXf remainder_configs = Eigen::Map<Eigen::MatrixXf>(
          edge_check_work.data() + configuration_dimension_ * num_batches *
                                       max_num_configs_per_batch,
          configuration_dimension_, remainder_size);
      std::vector<uint8_t> collision_free_results;
      collision_free_results = checkCollisionFreeVoxelsWithoutSelfCollisionCuda(
          &remainder_configs, &voxels, voxel_radius, &mplant_,
          robot_geometry_ids);
      // fill in results
      for (const auto& c : collision_free_results) {
        results_all_outgoing_edges.push_back(c);
      }
    }
    auto end_cc = std::chrono::high_resolution_clock::now();
    time_cc += (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_cc - start_cc)
                   .count();

    auto start_pp = std::chrono::high_resolution_clock::now();

    // handle evaulating results and filling in edge collision map
    int neigh_idx = 0;
    int starting_idx = 0;
    for (const auto& id2 : adjacent_ids) {
      int num_configs_in_edge = num_configs_to_check_for_neighbor[neigh_idx];
      if (num_configs_in_edge == 0) continue;

      for (int voxidx = 0; voxidx < num_vox_tot; ++voxidx) {
        for (int edgeconfigidx = 0; edgeconfigidx < num_configs_in_edge;
             ++edgeconfigidx) {
          int lookupindex = starting_idx + num_vox_tot * edgeconfigidx + voxidx;
          if (!results_all_outgoing_edges.at(lookupindex)) {
            edge_collision_map[voxidx].insert({id1, id2});
            edge_collision_map[voxidx].insert({id2, id1});
            break;
          }
        }
      }
      ++neigh_idx;
      starting_idx += num_configs_to_check_for_neighbor[neigh_idx];
    }

    ++ids_checked;
    auto end_pp = std::chrono::high_resolution_clock::now();
    time_postprocess +=
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(end_pp -
                                                                     start_pp)
            .count();
  }

  std::cout << fmt::format(" time cc {} time postprocess {} \n",
                           time_cc / 1000., time_postprocess / 1000.);
  // Erase entries that correspond to a single voxel disabling all edges
  size_t edges_in_drm = CountEdgesInAdjacencyMap(drm_.node_adjacency_map);

  std::vector<int32_t> elements_to_erase;
  for (const auto& [voxid, deactivated_edge_set] : edge_collision_map) {
    // check if deactivated_edge_map == drm_.node_adjacency_map
    if (deactivated_edge_set.size() == edges_in_drm) {
      elements_to_erase.push_back(voxid);
      //   std::cout << fmt::format("ID to erase {} \n", voxid);
    }
  }
  std::cout << "Erasing " << elements_to_erase.size() << " elements."
            << std::endl;
  for (const int32_t element_to_erase : elements_to_erase) {
    edge_collision_map.erase(element_to_erase);
  }

  // Populate drm_.edge_collision_map
  drm_.edge_collision_map.clear();
  for (const auto& [voxel_id, inner_set] : edge_collision_map) {
    auto& drm_inner_set = drm_.edge_collision_map[voxel_id];
    for (const auto& edge : inner_set) {
      drm_inner_set.insert(edge);
    }
  }
}

int32_t GetCollisionVoxelId(const Eigen::Vector3f& robot_p_voxel,
                            const RoadmapOptions& options) {
  int nx =
      ceil(options.robot_map_size_x / (2 * options.offline_voxel_resolution));
  int ny =
      ceil(options.robot_map_size_y / (2 * options.offline_voxel_resolution));
  int nz =
      ceil(options.robot_map_size_z / (2 * options.offline_voxel_resolution));

  float lx_true = nx * 2 * options.offline_voxel_resolution;
  float ly_true = ny * 2 * options.offline_voxel_resolution;
  float lz_true = nz * 2 * options.offline_voxel_resolution;
  Eigen::Vector3f pos_rel = robot_p_voxel - options.map_center;
  int x_bin_idx =
      floor(nx * (pos_rel.x() + options.robot_map_size_x / 2.0) / lx_true);
  int y_bin_idx =
      floor(ny * (pos_rel.y() + options.robot_map_size_y / 2.0) / ly_true);
  int z_bin_idx =
      floor(nz * (pos_rel.z() + options.robot_map_size_z / 2.0) / lz_true);
  if (x_bin_idx < 0 || y_bin_idx < 0 || z_bin_idx < 0) {
    return -1;
  }
  if (x_bin_idx > nx - 1 || y_bin_idx > ny - 1 || z_bin_idx > nz - 1) {
    return -1;
  }
  return x_bin_idx + y_bin_idx * nx + z_bin_idx * nx * ny;
}

Eigen::Vector3f GetCollisionVoxelCenter(const int32_t index,
                                        const RoadmapOptions& options) {
  if (index < 0) {
    throw std::runtime_error(fmt::format(
        "[RoadmapBuilder:GetCollisionVoxelCenter] invalid index {}\n.", index));
  }

  int nx =
      ceil(options.robot_map_size_x / (2 * options.offline_voxel_resolution));
  int ny =
      ceil(options.robot_map_size_y / (2 * options.offline_voxel_resolution));
  // int nz =
  //     ceil(options.robot_map_size_z / (2 *
  //     options.offline_voxel_resolution));

  int iz = index / (nx * ny);
  int iy = (index - nx * ny * iz) / nx;
  int ix = index - nx * ny * iz - nx * iy;

  float x = options.map_center[0] - options.robot_map_size_x / 2.0 +
            ix * 2 * options.offline_voxel_resolution +
            options.offline_voxel_resolution;

  float y = options.map_center[1] - options.robot_map_size_y / 2.0 +
            iy * 2 * options.offline_voxel_resolution +
            options.offline_voxel_resolution;

  float z = options.map_center[2] - options.robot_map_size_z / 2.0 +
            iz * 2 * options.offline_voxel_resolution +
            options.offline_voxel_resolution;

  return {x, y, z};
}
}  // namespace csdecomp