#include <glim_ros/glim_ros.hpp>

#define GLIM_ROS2

#include <deque>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <future>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <std_msgs/msg/int64.hpp>
#include <lifecycle_msgs/srv/change_state.hpp>
#include <lifecycle_msgs/srv/get_state.hpp>
#include <lifecycle_msgs/msg/transition.hpp>
#include <lifecycle_msgs/msg/state.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <glim/util/debug.hpp>
#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/util/ros_cloud_converter.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <glim/mapping/async_sub_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim_ros/ros_compatibility.hpp>
#include <glim_ros/ros_qos.hpp>

namespace glim {

namespace {

// Shared between ensure_resple_fresh_before_configuring() (reader) and
// GlimROS::resple_start_time_changed_callback() (writer) -- see the comment
// on kSyncedMarkerPath's use below for why this file exists at all.
constexpr const char* kSyncedMarkerPath = "/tmp/glim_ros_synced_resple_start_time_ns";

template <typename T>
T declare_and_get(rclcpp_lifecycle::LifecycleNode& node, const std::string& name, const T& default_value) {
  if (!node.has_parameter(name)) {
    node.declare_parameter<T>(name, default_value);
  }

  T value = default_value;
  node.get_parameter(name, value);
  return value;
}

// For the handful of fields that genuinely differ between mapping and
// localization mode (so_name, extension_modules -- everything else is
// reconciled to one shared value, see robots/README.md). Declares BOTH
// mode's parameters unconditionally under glim_profiles.<mode>.<field_name>
// -- ROS2 requires a parameter be declared before use, and declaring the
// inactive mode's value too is harmless, it's just never applied -- then
// returns whichever one matches the live glim_mode.
template <typename T>
T declare_and_get_for_mode(
  rclcpp_lifecycle::LifecycleNode& node,
  const std::string& mode,
  const std::string& field_name,
  const T& mapping_default,
  const T& localization_default)
{
  const T mapping_value = declare_and_get<T>(node, "glim_profiles.mapping." + field_name, mapping_default);
  const T localization_value = declare_and_get<T>(node, "glim_profiles.localization." + field_name, localization_default);
  return mode == "localization" ? localization_value : mapping_value;
}

nlohmann::json load_json(const boost::filesystem::path& path) {
  std::ifstream stream(path.string());
  if (!stream) {
    throw std::runtime_error("failed to open " + path.string());
  }

  nlohmann::json json;
  stream >> json;
  return json;
}

void save_json(const boost::filesystem::path& path, const nlohmann::json& json) {
  std::ofstream stream(path.string());
  if (!stream) {
    throw std::runtime_error("failed to write " + path.string());
  }

  stream << std::setw(2) << json << std::endl;
}

void copy_config_directory(const boost::filesystem::path& src, const boost::filesystem::path& dst) {
  if (boost::filesystem::exists(dst)) {
    boost::filesystem::remove_all(dst);
  }
  boost::filesystem::create_directories(dst);

  for (const auto& entry : boost::filesystem::directory_iterator(src)) {
    if (!boost::filesystem::is_regular_file(entry.status())) {
      continue;
    }
    boost::filesystem::copy_file(
      entry.path(), dst / entry.path().filename(),
      boost::filesystem::copy_option::overwrite_if_exists);
  }
}

std::string create_effective_config_from_ros_params(
  rclcpp_lifecycle::LifecycleNode& node,
  const std::string& base_config_path)
{
  const auto base_path = boost::filesystem::path(base_config_path);
  const auto effective_path =
    boost::filesystem::temp_directory_path() /
    ("glim_ros_" + std::string(node.get_name()) + "_" + std::to_string(getpid()));
  copy_config_directory(base_path, effective_path);

  // Selects between glim_profiles.mapping.* and glim_profiles.localization.*
  // for the handful of fields below that genuinely differ by mode (so_name,
  // extension_modules) -- base_config_path itself no longer needs to point at
  // a mode-specific directory; one shared template covers both modes now.
  const std::string glim_mode = declare_and_get<std::string>(node, "glim_mode", "mapping");

  auto config_ros = load_json(effective_path / "config_ros.json");
  auto& glim_ros = config_ros["glim_ros"];
  glim_ros["enable_local_mapping"] =
    declare_and_get<bool>(node, "glim_ros.enable_local_mapping", glim_ros.value("enable_local_mapping", true));
  glim_ros["enable_global_mapping"] =
    declare_and_get<bool>(node, "glim_ros.enable_global_mapping", glim_ros.value("enable_global_mapping", true));
  glim_ros["keep_raw_points"] =
    declare_and_get<bool>(node, "glim_ros.keep_raw_points", glim_ros.value("keep_raw_points", false));
  glim_ros["imu_time_offset"] =
    declare_and_get<double>(node, "glim_ros.imu_time_offset", glim_ros.value("imu_time_offset", 0.0));
  glim_ros["points_time_offset"] =
    declare_and_get<double>(node, "glim_ros.points_time_offset", glim_ros.value("points_time_offset", 0.0));
  glim_ros["acc_scale"] =
    declare_and_get<double>(node, "glim_ros.acc_scale", glim_ros.value("acc_scale", 1.0));
  glim_ros["imu_frame_id"] =
    declare_and_get<std::string>(node, "glim_ros.imu_frame_id", glim_ros.value("imu_frame_id", "imu"));
  glim_ros["lidar_frame_id"] =
    declare_and_get<std::string>(node, "glim_ros.lidar_frame_id", glim_ros.value("lidar_frame_id", "lidar"));
  glim_ros["base_frame_id"] =
    declare_and_get<std::string>(node, "glim_ros.base_frame_id", glim_ros.value("base_frame_id", "base_link"));
  glim_ros["odom_frame_id"] =
    declare_and_get<std::string>(node, "glim_ros.odom_frame_id", glim_ros.value("odom_frame_id", "odom"));
  glim_ros["map_frame_id"] =
    declare_and_get<std::string>(node, "glim_ros.map_frame_id", glim_ros.value("map_frame_id", "map"));
  glim_ros["publish_imu2lidar"] =
    declare_and_get<bool>(node, "glim_ros.publish_imu2lidar", glim_ros.value("publish_imu2lidar", true));
  glim_ros["publish_tf"] =
    declare_and_get<bool>(node, "glim_ros.publish_tf", glim_ros.value("publish_tf", true));
  glim_ros["tf_time_offset"] =
    declare_and_get<double>(node, "glim_ros.tf_time_offset", glim_ros.value("tf_time_offset", 1e-6));
  glim_ros["pose_corrected_odom_child_frame_id"] =
    declare_and_get<std::string>(
      node, "glim_ros.pose_corrected_odom_child_frame_id",
      glim_ros.value("pose_corrected_odom_child_frame_id", glim_ros.value("imu_frame_id", "imu")));
  glim_ros["pose_corrected_odom_covariance_diag"] =
    declare_and_get<std::vector<double>>(
      node, "glim_ros.pose_corrected_odom_covariance_diag",
      glim_ros.value(
        "pose_corrected_odom_covariance_diag",
        std::vector<double>{0.01, 0.01, 0.01, 0.0025, 0.0025, 0.0025}));
  // Which plugins load genuinely differs by mode: localization additionally
  // needs the reloc-capable global_mapping plugin and BBS3D itself.
  glim_ros["extension_modules"] =
    declare_and_get_for_mode<std::vector<std::string>>(
      node, glim_mode, "ros.extension_modules",
      glim_ros.value("extension_modules", std::vector<std::string>{"librviz_viewer.so", "libresple_spline_extension.so"}),
      std::vector<std::string>{"librviz_viewer.so", "libresple_spline_extension.so", "libglobal_mapping_reloc.so", "libglim_bbs3d.so"});
  glim_ros["image_topic"] =
    declare_and_get<std::string>(node, "glim_ros.image_topic", glim_ros.value("image_topic", "/image"));
  glim_ros["imu_topic"] =
    declare_and_get<std::string>(node, "glim_ros.imu_topic", glim_ros.value("imu_topic", "/imu"));
  glim_ros["points_topic"] =
    declare_and_get<std::string>(node, "glim_ros.points_topic", glim_ros.value("points_topic", "/points"));
  if (!glim_ros.contains("imu_qos") || !glim_ros["imu_qos"].is_object()) {
    glim_ros["imu_qos"] = nlohmann::json::object();
  }
  if (!glim_ros.contains("points_qos") || !glim_ros["points_qos"].is_object()) {
    glim_ros["points_qos"] = nlohmann::json::object();
  }
  glim_ros["imu_qos"]["profile"] =
    declare_and_get<std::string>(
      node, "glim_ros.imu_qos.profile", glim_ros["imu_qos"].value("profile", "sensor_data"));
  glim_ros["imu_qos"]["depth"] =
    declare_and_get<int>(node, "glim_ros.imu_qos.depth", glim_ros["imu_qos"].value("depth", 1000));
  glim_ros["points_qos"]["profile"] =
    declare_and_get<std::string>(
      node, "glim_ros.points_qos.profile", glim_ros["points_qos"].value("profile", "sensor_data"));
  save_json(effective_path / "config_ros.json", config_ros);

  auto config_sensors = load_json(effective_path / "config_sensors.json");
  auto& sensors = config_sensors["sensors"];
  sensors["imu_acc_noise"] =
    declare_and_get<double>(node, "sensors.imu_acc_noise", sensors.value("imu_acc_noise", 0.1));
  sensors["imu_gyro_noise"] =
    declare_and_get<double>(node, "sensors.imu_gyro_noise", sensors.value("imu_gyro_noise", 0.005));
  sensors["imu_int_noise"] =
    declare_and_get<double>(node, "sensors.imu_int_noise", sensors.value("imu_int_noise", 0.001));
  sensors["imu_bias_noise"] =
    declare_and_get<double>(node, "sensors.imu_bias_noise", sensors.value("imu_bias_noise", 1e-5));
  sensors["global_shutter_lidar"] =
    declare_and_get<bool>(node, "sensors.global_shutter_lidar", sensors.value("global_shutter_lidar", false));
  sensors["T_lidar_imu"] =
    declare_and_get<std::vector<double>>(
      node, "sensors.T_lidar_imu",
      sensors.value("T_lidar_imu", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));
  sensors["intensity_field"] =
    declare_and_get<std::string>(node, "sensors.intensity_field", sensors.value("intensity_field", "intensity"));
  sensors["ring_field"] =
    declare_and_get<std::string>(node, "sensors.ring_field", sensors.value("ring_field", ""));
  sensors["autoconf_perpoint_times"] =
    declare_and_get<bool>(
      node, "sensors.autoconf_perpoint_times", sensors.value("autoconf_perpoint_times", true));
  sensors["autoconf_prefer_frame_time"] =
    declare_and_get<bool>(
      node, "sensors.autoconf_prefer_frame_time", sensors.value("autoconf_prefer_frame_time", false));
  sensors["perpoint_relative_time"] =
    declare_and_get<bool>(node, "sensors.perpoint_relative_time", sensors.value("perpoint_relative_time", true));
  sensors["perpoint_time_scale"] =
    declare_and_get<double>(node, "sensors.perpoint_time_scale", sensors.value("perpoint_time_scale", 1.0));
  save_json(effective_path / "config_sensors.json", config_sensors);

  auto config_logging = load_json(effective_path / "config_logging.json");
  auto& logging = config_logging["logging"];
  logging["log_dir"] =
    declare_and_get<std::string>(node, "logging.log_dir", logging.value("log_dir", "/tmp/glim_log"));
  logging["save_logs"] =
    declare_and_get<bool>(node, "logging.save_logs", logging.value("save_logs", false));
  logging["rotate_logs"] =
    declare_and_get<bool>(node, "logging.rotate_logs", logging.value("rotate_logs", false));
  save_json(effective_path / "config_logging.json", config_logging);

  auto config_odometry = load_json(effective_path / "config_odometry.json");
  auto& odometry_estimation = config_odometry["odometry_estimation"];
  odometry_estimation["so_name"] =
    declare_and_get<std::string>(node, "odometry_estimation.so_name", odometry_estimation.value("so_name", "libodometry_estimation_cpu.so"));
  odometry_estimation["est_window_topic"] =
    declare_and_get<std::string>(node, "odometry_estimation.est_window_topic", odometry_estimation.value("est_window_topic", "/est_window"));
  odometry_estimation["start_time_topic"] =
    declare_and_get<std::string>(node, "odometry_estimation.start_time_topic", odometry_estimation.value("start_time_topic", "/start_time"));
  odometry_estimation["coverage_timeout_ms"] =
    declare_and_get<int>(node, "odometry_estimation.coverage_timeout_ms", odometry_estimation.value("coverage_timeout_ms", 300));
  odometry_estimation["deskew_sample_dt"] =
    declare_and_get<double>(node, "odometry_estimation.deskew_sample_dt", odometry_estimation.value("deskew_sample_dt", 0.01));
  odometry_estimation["covariance_estimation_num_threads"] =
    declare_and_get<int>(node, "odometry_estimation.covariance_estimation_num_threads", odometry_estimation.value("covariance_estimation_num_threads", 4));
  odometry_estimation["voxel_resolution"] =
    declare_and_get<double>(node, "odometry_estimation.voxel_resolution", odometry_estimation.value("voxel_resolution", 0.5));
  odometry_estimation["voxelmap_levels"] =
    declare_and_get<int>(node, "odometry_estimation.voxelmap_levels", odometry_estimation.value("voxelmap_levels", 2));
  odometry_estimation["voxelmap_scaling_factor"] =
    declare_and_get<double>(node, "odometry_estimation.voxelmap_scaling_factor", odometry_estimation.value("voxelmap_scaling_factor", 2.0));
  save_json(effective_path / "config_odometry.json", config_odometry);

  // so_name is the only global_mapping field that still genuinely differs by
  // mode (reconciled all the others -- enable_imu, registration_error_factor_type,
  // isam2_relinearize_skip, gpu_memory_offload_mb -- to one shared value each,
  // see robots/README.md); handled the same mode-aware way as extension_modules
  // above, not as a plain override below.
  auto config_global_mapping = load_json(effective_path / "config_global_mapping.json");
  auto& global_mapping = config_global_mapping["global_mapping"];
  global_mapping["so_name"] =
    declare_and_get_for_mode<std::string>(
      node, glim_mode, "global_mapping.so_name",
      global_mapping.value("so_name", "libglobal_mapping.so"),
      "libglobal_mapping_reloc.so");
  global_mapping["enable_imu"] =
    declare_and_get<bool>(node, "global_mapping.enable_imu", global_mapping.value("enable_imu", true));
  global_mapping["enable_optimization"] =
    declare_and_get<bool>(node, "global_mapping.enable_optimization", global_mapping.value("enable_optimization", true));
  global_mapping["init_pose_damping_scale"] =
    declare_and_get<double>(node, "global_mapping.init_pose_damping_scale", global_mapping.value("init_pose_damping_scale", 1e10));
  global_mapping["create_between_factors"] =
    declare_and_get<bool>(node, "global_mapping.create_between_factors", global_mapping.value("create_between_factors", true));
  global_mapping["between_registration_type"] =
    declare_and_get<std::string>(node, "global_mapping.between_registration_type", global_mapping.value("between_registration_type", "GICP"));
  global_mapping["registration_error_factor_type"] =
    declare_and_get<std::string>(node, "global_mapping.registration_error_factor_type", global_mapping.value("registration_error_factor_type", "VGICP_GPU"));
  global_mapping["randomsampling_rate"] =
    declare_and_get<double>(node, "global_mapping.randomsampling_rate", global_mapping.value("randomsampling_rate", 1.0));
  global_mapping["submap_voxel_resolution"] =
    declare_and_get<double>(node, "global_mapping.submap_voxel_resolution", global_mapping.value("submap_voxel_resolution", 1.0));
  global_mapping["submap_voxel_resolution_max"] =
    declare_and_get<double>(node, "global_mapping.submap_voxel_resolution_max", global_mapping.value("submap_voxel_resolution_max", 1.0));
  global_mapping["submap_voxel_resolution_dmin"] =
    declare_and_get<double>(node, "global_mapping.submap_voxel_resolution_dmin", global_mapping.value("submap_voxel_resolution_dmin", 5.0));
  global_mapping["submap_voxel_resolution_dmax"] =
    declare_and_get<double>(node, "global_mapping.submap_voxel_resolution_dmax", global_mapping.value("submap_voxel_resolution_dmax", 20.0));
  global_mapping["submap_voxelmap_levels"] =
    declare_and_get<int>(node, "global_mapping.submap_voxelmap_levels", global_mapping.value("submap_voxelmap_levels", 2));
  global_mapping["submap_voxelmap_scaling_factor"] =
    declare_and_get<double>(node, "global_mapping.submap_voxelmap_scaling_factor", global_mapping.value("submap_voxelmap_scaling_factor", 2.0));
  global_mapping["max_implicit_loop_distance"] =
    declare_and_get<double>(node, "global_mapping.max_implicit_loop_distance", global_mapping.value("max_implicit_loop_distance", 100.0));
  global_mapping["min_implicit_loop_overlap"] =
    declare_and_get<double>(node, "global_mapping.min_implicit_loop_overlap", global_mapping.value("min_implicit_loop_overlap", 0.1));
  global_mapping["use_isam2_dogleg"] =
    declare_and_get<bool>(node, "global_mapping.use_isam2_dogleg", global_mapping.value("use_isam2_dogleg", false));
  global_mapping["isam2_relinearize_skip"] =
    declare_and_get<int>(node, "global_mapping.isam2_relinearize_skip", global_mapping.value("isam2_relinearize_skip", 10));
  global_mapping["isam2_relinearize_thresh"] =
    declare_and_get<double>(node, "global_mapping.isam2_relinearize_thresh", global_mapping.value("isam2_relinearize_thresh", 0.1));
  global_mapping["gpu_memory_offload_mb"] =
    declare_and_get<int>(node, "global_mapping.gpu_memory_offload_mb", global_mapping.value("gpu_memory_offload_mb", 2560));
  // localization.* only means anything to libglobal_mapping_reloc.so - harmless
  // (unused) clutter in config_global_mapping.json when so_name selects the
  // plain libglobal_mapping.so instead, so it's fine to always write it out.
  auto& localization = config_global_mapping["localization"];
  localization["max_localization_distance"] =
    declare_and_get<double>(node, "localization.max_localization_distance", localization.value("max_localization_distance", 5.0));
  localization["min_localization_overlap"] =
    declare_and_get<double>(node, "localization.min_localization_overlap", localization.value("min_localization_overlap", 0.1));
  localization["linear_search_window"] =
    declare_and_get<double>(node, "localization.linear_search_window", localization.value("linear_search_window", 6.0));
  localization["angular_search_window"] =
    declare_and_get<double>(node, "localization.angular_search_window", localization.value("angular_search_window", 0.1));
  localization["relocalization_factor_weight"] =
    declare_and_get<double>(node, "localization.relocalization_factor_weight", localization.value("relocalization_factor_weight", 1e2));
  localization["loc_between_factor_weight"] =
    declare_and_get<double>(node, "localization.loc_between_factor_weight", localization.value("loc_between_factor_weight", 1e2));
  localization["num_keep_submaps"] =
    declare_and_get<int>(node, "localization.num_keep_submaps", localization.value("num_keep_submaps", 3));
  localization["max_localization_submaps"] =
    declare_and_get<int>(node, "localization.max_localization_submaps", localization.value("max_localization_submaps", 4));
  localization["prebuilt_submap_pin_precision"] =
    declare_and_get<double>(node, "localization.prebuilt_submap_pin_precision", localization.value("prebuilt_submap_pin_precision", 1e8));
  save_json(effective_path / "config_global_mapping.json", config_global_mapping);

  auto config_preprocess = load_json(effective_path / "config_preprocess.json");
  auto& preprocess = config_preprocess["preprocess"];
  preprocess["distance_near_thresh"] =
    declare_and_get<double>(node, "preprocess.distance_near_thresh", preprocess.value("distance_near_thresh", 1.0));
  preprocess["distance_far_thresh"] =
    declare_and_get<double>(node, "preprocess.distance_far_thresh", preprocess.value("distance_far_thresh", 50.0));
  preprocess["downsample_resolution"] =
    declare_and_get<double>(node, "preprocess.downsample_resolution", preprocess.value("downsample_resolution", 0.3));
  preprocess["random_downsample_target"] =
    declare_and_get<int>(node, "preprocess.random_downsample_target", preprocess.value("random_downsample_target", 10000));
  preprocess["use_random_grid_downsampling"] =
    declare_and_get<bool>(node, "preprocess.use_random_grid_downsampling", preprocess.value("use_random_grid_downsampling", false));
  preprocess["random_downsample_rate"] =
    declare_and_get<double>(node, "preprocess.random_downsample_rate", preprocess.value("random_downsample_rate", 0.1));
  preprocess["enable_outlier_removal"] =
    declare_and_get<bool>(node, "preprocess.enable_outlier_removal", preprocess.value("enable_outlier_removal", false));
  preprocess["outlier_removal_k"] =
    declare_and_get<int>(node, "preprocess.outlier_removal_k", preprocess.value("outlier_removal_k", 10));
  preprocess["outlier_std_mul_factor"] =
    declare_and_get<double>(node, "preprocess.outlier_std_mul_factor", preprocess.value("outlier_std_mul_factor", 1.0));
  preprocess["enable_cropbox_filter"] =
    declare_and_get<bool>(node, "preprocess.enable_cropbox_filter", preprocess.value("enable_cropbox_filter", false));
  preprocess["k_correspondences"] =
    declare_and_get<int>(node, "preprocess.k_correspondences", preprocess.value("k_correspondences", 10));
  preprocess["num_threads"] =
    declare_and_get<int>(node, "preprocess.num_threads", preprocess.value("num_threads", 4));
  save_json(effective_path / "config_preprocess.json", config_preprocess);

  auto config_sub_mapping = load_json(effective_path / "config_sub_mapping.json");
  auto& sub_mapping = config_sub_mapping["sub_mapping"];
  sub_mapping["so_name"] =
    declare_and_get<std::string>(node, "sub_mapping.so_name", sub_mapping.value("so_name", "libsub_mapping.so"));
  sub_mapping["enable_imu"] =
    declare_and_get<bool>(node, "sub_mapping.enable_imu", sub_mapping.value("enable_imu", true));
  sub_mapping["enable_optimization"] =
    declare_and_get<bool>(node, "sub_mapping.enable_optimization", sub_mapping.value("enable_optimization", false));
  sub_mapping["max_num_keyframes"] =
    declare_and_get<int>(node, "sub_mapping.max_num_keyframes", sub_mapping.value("max_num_keyframes", 15));
  sub_mapping["keyframe_update_strategy"] =
    declare_and_get<std::string>(node, "sub_mapping.keyframe_update_strategy", sub_mapping.value("keyframe_update_strategy", "OVERLAP"));
  sub_mapping["keyframe_update_min_points"] =
    declare_and_get<int>(node, "sub_mapping.keyframe_update_min_points", sub_mapping.value("keyframe_update_min_points", 500));
  sub_mapping["keyframe_update_interval_rot"] =
    declare_and_get<double>(node, "sub_mapping.keyframe_update_interval_rot", sub_mapping.value("keyframe_update_interval_rot", 0.5));
  sub_mapping["keyframe_update_interval_trans"] =
    declare_and_get<double>(node, "sub_mapping.keyframe_update_interval_trans", sub_mapping.value("keyframe_update_interval_trans", 0.2));
  sub_mapping["max_keyframe_overlap"] =
    declare_and_get<double>(node, "sub_mapping.max_keyframe_overlap", sub_mapping.value("max_keyframe_overlap", 0.9));
  sub_mapping["create_between_factors"] =
    declare_and_get<bool>(node, "sub_mapping.create_between_factors", sub_mapping.value("create_between_factors", true));
  sub_mapping["between_registration_type"] =
    declare_and_get<std::string>(node, "sub_mapping.between_registration_type", sub_mapping.value("between_registration_type", "GICP"));
  sub_mapping["registration_error_factor_type"] =
    declare_and_get<std::string>(node, "sub_mapping.registration_error_factor_type", sub_mapping.value("registration_error_factor_type", "VGICP_GPU"));
  sub_mapping["keyframe_randomsampling_rate"] =
    declare_and_get<double>(node, "sub_mapping.keyframe_randomsampling_rate", sub_mapping.value("keyframe_randomsampling_rate", 1.0));
  sub_mapping["keyframe_voxel_resolution"] =
    declare_and_get<double>(node, "sub_mapping.keyframe_voxel_resolution", sub_mapping.value("keyframe_voxel_resolution", 0.2));
  sub_mapping["keyframe_voxelmap_levels"] =
    declare_and_get<int>(node, "sub_mapping.keyframe_voxelmap_levels", sub_mapping.value("keyframe_voxelmap_levels", 2));
  sub_mapping["keyframe_voxelmap_scaling_factor"] =
    declare_and_get<double>(node, "sub_mapping.keyframe_voxelmap_scaling_factor", sub_mapping.value("keyframe_voxelmap_scaling_factor", 2.0));
  sub_mapping["submap_downsample_resolution"] =
    declare_and_get<double>(node, "sub_mapping.submap_downsample_resolution", sub_mapping.value("submap_downsample_resolution", 0.1));
  sub_mapping["submap_voxel_resolution"] =
    declare_and_get<double>(node, "sub_mapping.submap_voxel_resolution", sub_mapping.value("submap_voxel_resolution", 0.5));
  sub_mapping["submap_target_num_points"] =
    declare_and_get<int>(node, "sub_mapping.submap_target_num_points", sub_mapping.value("submap_target_num_points", 50000));
  save_json(effective_path / "config_sub_mapping.json", config_sub_mapping);

  // glim_relocalization's own config files - only present at all when
  // config_path points at a localization profile (copy_config_directory()
  // only copies whatever the profile directory actually contains). glim_ros
  // has no other business knowing this package's schema, but this is the
  // only point in the whole pipeline with both ROS node access and control
  // over the effective config directory before GlobalConfig locks in -
  // BBS3DExtension/LocalizationReloc read their config at construction time,
  // before an extension module ever gets a Node& (see create_subscriptions()).
  const auto config_bbs3d_path = effective_path / "config_bbs3d.json";
  if (boost::filesystem::exists(config_bbs3d_path)) {
    auto config_bbs3d = load_json(config_bbs3d_path);
    auto& bbs3d = config_bbs3d["bbs3d"];
    bbs3d["reference_map_path"] =
      declare_and_get<std::string>(node, "bbs3d.reference_map_path", bbs3d.value("reference_map_path", ""));
    bbs3d["voxelmap_cache_path"] =
      declare_and_get<std::string>(node, "bbs3d.voxelmap_cache_path", bbs3d.value("voxelmap_cache_path", ""));
    bbs3d["min_level_res"] =
      declare_and_get<double>(node, "bbs3d.min_level_res", bbs3d.value("min_level_res", 2.0));
    bbs3d["max_level"] =
      declare_and_get<int>(node, "bbs3d.max_level", bbs3d.value("max_level", 6));
    bbs3d["score_threshold_pct"] =
      declare_and_get<double>(node, "bbs3d.score_threshold_pct", bbs3d.value("score_threshold_pct", 0.0));
    bbs3d["lidar_topic"] =
      declare_and_get<std::string>(node, "bbs3d.lidar_topic", bbs3d.value("lidar_topic", "/lidar_points"));
    bbs3d["min_src_frames"] =
      declare_and_get<int>(node, "bbs3d.min_src_frames", bbs3d.value("min_src_frames", 5));
    bbs3d["max_src_frames"] =
      declare_and_get<int>(node, "bbs3d.max_src_frames", bbs3d.value("max_src_frames", 50));
    bbs3d["max_src_points"] =
      declare_and_get<int>(node, "bbs3d.max_src_points", bbs3d.value("max_src_points", 100000));
    bbs3d["max_tar_points"] =
      declare_and_get<int>(node, "bbs3d.max_tar_points", bbs3d.value("max_tar_points", 500000));
    bbs3d["timeout_ms"] =
      declare_and_get<int>(node, "bbs3d.timeout_ms", bbs3d.value("timeout_ms", 0));
    bbs3d["publish_best_effort"] =
      declare_and_get<bool>(node, "bbs3d.publish_best_effort", bbs3d.value("publish_best_effort", false));
    bbs3d["best_effort_min_pct"] =
      declare_and_get<double>(node, "bbs3d.best_effort_min_pct", bbs3d.value("best_effort_min_pct", 0.05));
    bbs3d["map_frame_id"] =
      declare_and_get<std::string>(node, "bbs3d.map_frame_id", bbs3d.value("map_frame_id", "ardia_map"));
    bbs3d["odom_frame_id"] =
      declare_and_get<std::string>(node, "bbs3d.odom_frame_id", bbs3d.value("odom_frame_id", "odom"));
    bbs3d["source_frame_id"] =
      declare_and_get<std::string>(node, "bbs3d.source_frame_id", bbs3d.value("source_frame_id", "ardia_map"));
    bbs3d["force_planar_pose"] =
      declare_and_get<bool>(node, "bbs3d.force_planar_pose", bbs3d.value("force_planar_pose", true));
    bbs3d["invert_yaw"] =
      declare_and_get<bool>(node, "bbs3d.invert_yaw", bbs3d.value("invert_yaw", false));
    bbs3d["roll_pitch_search_rad"] =
      declare_and_get<double>(node, "bbs3d.roll_pitch_search_rad", bbs3d.value("roll_pitch_search_rad", 0.0));
    save_json(config_bbs3d_path, config_bbs3d);
  }

  const auto config_global_mapping_reloc_path = effective_path / "config_global_mapping_reloc.json";
  if (boost::filesystem::exists(config_global_mapping_reloc_path)) {
    auto config_global_mapping_reloc = load_json(config_global_mapping_reloc_path);
    auto& global_mapping_reloc = config_global_mapping_reloc["global_mapping_reloc"];
    global_mapping_reloc["reference_map_path"] =
      declare_and_get<std::string>(node, "global_mapping_reloc.reference_map_path", global_mapping_reloc.value("reference_map_path", ""));
    save_json(config_global_mapping_reloc_path, config_global_mapping_reloc);
  }

  return effective_path.string();
}

// resple_bridge's SharedSplineState singleton (glim_ext_resple_bridge, a
// separate .so) tracks RESPLE's own knot indexing entirely in this
// process's memory. If glim_ros is a fresh process (its own
// already-activated-once guard forced a respawn) but resple has been
// running continuously since well before that -- its own knot/index
// counters already huge -- the bridge's fresh, zero-based tracking desyncs
// against them: updateKnots() ends up always appending instead of refining
// overlapping windows, and the spline's local-index <-> real-time
// correspondence drifts. Confirmed live: this silently produces noisy,
// biased odometry (10+ deg roll error, several deg of pitch/yaw jitter
// while genuinely stationary) rather than any visible error.
//
// Fix: whenever glim_ros configures, check how long resple's current
// process has actually been alive (via its one-shot, transient_local
// /start_time message) and, if it clearly didn't just boot alongside this
// same glim_ros process, force it through a full restart first -- so both
// sides' indexing resets to zero together. Runs on a throwaway node of its
// own: GlimROS's own SingleThreadedExecutor is busy inside on_configure()
// right now and can't also process the responses this needs.
void ensure_resple_fresh_before_configuring() {
  constexpr auto kStartTimeWaitTimeout = std::chrono::milliseconds(3000);
  constexpr auto kServiceTimeout = std::chrono::milliseconds(2000);
  // RESPLE's own on_deactivate() can block for up to 5s on a bounded-join
  // wait for its processing thread (RESPLE.cpp's join_processing_thread_with_
  // timeout()) before it can return success or failure -- a 2s client-side
  // timeout on THIS specific call would give up and read that as a failure
  // while RESPLE is still within its own legitimate wait, not actually stuck.
  // Confirmed live: this was silently short-circuiting the whole restart
  // sequence on every deactivate that took longer than 2s, even when RESPLE
  // was behaving normally. Give this one call more room than RESPLE's own
  // bound, so a real timeout here only fires past RESPLE's own guard.
  constexpr auto kDeactivateTimeout = std::chrono::milliseconds(7000);
  constexpr auto kRespawnPollTimeout = std::chrono::milliseconds(30000);
  constexpr auto kRespawnPollInterval = std::chrono::milliseconds(500);
  // How old resple's /start_time needs to be for us to conclude it's a
  // stale, long-running process rather than one that just finished
  // configuring+activating moments ago as part of this same cold start
  // (livox_driver -> resple -> glim_ros is the normal boot order, and
  // resple reaching ACTIVE typically takes well under this).
  constexpr int64_t kStaleThresholdNs = 8'000'000'000;  // 8s
  // kSyncedMarkerPath (file-scope, above): written by resple_start_time_
  // changed_callback() right before it forces our own respawn -- lets the
  // FRESH process that comes back know "the resple instance I'm about to
  // check was already confirmed fresh, that's exactly why I'm restarting"
  // instead of re-deriving the same conclusion from age alone. Without this,
  // a self-restart triggered by that monitor can still see resple as
  // "stale" here (glim_ros's own respawn+reconfigure cycle alone can take
  // longer than kStaleThresholdNs), causing one needless extra resple
  // restart right after the one that was already just confirmed to work.

  auto probe_node = std::make_shared<rclcpp::Node>("glim_ros_resple_freshness_probe");
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(probe_node);
  std::thread spin_thread([&exec]() { exec.spin(); });
  bool stopped = false;
  auto stop_spinning = [&]() {
    if (stopped) return;
    stopped = true;
    exec.cancel();
    if (spin_thread.joinable()) {
      spin_thread.join();
    }
  };

  std::mutex m;
  std::condition_variable cv;
  std::optional<int64_t> start_time_ns;
  auto sub = probe_node->create_subscription<std_msgs::msg::Int64>(
    "/start_time", rclcpp::QoS(1).transient_local(),
    [&](const std_msgs::msg::Int64::SharedPtr msg) {
      {
        std::lock_guard<std::mutex> lk(m);
        start_time_ns = msg->data;
      }
      cv.notify_one();
    });

  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait_for(lk, kStartTimeWaitTimeout, [&] { return start_time_ns.has_value(); });
  }

  if (!start_time_ns.has_value()) {
    spdlog::info("[glim_ros] resple /start_time not seen yet -- nothing stale to restart");
    stop_spinning();
    return;
  }

  {
    std::ifstream marker_in(kSyncedMarkerPath);
    int64_t marked_value = 0;
    if (marker_in && (marker_in >> marked_value) && marked_value == *start_time_ns) {
      spdlog::info(
        "[glim_ros] resple's /start_time matches the value that triggered our "
        "own last self-restart -- already confirmed fresh, skipping a redundant "
        "restart");
      stop_spinning();
      return;
    }
  }

  const int64_t now_ns = probe_node->now().nanoseconds();
  const int64_t age_ns = now_ns - *start_time_ns;
  if (age_ns < kStaleThresholdNs) {
    spdlog::info("[glim_ros] resple started {:.1f}s ago -- fresh, booted alongside this process", age_ns / 1e9);
    stop_spinning();
    return;
  }

  spdlog::warn(
    "[glim_ros] resple has been running for {:.1f}s already while this glim_ros "
    "process just started fresh -- its spline/knot indexing would desync "
    "against a fresh consumer, so restarting resple before proceeding",
    age_ns / 1e9);

  using lifecycle_msgs::msg::State;
  using lifecycle_msgs::msg::Transition;
  using ChangeState = lifecycle_msgs::srv::ChangeState;
  using GetState = lifecycle_msgs::srv::GetState;

  auto change_state_client = probe_node->create_client<ChangeState>("/resple/change_state");
  auto get_state_client = probe_node->create_client<GetState>("/resple/get_state");

  auto get_state = [&]() -> std::optional<uint8_t> {
    if (!get_state_client->wait_for_service(kServiceTimeout)) {
      return std::nullopt;
    }
    auto req = std::make_shared<GetState::Request>();
    auto fut = get_state_client->async_send_request(req);
    if (fut.wait_for(kServiceTimeout) != std::future_status::ready) {
      return std::nullopt;
    }
    return fut.get()->current_state.id;
  };

  auto send_transition = [&](uint8_t transition_id, std::chrono::milliseconds timeout) -> bool {
    if (!change_state_client->wait_for_service(timeout)) {
      return false;
    }
    auto req = std::make_shared<ChangeState::Request>();
    req->transition.id = transition_id;
    auto fut = change_state_client->async_send_request(req);
    if (fut.wait_for(timeout) != std::future_status::ready) {
      return false;
    }
    return fut.get()->success;
  };

  const auto current_state = get_state();
  if (!current_state.has_value() || *current_state != State::PRIMARY_STATE_ACTIVE) {
    spdlog::info("[glim_ros] resple is not currently active -- nothing to restart");
    stop_spinning();
    return;
  }

  if (!send_transition(Transition::TRANSITION_DEACTIVATE, kDeactivateTimeout)) {
    // Not fatal: this is exactly what happens when resple's own processData()
    // hang guard fires during this same deactivate call. That guard now
    // force-exits the process (_exit()) once its own 5s bounded-join gives
    // up, and a process that has already exited can't send a service
    // response to the request that triggered it -- so "no response" here is
    // indistinguishable from, and just as fine as, a normal deactivate. The
    // respawn-poll below is what actually confirms whether a fresh process
    // shows up; treat this the same way the activate attempt below is
    // already treated, not as a reason to give up.
    spdlog::warn(
      "[glim_ros] resple did not confirm deactivate -- possibly because its "
      "own processData()-hang guard already fired and force-exited the "
      "process mid-request; proceeding to check for a fresh process either way");
  }

  // Requesting activate again on what is still the SAME resple process is
  // expected to fail or time out: resple has its own already-activated-once
  // guard (ikd-tree double-Build() protection) that, on exactly this
  // request, exits the process so respawn:true gives a fresh one. That is
  // the intended outcome here, not an error.
  if (!send_transition(Transition::TRANSITION_ACTIVATE, kServiceTimeout)) {
    spdlog::info(
      "[glim_ros] resple did not confirm activate after deactivate -- "
      "expected if this tripped its own restart-on-reactivate guard; "
      "waiting for a fresh process instead");
  }

  const auto deadline = std::chrono::steady_clock::now() + kRespawnPollTimeout;
  bool respawned = false;
  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(kRespawnPollInterval);
    const auto state = get_state();
    if (state.has_value() && *state == State::PRIMARY_STATE_UNCONFIGURED) {
      respawned = true;
      break;
    }
  }

  if (!respawned) {
    spdlog::error(
      "[glim_ros] resple did not respawn as a fresh process within {}ms -- "
      "proceeding anyway, odometry may be corrupted",
      kRespawnPollTimeout.count());
    stop_spinning();
    return;
  }

  if (!send_transition(Transition::TRANSITION_CONFIGURE, kServiceTimeout) ||
      !send_transition(Transition::TRANSITION_ACTIVATE, kServiceTimeout)) {
    spdlog::error(
      "[glim_ros] failed to bring the fresh resple process back to active "
      "-- proceeding anyway, odometry may be corrupted");
    stop_spinning();
    return;
  }

  spdlog::info("[glim_ros] resple restarted fresh -- proceeding with our own configure");
  stop_spinning();
}

}  // namespace

GlimROS::GlimROS(const rclcpp::NodeOptions& options) : rclcpp_lifecycle::LifecycleNode("glim_ros", options) {
  bond_topic_name_ = this->declare_parameter<std::string>("bond_topic_name", "/bond");
  bond_id_ = this->declare_parameter<std::string>("bond_id", "glim_ros");
  enable_bond_ = this->declare_parameter<bool>("enable_bond", true);
  bond_heartbeat_period_s_ = this->declare_parameter<double>("bond_heartbeat_period_s", 0.1);
  bond_heartbeat_timeout_s_ = this->declare_parameter<double>("bond_heartbeat_timeout_s", 1.0);
}

GlimROS::CallbackReturn GlimROS::on_configure(const rclcpp_lifecycle::State&) {
  // Setup logger
  auto logger = spdlog::stdout_color_mt("glim");
  logger->sinks().push_back(get_ringbuffer_sink());
  spdlog::set_default_logger(logger);

  // Must run before anything below subscribes to resple's stream: see the
  // function's own comment for why a stale, still-running resple process
  // needs a coupled restart whenever THIS glim_ros process is fresh.
  ensure_resple_fresh_before_configuring();

  bool debug = false;
  this->declare_parameter<bool>("debug", false);
  this->get_parameter<bool>("debug", debug);

  if (debug) {
    spdlog::info("enable debug printing");
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/glim_log.log", true);
    logger->sinks().push_back(file_sink);
    logger->set_level(spdlog::level::trace);

    print_system_info(logger);
  }

  dump_on_unload = false;
  saved = false;
  dump_path_ = "/tmp/dump";
  this->declare_parameter<bool>("dump_on_unload", false);
  this->get_parameter<bool>("dump_on_unload", dump_on_unload);
  this->declare_parameter<std::string>("dump_path", dump_path_);
  this->get_parameter<std::string>("dump_path", dump_path_);

  if (dump_on_unload) {
    spdlog::info("dump_on_unload={} dump_path={}", dump_on_unload, dump_path_);
  }

  std::string config_path;
  this->declare_parameter<std::string>("config_path", "config");
  this->get_parameter<std::string>("config_path", config_path);

  if (config_path[0] != '/') {
    // config_path is relative to the glim directory
    config_path = ament_index_cpp::get_package_share_directory("glim") + "/" + config_path;
  }

  bool use_ros_parameter_overrides = true;
  this->declare_parameter<bool>("use_ros_parameter_overrides", true);
  this->get_parameter<bool>("use_ros_parameter_overrides", use_ros_parameter_overrides);
  if (use_ros_parameter_overrides) {
    config_path = create_effective_config_from_ros_params(*this, config_path);
  }

  logger->info("config_path: {}", config_path);
  glim::GlobalConfig::instance(config_path, true);
  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  keep_raw_points = config_ros.param<bool>("glim_ros", "keep_raw_points", false);
  imu_time_offset = config_ros.param<double>("glim_ros", "imu_time_offset", 0.0);
  points_time_offset = config_ros.param<double>("glim_ros", "points_time_offset", 0.0);
  acc_scale = config_ros.param<double>("glim_ros", "acc_scale", 0.0);

  glim::Config config_sensors(glim::GlobalConfig::get_config_path("config_sensors"));
  intensity_field = config_sensors.param<std::string>("sensors", "intensity_field", "intensity");
  ring_field = config_sensors.param<std::string>("sensors", "ring_field", "");

  // Setup GPU-based linearization
#ifdef BUILD_GTSAM_POINTS_GPU
  gtsam_points::LinearizationHook::register_hook([]() { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

  // Preprocessing
  time_keeper.reset(new glim::TimeKeeper);
  preprocessor.reset(new glim::CloudPreprocessor);

  // Odometry estimation
  glim::Config config_odometry(glim::GlobalConfig::get_config_path("config_odometry"));
  const std::string odometry_estimation_so_name = config_odometry.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");
  spdlog::info("load {}", odometry_estimation_so_name);

  std::shared_ptr<glim::OdometryEstimationBase> odom = OdometryEstimationBase::load_module(odometry_estimation_so_name);
  if (!odom) {
    spdlog::critical("failed to load odometry estimation module");
    abort();
  }
  odometry_estimation.reset(new glim::AsyncOdometryEstimation(odom, odom->requires_imu()));

  // Sub mapping
  if (config_ros.param<bool>("glim_ros", "enable_local_mapping", true)) {
    const std::string sub_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_sub_mapping")).param<std::string>("sub_mapping", "so_name", "libsub_mapping.so");
    if (!sub_mapping_so_name.empty()) {
      spdlog::info("load {}", sub_mapping_so_name);
      auto sub = SubMappingBase::load_module(sub_mapping_so_name);
      if (sub) {
        sub_mapping.reset(new AsyncSubMapping(sub));
      }
    }
  }

  // Global mapping
  if (config_ros.param<bool>("glim_ros", "enable_global_mapping", true)) {
    const std::string global_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_global_mapping")).param<std::string>("global_mapping", "so_name", "libglobal_mapping.so");
    is_localization_mode_ = global_mapping_so_name.find("global_mapping_reloc") != std::string::npos;
    if (!global_mapping_so_name.empty()) {
      spdlog::info("load {}", global_mapping_so_name);
      auto global = GlobalMappingBase::load_module(global_mapping_so_name);
      if (global) {
        global_mapping.reset(new AsyncGlobalMapping(global));
      }
    }
  }

  // Extention modules
  const auto extensions = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules");
  if (extensions && !extensions->empty()) {
    for (const auto& extension : *extensions) {
      if (extension.find("viewer") == std::string::npos && extension.find("monitor") == std::string::npos) {
        spdlog::warn("Extension modules are enabled!!");
        spdlog::warn("You must carefully check and follow the licenses of ext modules");

        try {
          const std::string config_ext_path = ament_index_cpp::get_package_share_directory("glim_ext") + "/config";
          spdlog::info("config_ext_path: {}", config_ext_path);
          glim::GlobalConfig::instance()->override_param<std::string>("global", "config_ext", config_ext_path);
        } catch (ament_index_cpp::PackageNotFoundError& e) {
          spdlog::warn("glim_ext package path was not found!!");
        }

        break;
      }
    }

    for (const auto& extension : *extensions) {
      spdlog::info("load {}", extension);
      auto ext_module = ExtensionModule::load_module(extension);
      if (ext_module == nullptr) {
        spdlog::error("failed to load {}", extension);
        continue;
      } else {
        extension_modules.push_back(ext_module);

        auto ext_module_ros = std::dynamic_pointer_cast<ExtensionModuleROS2>(ext_module);
        if (ext_module_ros) {
          const auto subs = ext_module_ros->create_subscriptions(*this);
          extension_subs.insert(extension_subs.end(), subs.begin(), subs.end());
        }
      }
    }
  }

  // ROS-related
  using std::placeholders::_1;
  const std::string imu_topic = config_ros.param<std::string>("glim_ros", "imu_topic", "");
  const std::string points_topic = config_ros.param<std::string>("glim_ros", "points_topic", "");
  const std::string image_topic = config_ros.param<std::string>("glim_ros", "image_topic", "");

  // Subscribers
  rclcpp::SensorDataQoS default_imu_qos;
  default_imu_qos.get_rmw_qos_profile().depth = 1000;
  auto qos = get_qos_settings(config_ros, "glim_ros", "imu_qos", default_imu_qos);
  imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, qos, std::bind(&GlimROS::imu_callback, this, _1));

  qos = get_qos_settings(config_ros, "glim_ros", "points_qos");
  points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(points_topic, qos, std::bind(&GlimROS::points_callback, this, _1));
#ifdef BUILD_WITH_CV_BRIDGE
  qos = get_qos_settings(config_ros, "glim_ros", "image_qos");
  image_sub = image_transport::create_subscription(this, image_topic, std::bind(&GlimROS::image_callback, this, _1), "raw", qos.get_rmw_qos_profile());
#endif

  for (const auto& sub : this->extension_subscriptions()) {
    spdlog::debug("subscribe to {}", sub->topic);
    sub->create_subscriber(*this);
  }

  spdlog::debug("initialized");
  return CallbackReturn::SUCCESS;
}

GlimROS::CallbackReturn GlimROS::on_activate(const rclcpp_lifecycle::State&) {
  // See the note on activated_once_ in glim_ros.hpp -- a reactivation on the
  // same still-running process isn't a proven-safe path for GLIM's
  // background estimation/mapping threads, so refuse it and force a full
  // process restart via respawn:true instead of risking it, same as RESPLE.
  if (activated_once_) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Refusing to activate: this process has already activated glim_ros once. "
      "Reactivating in place is not a proven-safe path for GLIM's background "
      "estimation/mapping threads. Exiting this process so respawn:true gives "
      "a fresh one instead of leaving this node stuck inactive.");
    rclcpp::shutdown();
    // rclcpp::shutdown() alone does NOT terminate the process: main()'s
    // glim->wait() unconditionally calls odometry_estimation->join(), and
    // that background thread has no way to know it should stop early (it's
    // built to run to natural completion, e.g. end of a bag), so it just
    // keeps consuming LiDAR data and computing forever with a dead ROS
    // context -- permanently unable to publish anything again, but never
    // actually exiting for respawn:true to kick in. Force a real, immediate
    // process exit instead of relying on main()'s normal unwind.
    _exit(1);
  }
  activated_once_ = true;

  for (const auto& ext_module : extension_modules) {
    auto ext_module_ros = std::dynamic_pointer_cast<ExtensionModuleROS2>(ext_module);
    if (ext_module_ros) {
      ext_module_ros->on_activate();
    }
  }

  // See the long comment on resple_start_time_watch_sub_ in glim_ros.hpp:
  // ensure_resple_fresh_before_configuring() only protects against resple
  // ALREADY being stale at our own configure time -- this catches resple
  // restarting independently anytime after that, for the rest of this
  // process's life.
  resple_start_time_baseline_.reset();
  resple_start_time_watch_sub_ = this->create_subscription<std_msgs::msg::Int64>(
    "/start_time", rclcpp::QoS(1).transient_local(),
    [this](const std_msgs::msg::Int64::SharedPtr msg) { resple_start_time_changed_callback(msg); });

  timer = this->create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });

  start_bond();
  return CallbackReturn::SUCCESS;
}

void GlimROS::resple_start_time_changed_callback(const std_msgs::msg::Int64::SharedPtr msg) {
  if (!resple_start_time_baseline_.has_value()) {
    // First delivery: transient_local means this could be replaying the
    // value from whichever resple process ensure_resple_fresh_before_
    // configuring() already confirmed fresh moments ago. Just record it as
    // the baseline for this activation -- nothing to react to yet.
    resple_start_time_baseline_ = msg->data;
    return;
  }

  if (msg->data == *resple_start_time_baseline_) {
    return;
  }

  RCLCPP_ERROR(
    this->get_logger(),
    "resple's /start_time changed while this glim_ros process is still "
    "active (was %ld, now %ld) -- resple restarted on its own and our "
    "spline bridge has no way to resync against a running process. Exiting "
    "this process so respawn:true gives a fresh one that will re-sync "
    "against it, instead of silently freezing on stale tracking.",
    static_cast<long>(*resple_start_time_baseline_), static_cast<long>(msg->data));

  // Tell the fresh process that's about to come back "this exact resple
  // instance is the one that made me restart -- you don't need to restart
  // it again". See kSyncedMarkerPath's comment for why this exists at all.
  // Best-effort: if this write fails, ensure_resple_fresh_before_
  // configuring() just falls back to its normal age-based check, which is
  // slower (one redundant restart) but not wrong.
  {
    std::ofstream marker_out(kSyncedMarkerPath, std::ios::trunc);
    marker_out << msg->data;
  }

  rclcpp::shutdown();
  _exit(1);
}

GlimROS::CallbackReturn GlimROS::on_deactivate(const rclcpp_lifecycle::State&) {
  stop_bond();
  timer.reset();
  resple_start_time_watch_sub_.reset();

  for (const auto& ext_module : extension_modules) {
    auto ext_module_ros = std::dynamic_pointer_cast<ExtensionModuleROS2>(ext_module);
    if (ext_module_ros) {
      ext_module_ros->on_deactivate();
    }
  }

  return CallbackReturn::SUCCESS;
}

GlimROS::CallbackReturn GlimROS::on_cleanup(const rclcpp_lifecycle::State&) {
  imu_sub.reset();
  points_sub.reset();
#ifdef BUILD_WITH_CV_BRIDGE
  image_sub = image_transport::Subscriber();
#endif
  return CallbackReturn::SUCCESS;
}

GlimROS::CallbackReturn GlimROS::on_shutdown(const rclcpp_lifecycle::State&) {
  stop_bond();
  timer.reset();
  return CallbackReturn::SUCCESS;
}

void GlimROS::start_bond() {
  if (!enable_bond_) {
    return;
  }

  lifecycle_bond_ = std::make_unique<bond::Bond>(
    bond_topic_name_, bond_id_,
    get_node_base_interface(),
    get_node_logging_interface(),
    get_node_parameters_interface(),
    get_node_timers_interface(),
    get_node_topics_interface(),
    std::bind(&GlimROS::bond_broken_callback, this),
    std::bind(&GlimROS::bond_formed_callback, this));
  lifecycle_bond_->setHeartbeatPeriod(bond_heartbeat_period_s_);
  lifecycle_bond_->setHeartbeatTimeout(bond_heartbeat_timeout_s_);
  lifecycle_bond_->start();
}

void GlimROS::stop_bond() {
  if (lifecycle_bond_) {
    lifecycle_bond_->breakBond();
    lifecycle_bond_.reset();
  }
  bond_formed_ = false;
  bond_broken_ = false;
}

void GlimROS::bond_formed_callback() {
  bond_formed_ = true;
  bond_broken_ = false;
  RCLCPP_INFO(this->get_logger(), "Lifecycle bond formed on %s", bond_topic_name_.c_str());
}

void GlimROS::bond_broken_callback() {
  bond_formed_ = false;
  bond_broken_ = true;
  RCLCPP_WARN(this->get_logger(), "Lifecycle bond broken on %s", bond_topic_name_.c_str());
}

GlimROS::~GlimROS() {
  spdlog::debug("quit");
  extension_modules.clear();

  if (dump_on_unload && !saved) {
    wait(true);
    save(dump_path_);
  }
}

const std::vector<std::shared_ptr<GenericTopicSubscription>>& GlimROS::extension_subscriptions() {
  return extension_subs;
}

std::string GlimROS::dump_path() const {
  return dump_path_;
}

void GlimROS::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  spdlog::trace("IMU: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "imu_frame_id")) {
    spdlog::debug("auto-detecting IMU frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "imu_frame_id", msg->header.frame_id);
  }

  if (std::abs(acc_scale) < 1e-6) {
    const double norm = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z).norm();
    if (norm > 7.0 && norm < 12.0) {
      acc_scale = 1.0;
      spdlog::debug("assuming [m/s^2] for acceleration unit (acc_scale={}, norm={})", acc_scale, norm);
    } else if (norm > 0.8 && norm < 1.2) {
      acc_scale = 9.80665;
      spdlog::debug("assuming [g] for acceleration unit (acc_scale={}, norm={})", acc_scale, norm);
    } else {
      acc_scale = 1.0;
      spdlog::warn("unexpected acceleration norm {}. assuming [m/s^2] for acceleration unit (acc_scale={})", norm, acc_scale);
    }
  }

  const double imu_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9 + imu_time_offset;
  const Eigen::Vector3d linear_acc = acc_scale * Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d angular_vel(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (!time_keeper->validate_imu_stamp(imu_stamp)) {
    spdlog::warn("skip an invalid IMU data (stamp={})", imu_stamp);
    return;
  }

  odometry_estimation->insert_imu(imu_stamp, linear_acc, angular_vel);
  if (sub_mapping) {
    sub_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  }
  if (global_mapping) {
    global_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  }
}

#ifdef BUILD_WITH_CV_BRIDGE
void GlimROS::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  spdlog::trace("image: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "image_frame")) {
    spdlog::debug("auto-detecting image frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "image_frame", msg->header.frame_id);
  }

  auto cv_image = cv_bridge::toCvCopy(msg, "bgr8");

  const double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
  odometry_estimation->insert_image(stamp, cv_image->image);
  if (sub_mapping) {
    sub_mapping->insert_image(stamp, cv_image->image);
  }
  if (global_mapping) {
    global_mapping->insert_image(stamp, cv_image->image);
  }
}
#endif

size_t GlimROS::points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  // TEMPORARY DEBUG: proves whether this callback is still being invoked by
  // the executor at all during a freeze (vs. being invoked but stuck/rejecting
  // downstream, which would show up as the warnings a few lines below
  // instead). info level, not trace, so it's visible without debug:true.
  static uint64_t points_callback_hit_count = 0;
  if (++points_callback_hit_count % 20 == 1) {
    spdlog::info("[DEBUG] points_callback() alive, hit #{} stamp={}.{}", points_callback_hit_count, msg->header.stamp.sec, msg->header.stamp.nanosec);
  }

  spdlog::trace("points: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "lidar_frame_id")) {
    spdlog::debug("auto-detecting LiDAR frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "lidar_frame_id", msg->header.frame_id);
  }

  auto raw_points = glim::extract_raw_points(*msg, intensity_field, ring_field);
  if (raw_points == nullptr) {
    spdlog::warn("failed to extract points from message");
    return 0;
  }

  raw_points->stamp += points_time_offset;
  if (!time_keeper->process(raw_points)) {
    spdlog::warn("skip an invalid point cloud (stamp={})", raw_points->stamp);
    return 0;
  }
  auto preprocessed = preprocessor->preprocess(raw_points);

  if (keep_raw_points) {
    // note: Raw points are used only in extension modules for visualization purposes.
    //       If you need to reduce the memory footprint, you can safely comment out the following line.
    preprocessed->raw_points = raw_points;
  }

  odometry_estimation->insert_frame(preprocessed);

  const size_t workload = odometry_estimation->workload();
  spdlog::debug("workload={}", workload);

  return workload;
}

bool GlimROS::needs_wait() {
  for (const auto& ext_module : extension_modules) {
    if (ext_module->needs_wait()) {
      return true;
    }
  }

  return false;
}

void GlimROS::timer_callback() {
  for (const auto& ext_module : extension_modules) {
    if (!ext_module->ok()) {
      rclcpp::shutdown();
    }
  }

  std::vector<glim::EstimationFrame::ConstPtr> estimation_frames;
  std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
  odometry_estimation->get_results(estimation_frames, marginalized_frames);

  if (sub_mapping) {
    for (const auto& frame : marginalized_frames) {
      sub_mapping->insert_frame(frame);
    }

    auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }
    }
  }
}

void GlimROS::wait(bool auto_quit) {
  spdlog::info("waiting for odometry estimation");
  odometry_estimation->join();

  if (sub_mapping) {
    std::vector<glim::EstimationFrame::ConstPtr> estimation_results;
    std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_results, marginalized_frames);
    for (const auto& marginalized_frame : marginalized_frames) {
      sub_mapping->insert_frame(marginalized_frame);
    }

    spdlog::info("waiting for local mapping");
    sub_mapping->join();

    const auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }
      spdlog::info("waiting for global mapping");
      global_mapping->join();
    }
  }

  if (!auto_quit) {
    bool terminate = false;
    while (!terminate && rclcpp::ok()) {
      for (const auto& ext_module : extension_modules) {
        terminate |= (!ext_module->ok());
      }
    }
  }
}

void GlimROS::save(const std::string& path) {
  if (is_localization_mode_) {
    // Localization mode loads its reference map FROM this same dump_path at
    // startup (LocalizationReloc::ensure_reference_loaded()). Writing a live
    // session's own global-mapping state (pinned reference submaps plus
    // whatever ephemeral/synthetic submaps that session added) back to that
    // same path would silently overwrite the reference map on every shutdown
    // -- including the guard-forced restarts in glim_ros.cpp's on_activate(),
    // which happen far more often than an intentional map re-save. There is
    // nothing worth persisting from a localization run, so this is a no-op.
    saved = true;
    return;
  }

  if (global_mapping) global_mapping->save(path);
  for (auto& module : extension_modules) {
    module->at_exit(path);
  }
  saved = true;
}

}  // namespace glim

RCLCPP_COMPONENTS_REGISTER_NODE(glim::GlimROS);
