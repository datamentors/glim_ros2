#include <glim_ros/glim_ros.hpp>

#define GLIM_ROS2

#include <deque>
#include <fstream>
#include <iomanip>
#include <thread>
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

template <typename T>
T declare_and_get(rclcpp::Node& node, const std::string& name, const T& default_value) {
  if (!node.has_parameter(name)) {
    node.declare_parameter<T>(name, default_value);
  }

  T value = default_value;
  node.get_parameter(name, value);
  return value;
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
  rclcpp::Node& node,
  const std::string& base_config_path)
{
  const auto base_path = boost::filesystem::path(base_config_path);
  const auto effective_path =
    boost::filesystem::temp_directory_path() /
    ("glim_ros_" + std::string(node.get_name()) + "_" + std::to_string(getpid()));
  copy_config_directory(base_path, effective_path);

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
  glim_ros["extension_modules"] =
    declare_and_get<std::vector<std::string>>(
      node, "glim_ros.extension_modules",
      glim_ros.value("extension_modules", std::vector<std::string>{"librviz_viewer.so"}));
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
    declare_and_get<int>(node, "odometry_estimation.coverage_timeout_ms", odometry_estimation.value("coverage_timeout_ms", 150));
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

  // Fields NOT listed here (so_name, enable_imu, registration_error_factor_type,
  // isam2_relinearize_skip, gpu_memory_offload_mb) genuinely differ between G1's
  // mapping and localization profiles -- overridable for completeness, but no
  // robots/*/pulse.yaml should set a single static value for them, since one
  // profile's value would silently leak into the other whenever config_path
  // switches (they share one glim_ros node parameter block).
  auto config_global_mapping = load_json(effective_path / "config_global_mapping.json");
  auto& global_mapping = config_global_mapping["global_mapping"];
  global_mapping["so_name"] =
    declare_and_get<std::string>(node, "global_mapping.so_name", global_mapping.value("so_name", "libglobal_mapping.so"));
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
    declare_and_get<std::string>(node, "global_mapping.registration_error_factor_type", global_mapping.value("registration_error_factor_type", "VGICP"));
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
    declare_and_get<int>(node, "global_mapping.isam2_relinearize_skip", global_mapping.value("isam2_relinearize_skip", 1));
  global_mapping["isam2_relinearize_thresh"] =
    declare_and_get<double>(node, "global_mapping.isam2_relinearize_thresh", global_mapping.value("isam2_relinearize_thresh", 0.1));
  global_mapping["gpu_memory_offload_mb"] =
    declare_and_get<int>(node, "global_mapping.gpu_memory_offload_mb", global_mapping.value("gpu_memory_offload_mb", 0));
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

  // Fields NOT listed here (distance_far_thresh, downsample_resolution,
  // random_downsample_target) genuinely differ between G1's mapping and
  // localization profiles - see the global_mapping note above.
  auto config_preprocess = load_json(effective_path / "config_preprocess.json");
  auto& preprocess = config_preprocess["preprocess"];
  preprocess["distance_near_thresh"] =
    declare_and_get<double>(node, "preprocess.distance_near_thresh", preprocess.value("distance_near_thresh", 1.0));
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

  // registration_error_factor_type genuinely differs between G1's mapping and
  // localization profiles - see the global_mapping note above.
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
    declare_and_get<std::string>(node, "sub_mapping.registration_error_factor_type", sub_mapping.value("registration_error_factor_type", "VGICP"));
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

}  // namespace

GlimROS::GlimROS(const rclcpp::NodeOptions& options) : Node("glim_ros", options) {
  // Setup logger
  auto logger = spdlog::stdout_color_mt("glim");
  logger->sinks().push_back(get_ringbuffer_sink());
  spdlog::set_default_logger(logger);

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

  // Start timer
  timer = this->create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });

  spdlog::debug("initialized");
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
  if (global_mapping) global_mapping->save(path);
  for (auto& module : extension_modules) {
    module->at_exit(path);
  }
  saved = true;
}

}  // namespace glim

RCLCPP_COMPONENTS_REGISTER_NODE(glim::GlimROS);
