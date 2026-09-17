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
