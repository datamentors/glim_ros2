#pragma once

#include <any>
#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <bondcpp/bond.hpp>

#include <std_msgs/msg/int64.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#ifdef BUILD_WITH_CV_BRIDGE
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#endif

namespace glim {
class TimeKeeper;
class CloudPreprocessor;
class AsyncOdometryEstimation;
class AsyncSubMapping;
class AsyncGlobalMapping;

class ExtensionModule;
class GenericTopicSubscription;

class GlimROS : public rclcpp_lifecycle::LifecycleNode {
public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  explicit GlimROS(const rclcpp::NodeOptions& options);
  ~GlimROS();

  CallbackReturn on_configure(const rclcpp_lifecycle::State&) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State&) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State&) override;

  bool needs_wait();
  void timer_callback();

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
#ifdef BUILD_WITH_CV_BRIDGE
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
#endif
  size_t points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  void wait(bool auto_quit = false);
  void save(const std::string& path);
  std::string dump_path() const;

  const std::vector<std::shared_ptr<GenericTopicSubscription>>& extension_subscriptions();

private:
  std::unique_ptr<glim::TimeKeeper> time_keeper;
  std::unique_ptr<glim::CloudPreprocessor> preprocessor;

  std::shared_ptr<glim::AsyncOdometryEstimation> odometry_estimation;
  std::unique_ptr<glim::AsyncSubMapping> sub_mapping;
  std::unique_ptr<glim::AsyncGlobalMapping> global_mapping;

  bool keep_raw_points;
  double imu_time_offset;
  double points_time_offset;
  double acc_scale;
  bool dump_on_unload;
  bool saved;
  std::string dump_path_;
  // True when global_mapping's so_name is the relocalization/localization
  // module (libglobal_mapping_reloc.so), not plain mapping. save() below is a
  // no-op in that case -- see the comment there for why.
  bool is_localization_mode_ = false;

  std::string intensity_field, ring_field;

  // Extension modulles
  std::vector<std::shared_ptr<ExtensionModule>> extension_modules;
  std::vector<std::shared_ptr<GenericTopicSubscription>> extension_subs;

  // ROS-related
  rclcpp::TimerBase::SharedPtr timer;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
#ifdef BUILD_WITH_CV_BRIDGE
  image_transport::Subscriber image_sub;
#endif

  // Refuses a second on_activate() in the same process -- mirrors RESPLE's
  // ikd-Tree guard. GLIM's AsyncOdometryEstimation/AsyncSubMapping/
  // AsyncGlobalMapping each own background worker threads constructed once in
  // on_configure(); reactivating in place (bond broke, process survives)
  // without a proven-safe teardown/rebuild path for those threads is the same
  // category of risk, so a reactivation attempt forces a full process
  // restart instead, via respawn:true, same as RESPLE.
  bool activated_once_ = false;

  // Watches resple's own /start_time for a value change while THIS process
  // is already active -- ensure_resple_fresh_before_configuring() (see
  // glim_ros.cpp) only ever checks/restarts resple once, at our own
  // on_configure() time. If resple restarts on its own AFTER that (its own
  // processData()-hang guard firing independently, or an operator cycling it
  // directly) while this glim_ros process keeps running unchanged, the
  // resple_bridge extension's tracked spline never finds out and gets stuck
  // permanently dropping frames (confirmed live: a "spline gap" warning that
  // never recovers). resple's /start_time is transient_local and published
  // exactly once per resple process lifetime, so a NEW value arriving here
  // means a NEW resple process exists -- force our own respawn (mirroring
  // activated_once_'s guard) so the fresh glim_ros process that comes back
  // re-syncs against it, instead of silently running on a stale bridge.
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr resple_start_time_watch_sub_;
  std::optional<int64_t> resple_start_time_baseline_;
  void resple_start_time_changed_callback(const std_msgs::msg::Int64::SharedPtr msg);

  std::string bond_topic_name_;
  std::string bond_id_;
  bool enable_bond_ = true;
  double bond_heartbeat_period_s_ = 0.1;
  double bond_heartbeat_timeout_s_ = 1.0;
  std::atomic_bool bond_formed_{false};
  std::atomic_bool bond_broken_{false};
  std::unique_ptr<bond::Bond> lifecycle_bond_;

  void start_bond();
  void stop_bond();
  void bond_formed_callback();
  void bond_broken_callback();
};

}  // namespace glim
