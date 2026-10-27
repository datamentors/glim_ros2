#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

#include <gtest/gtest.h>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/util/config.hpp>
#include <glim_ros/rviz_viewer.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace {
using namespace std::chrono_literals;

class RvizViewerTest : public ::testing::Test {
protected:
  void SetUp() override {
    rclcpp::init(0, nullptr);
    directory = std::filesystem::temp_directory_path() / ("glim_viewer_test_" + std::to_string(getpid()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    std::ofstream(directory / "config.json") << R"({"global": {
      "config_ros": "config_ros.json", "config_logging": "config_logging.json"
    }})";
    std::ofstream(directory / "config_logging.json") << R"({"logging": {
      "log_dir": "/tmp", "save_logs": false
    }})";
    std::ofstream(directory / "config_ros.json") << R"({"glim_ros": {
      "imu_frame_id": "imu", "lidar_frame_id": "lidar", "base_frame_id": "imu",
      "odom_frame_id": "odom", "map_frame_id": "map", "publish_imu2lidar": false,
      "tf_time_offset": 1e-6
    }})";
    glim::GlobalConfig::instance(directory.string(), true);
    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("viewer_test");
    observer = std::make_shared<rclcpp::Node>("viewer_observer");
    executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor->add_node(observer);
    viewer = std::make_unique<glim::RvizViewer>();
    viewer->create_subscriptions(*node);
    viewer->on_activate();
  }

  void TearDown() override {
    if (viewer) {
      viewer->on_deactivate();
      viewer.reset();
    }
    if (observer) {
      executor->remove_node(observer);
      observer.reset();
    }
    executor.reset();
    node.reset();
    rclcpp::shutdown();
    std::filesystem::remove_all(directory);
  }

  template <typename Predicate>
  bool wait_for(Predicate ready, std::chrono::seconds timeout = 15s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
      executor->spin_some();
      if (ready()) return true;
      std::this_thread::sleep_for(10ms);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

  glim::SubMap::Ptr submap(int id, double translation) {
    auto map = std::make_shared<glim::SubMap>();
    map->id = id;
    map->T_world_origin = Eigen::Isometry3d::Identity();
    map->T_world_origin.translation().x() = translation;
    map->T_origin_endpoint_L = map->T_origin_endpoint_R = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Vector3d> points{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
    for (auto& point : points) point.x() += 0.25 * id;
    map->frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
    auto frame = std::make_shared<glim::EstimationFrame>();
    frame->id = id;
    frame->stamp = 100.0 + id;
    frame->frame_id = glim::FrameID::IMU;
    frame->T_lidar_imu = frame->T_world_imu = frame->T_world_lidar = Eigen::Isometry3d::Identity();
    frame->v_world_imu.setZero();
    frame->imu_bias.setZero();
    map->frames = {frame};
    map->odom_frames = {frame};
    glim::OdometryEstimationCallbacks::on_new_frame(frame);
    return map;
  }

  std::filesystem::path directory;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  rclcpp_lifecycle::LifecycleNode::SharedPtr node;
  rclcpp::Node::SharedPtr observer;
  std::unique_ptr<glim::RvizViewer> viewer;
};

// GLIM's callback registry is process-wide; exercise all updates with one viewer.
TEST_F(RvizViewerTest, ReleasesCloudHistoryAndPublishesConsistentSnapshots) {
  std::vector<std::weak_ptr<gtsam_points::PointCloud>> references;
  for (int i = 0; i < 12; ++i) {
    auto map = submap(i, i);
    references.push_back(map->frame);
    glim::GlobalMappingCallbacks::on_update_submaps(std::vector<glim::SubMap::Ptr>{map});
  }
  EXPECT_TRUE(wait_for([&] {
    return std::all_of(references.begin(), references.end(), [](const auto& p) { return p.expired(); });
  }, 2s)) << "No map subscriber should retain historical clouds";
  glim::GlobalMappingCallbacks::on_update_submaps(std::vector<glim::SubMap::Ptr>{});

  geometry_msgs::msg::PoseStamped::ConstSharedPtr pose;
  auto pose_sub = observer->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/viewer_test/pose", 10, [&](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { pose = msg; });
  ASSERT_TRUE(wait_for([&] { return node->count_subscribers("/viewer_test/pose") == 1; }));
  auto first = submap(12, 10);
  glim::GlobalMappingCallbacks::on_update_submaps(std::vector<glim::SubMap::Ptr>{first});
  ASSERT_TRUE(wait_for([&] {
    glim::OdometryEstimationCallbacks::on_new_frame(first->odom_frames.back());
    return pose && std::abs(pose->pose.position.x - 10.0) < 1e-6;
  })) << "Global pose updates must still work without a dense-map subscriber";

  std::vector<float> received;
  auto map_sub = observer->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/viewer_test/map", rclcpp::QoS(1).reliable().transient_local(),
    [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      EXPECT_EQ(msg->header.frame_id, "map");
      received.clear();
      sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x");
      for (; x != x.end(); ++x) received.push_back(*x);
      std::sort(received.begin(), received.end());
    });
  ASSERT_TRUE(wait_for([&] { return node->count_subscribers("/viewer_test/map") == 1; }));
  auto second = submap(13, 20);
  auto publish = [&](const std::vector<glim::SubMap::Ptr>& maps, const std::vector<float>& expected) {
    received.clear();
    // Keep the production ten-second throttle; retry until the next eligible update.
    return wait_for([&] {
      glim::GlobalMappingCallbacks::on_update_submaps(maps);
      return received == expected;
    });
  };
  ASSERT_TRUE(publish({first, second}, {13, 14, 15, 23.25, 24.25, 25.25}));
  first->T_world_origin.translation().x() = 30;
  second->T_world_origin.translation().x() = 40;
  ASSERT_TRUE(publish({first, second}, {33, 34, 35, 43.25, 44.25, 45.25}));

  std::weak_ptr<gtsam_points::PointCloud> retired = first->frame;
  first.reset();
  auto third = submap(14, 50);
  ASSERT_TRUE(publish({second, third}, {43.25, 44.25, 45.25, 53.5, 54.5, 55.5}));
  EXPECT_TRUE(wait_for([&] { return retired.expired(); }, 2s));
}
}  // namespace
