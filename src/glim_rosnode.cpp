#include <iostream>
#include <spdlog/spdlog.h>
#include <rclcpp/rclcpp.hpp>

#include <glim_ros/glim_ros.hpp>
#include <glim/util/config.hpp>
#include <glim/util/extension_module_ros2.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::NodeOptions options;

  auto glim = std::make_shared<glim::GlimROS>(options);

  rclcpp::spin(glim->get_node_base_interface());
  // spin() also returns when on_activate()'s guard already called
  // rclcpp::shutdown() itself (forcing a process exit) -- calling it again
  // on an already-shutdown context throws, so only call it if still needed.
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  std::string dump_path = "/tmp/dump";
  // on_configure() already declares "dump_path" (it needs the value earlier,
  // to set dump_path_) -- redeclaring it here threw
  // ParameterAlreadyDeclaredException on any run that actually reaches this
  // line, which nothing did until on_activate()'s guard started forcing a
  // clean self-shutdown instead of leaving the node stuck inactive.
  if (!glim->has_parameter("dump_path")) {
    glim->declare_parameter<std::string>("dump_path", dump_path);
  }
  glim->get_parameter<std::string>("dump_path", dump_path);

  glim->wait();
  glim->save(dump_path);

  return 0;
}