cmake_minimum_required(VERSION 3.8)
project(dq_cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(mujoco_msgs REQUIRED)

# ARM NEON flags
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
  message("enabling ARM neon optimizations")
endif()

# flags for speed (should already be enabled by default)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -O3")

set(acados_path "$ENV{ACADOS_SOURCE_DIR}")
set(acados_lib "${acados_path}/lib")
set(acados_generated_code "${PROJECT_SOURCE_DIR}/c_generated_code")

set(dependencies
  rclcpp
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  std_msgs
  sensor_msgs
  Eigen3
  yaml-cpp
  tf2
  tf2_ros
  rclcpp_components
  mujoco_msgs
)

include_directories(include
  ${EIGEN3_INCLUDE_DIR}
  ${acados_generated_code}
  ${acados_path}/interfaces
  ${acados_path}/acados
  ${acados_path}/acados/utils
  ${acados_path}
  ${acados_path}/external
  ${acados_path}/external/blasfeo/include
  ${acados_path}/external/hpipm/include
)
link_directories(${acados_generated_code})

add_library(dq_nmpc_control_nodelet SHARED src/nodelet.cpp src/wrapper.cpp src/nmpc_control.cpp)
set_target_properties(dq_nmpc_control_nodelet PROPERTIES COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL")

ament_target_dependencies(dq_nmpc_control_nodelet ${dependencies})
target_link_libraries(dq_nmpc_control_nodelet
  ${YAML_CPP_LIBRARIES}
  libacados_ocp_solver_quadrotor.so
  ${acados_lib}/libacados.so
)

rclcpp_components_register_nodes(dq_nmpc_control_nodelet "dq_nmpc_control_nodelet::NMPCControlNodelet")
ament_export_libraries(dq_nmpc_control_nodelet)
ament_export_dependencies(${dependencies})

install(TARGETS
  dq_nmpc_control_nodelet
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()