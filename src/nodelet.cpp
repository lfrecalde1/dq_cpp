#include <dq_cpp/nmpc_control.h>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <quadrotor_msgs/msg/aux_command.hpp>
#include <quadrotor_msgs/msg/position_command.hpp>
#include <quadrotor_msgs/msg/trajectory_point.hpp>
#include <quadrotor_msgs/msg/trpy_command.hpp>
#include <mujoco_msgs/msg/dual.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>

namespace dq_nmpc_control_nodelet {
class NMPCControlNodelet : public rclcpp::Node {
public:
    NMPCControlNodelet(const rclcpp::NodeOptions &options)
        : Node("nmpc_control_nodelet", options),
          frame_id_("world"),
          enable_motors_(false),
          _optimization_error(false),
          _aux_initial(false),
          set_pre_odom_quat_(false) {
        this->declare_parameter("mass", 0.0);
        this->declare_parameter("gravity", 0.0);
        this->declare_parameter("kf", 0.0);
        this->declare_parameter("km", 0.0);
        this->declare_parameter("frame_dx", 0.0);
        this->declare_parameter("frame_dy", 0.0);
        logParameter("mass", mass_, "%.4f");
        logParameter("gravity", gravity_, "%.4f");
        logParameter("kf", kf_, "%.14f");
        logParameter("km", km_, "%.14f");
        logParameter("frame_dx", frame_dx_, "%.4f");
        logParameter("frame_dy", frame_dy_, "%.4f");

        inertia_matrix_ = Eigen::Matrix3d::Zero();
        this->declare_parameter("ixx", 0.0);
        this->declare_parameter("iyy", 0.0);
        this->declare_parameter("izz", 0.0);
        logParameter("ixx", inertia_matrix_(0, 0), "%.4f");
        logParameter("iyy", inertia_matrix_(1, 1), "%.4f");
        logParameter("izz", inertia_matrix_(2, 2), "%.4f");

        this->declare_parameter("platform_type", "");
        logParameter("platform_type", platform_type_, "%s");

        this->declare_parameter("nmpc.Q", rclcpp::PARAMETER_DOUBLE_ARRAY);
        this->declare_parameter("nmpc.Q_e", rclcpp::PARAMETER_DOUBLE_ARRAY);
        this->declare_parameter("nmpc.R", rclcpp::PARAMETER_DOUBLE_ARRAY);
        rclcpp::Parameter Q_param = this->get_parameter("nmpc.Q");
        rclcpp::Parameter Q_e_param = this->get_parameter("nmpc.Q_e");
        rclcpp::Parameter R_param = this->get_parameter("nmpc.R");
        RCLCPP_INFO(this->get_logger(), "[NMPC] Q: %s", Q_param.value_to_string().c_str());
        RCLCPP_INFO(this->get_logger(), "[NMPC] Q_e: %s", Q_e_param.value_to_string().c_str());
        RCLCPP_INFO(this->get_logger(), "[NMPC] R: %s", R_param.value_to_string().c_str());
        Q_param_ = Q_param.as_double_array();
        Q_e_param_ = Q_e_param.as_double_array();
        R_param_ = R_param.as_double_array();

        // NOTE: this mixer matrix misses kf multiplication over each element. This
        //       is because we have thrusts in the control input implementation of NMPC!
        Eigen::Matrix4d mixer_matrix;
        if (platform_type_ == "voxl2" || platform_type_ == "raxl") {
            mixer_matrix << 1, 1, 1, 1, frame_dy_, frame_dy_, -frame_dy_, -frame_dy_, -frame_dx_, frame_dx_,
                frame_dx_, -frame_dx_, km_ / kf_, -km_ / kf_, km_ / kf_, -km_ / kf_;
            mixer_matrix_inv_ = mixer_matrix.inverse();
        }
        else if (platform_type_ == "race" || platform_type_ == "race_S") {
            mixer_matrix << 1, 1, 1, 1, -frame_dy_, frame_dy_, frame_dy_, -frame_dy_, -frame_dx_, frame_dx_,
                -frame_dx_, frame_dx_, -km_ / kf_, -km_ / kf_, km_ / kf_, km_ / kf_;
            mixer_matrix_inv_ = mixer_matrix.inverse();
        }
        else if (platform_type_ == "iris") {
            mixer_matrix << 1, 1, 1, 1, frame_dy_, -frame_dy_, -frame_dy_, frame_dy_, -frame_dx_, frame_dx_,
                -frame_dx_, frame_dx_, km_ / kf_, km_ / kf_, -km_ / kf_, -km_ / kf_;
            mixer_matrix_inv_ = mixer_matrix.inverse();
        }
        else {
            std::cerr << "\n\n[ERROR] Cannot set Mixer Matrix Inv - "
                         "Wrong platform type for moments, implemented are "
                         "race, raceS, iris, voxl2, raxl!\n\n"
                      << std::endl;
            exit(0);
        }

        clock_ = rclcpp::Clock();
        pre_odom_quat_ << 1.0, 0.0, 0.0, 0.0;
        kom_ = {0.13, 0.13, 1.0};
        kr_ = {1.5, 1.5, 1.0};

        hover_thrust_ = mass_ * gravity_/4;

        controller_.setMass(mass_);
        controller_.setGravity(gravity_);
        controller_.setWeightMatrices(Q_param_, Q_e_param_, R_param_);

        // custom QoS
        auto qos_profile = rclcpp::SensorDataQoS();

        pub_trpy_cmd_ = this->create_publisher<quadrotor_msgs::msg::TRPYCommand>("trpy_cmd", 1);
        pub_ref_traj_ = this->create_publisher<nav_msgs::msg::Path>("reference_path", 1);
        pub_pred_traj_ = this->create_publisher<nav_msgs::msg::Path>("predicted_path", 1);
        //pub_dual_ = this->create_publisher<mujoco_msgs::msg::Dual>("dual_cpp", 10);

        sub_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", qos_profile, std::bind(&NMPCControlNodelet::odomCallback, this, std::placeholders::_1));
        sub_position_cmd_ = this->create_subscription<quadrotor_msgs::msg::PositionCommand>(
            "position_cmd", 1, std::bind(&NMPCControlNodelet::referenceCallback, this, std::placeholders::_1));
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu", 1, std::bind(&NMPCControlNodelet::imuCallback, this, std::placeholders::_1));
        sub_motors_ = this->create_subscription<std_msgs::msg::Bool>(
            "motors", 1, std::bind(&NMPCControlNodelet::motorsCallback, this, std::placeholders::_1));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    template <typename T>
    void logParameter(const std::string &param_name, T &param_value, const std::string &format) {
        if (!this->get_parameter(param_name, param_value)) {
            RCLCPP_ERROR(this->get_logger(), "[NMPC] No %s!", param_name.c_str());
        }
        else {
            if constexpr (std::is_same_v<T, std::string>) {
                RCLCPP_INFO(this->get_logger(), "[NMPC] %s: %s", param_name.c_str(), param_value.c_str());
            }
            else {
                RCLCPP_INFO(this->get_logger(), ("[NMPC] " + param_name + ": " + format).c_str(), param_value);
            }
        }
    }

    NMPCControl controller_;
    rclcpp::Clock clock_;
    double hover_thrust_;

    std::array<double, 3> kom_;
    std::array<double, 3> kr_;

    // from odom callback
    std::string frame_id_;
    Eigen::Vector4d pre_odom_quat_;
    bool enable_motors_;
    bool _optimization_error;
    bool _aux_initial;
    bool set_pre_odom_quat_;

    // from param server
    double mass_;
    double gravity_;
    double kf_;
    double km_;
    double frame_dx_;
    double frame_dy_;
    Eigen::Matrix4d mixer_matrix_inv_;
    Eigen::Matrix3d inertia_matrix_;
    std::string platform_type_;
    std::vector<double> Q_param_;
    std::vector<double> Q_e_param_;
    std::vector<double> R_param_;

    // ros2
    void run();
    void logParameter();
    void publishControl(Eigen::Matrix<double, kStateSize, 1> pred_state,
                        Eigen::Matrix<double, kInputSize, 1> pred_input);
    void publishSafeControl();
    void publishReference();
    void publishPrediction();
    void referenceCallback(const quadrotor_msgs::msg::PositionCommand::SharedPtr pos_cmd);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg);
    void motorsCallback(const std_msgs::msg::Bool::SharedPtr msg);

    rclcpp::Publisher<quadrotor_msgs::msg::TRPYCommand>::SharedPtr pub_trpy_cmd_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_ref_traj_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pred_traj_;
    //rclcpp::Publisher<mujoco_msgs::msg::Dual>::SharedPtr pub_dual_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
    rclcpp::Subscription<quadrotor_msgs::msg::PositionCommand>::SharedPtr sub_position_cmd_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_motors_;

};

void NMPCControlNodelet::referenceCallback(const quadrotor_msgs::msg::PositionCommand::SharedPtr reference_msg) {
    
    // filter quaternion
    quadrotor_msgs::msg::PositionCommand::SharedPtr filt_reference_msg(reference_msg);
    Eigen::Vector4d pre_quat(pre_odom_quat_(0), pre_odom_quat_(1), pre_odom_quat_(2), pre_odom_quat_(3));
    Eigen::Vector4d current_quat;
    for (auto point : filt_reference_msg->points) {
        current_quat =
            Eigen::Vector4d(point.quaternion.w, point.quaternion.x, point.quaternion.y, point.quaternion.z);
        if (current_quat.dot(pre_quat) < 0) {
            point.quaternion.w = -point.quaternion.w;
            point.quaternion.x = -point.quaternion.x;
            point.quaternion.y = -point.quaternion.y;
            point.quaternion.z = -point.quaternion.z;
            current_quat = -current_quat;
        }
        pre_quat = current_quat;
    }

    Eigen::Matrix<double, kStateSize, kSamples> reference_states;
    Eigen::Matrix<double, kInputSize, kSamples> reference_inputs;
    reference_states = Eigen::Matrix<double, kStateSize, kSamples>::Zero();
    reference_inputs = Eigen::Matrix<double, kInputSize, kSamples>::Zero();

    Eigen::Vector4d force_moments = Eigen::Vector4d::Zero();
    Eigen::Vector3d ang_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d moments = Eigen::Vector3d::Zero();

    if (filt_reference_msg->points.size() == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), clock_, 1000, "[NMPC] Reference has no points.");
        return;
    }

    auto iterator(filt_reference_msg->points.begin());
    for (int i = 0; i < kSamples; i++) {
        Eigen::Matrix<double, kStateSize, 1> dual;
        Eigen::Matrix<double, 13, 1> state;
        state(0) = iterator->position.x;
        state(1) = iterator->position.y;
        state(2) = iterator->position.z;

        state(3) = iterator->velocity.x;
        state(4) = iterator->velocity.y;
        state(5) = iterator->velocity.z;

        state(6) = iterator->quaternion.w;
        state(7) = iterator->quaternion.x;
        state(8) = iterator->quaternion.y;
        state(9) = iterator->quaternion.z;

        state(10) = iterator->angular_velocity.x;
        state(11) = iterator->angular_velocity.y;
        state(12) = iterator->angular_velocity.z;

        Eigen::Matrix<double, 4, 1> t;
        Eigen::Matrix<double, 4, 1> q;
        Eigen::Matrix<double, 3, 1> w;
        Eigen::Matrix<double, 3, 1> v;
        // Translation 
        t << 0.0, state(0), state(1), state(2);
        // Quaternion
        q << state(6), state(7), state(8), state(9);
        v << state(3), state(4), state(5);
        w << state(10), state(11), state(12);
        
        // Define the H_plus_q matrix
        Eigen::Matrix<double, 4, 4> H_plus_t;
        H_plus_t << t(0), -t(1), -t(2), -t(3),
                    t(1),  t(0), -t(3),  t(2),
                    t(2),  t(3),  t(0), -t(1),
                    t(3), -t(2),  t(1),  t(0);
        Eigen::Matrix<double, 4, 1> aux_dual =(0.5) * H_plus_t * q;
        // Rotation Velocity Body frame
        // Convert the vector to a pure quaternion (0, vector)
        Eigen::Matrix<double, 4, 1> vector;
        vector << 0.0, v(0), v(1), v(2);
        // Compute the conjugate of the quaternion
        Eigen::Matrix<double, 4, 1> quat_c;
        Eigen::Matrix<double, 4, 1> quat;
        quat_c << q(0), -q(1), -q(2), -q(3);
        quat << q(0), q(1), q(2), q(3);
        // Define the H_plus_q_c matrix for the quaternion conjugate
        Eigen::Matrix<double, 4, 4> H_plus_q_c;
        H_plus_q_c << quat_c(0), -quat_c(1), -quat_c(2), -quat_c(3),
                        quat_c(1),  quat_c(0), -quat_c(3),  quat_c(2),
                        quat_c(2),  quat_c(3),  quat_c(0), -quat_c(1),
                        quat_c(3), -quat_c(2),  quat_c(1),  quat_c(0);
        // Perform the first multiplication
        Eigen::Matrix<double, 4, 1> aux_value = H_plus_q_c * vector;
        // Define the H_plus_aux matrix for the result of the first multiplication
        Eigen::Matrix<double, 4, 4> H_plus_aux;
        H_plus_aux << aux_value(0), -aux_value(1), -aux_value(2), -aux_value(3),
                      aux_value(1),  aux_value(0), -aux_value(3),  aux_value(2),
                      aux_value(2),  aux_value(3),  aux_value(0), -aux_value(1),
                      aux_value(3), -aux_value(2),  aux_value(1),  aux_value(0);
        // Perform the second multiplication
        Eigen::Matrix<double, 4, 1> vector_b = H_plus_aux * quat;
        // Final Dual quat and Twist
        dual(0) = q(0);
        dual(1) = q(1);
        dual(2) = q(2);
        dual(3) = q(3);

        dual(4) = aux_dual(0);
        dual(5) = aux_dual(1);
        dual(6) = aux_dual(2);
        dual(7) = aux_dual(3);

        dual(8) = state(10);
        dual(9) = state(11);
        dual(10) = state(12);

        dual(11) = vector_b(1);
        dual(12) = vector_b(2);
        dual(13) = vector_b(3);

        reference_states.col(i) << dual(0), dual(1), dual(2), dual(3),
            dual(4), dual(5), dual(6), dual(7),
            dual(8), dual(9), dual(10),
            dual(11), dual(12), dual(13);

        ang_vel << iterator->angular_velocity.x, iterator->angular_velocity.y, iterator->angular_velocity.z;
        ang_acc << iterator->angular_velocity_dot.x, iterator->angular_velocity_dot.y,
            iterator->angular_velocity_dot.z;
        moments = inertia_matrix_ * ang_acc + ang_vel.cross(inertia_matrix_ * ang_vel);
        force_moments << iterator->force, moments;
        reference_inputs.col(i) << iterator->force, moments(0), moments(1), moments(2);
        iterator++;
    }
    controller_.setReferenceStates(reference_states);
    controller_.setReferenceInputs(reference_inputs);

    if ((reference_msg->header.stamp.sec + reference_msg->header.stamp.nanosec * 1e-9) -
            controller_.getStampState() > 0.01)
        RCLCPP_WARN_THROTTLE(this->get_logger(), clock_, 1000, "[NMPC] Outdated odometry.");

    // Run controller but stop when error found
    if (!_optimization_error) {
        run();
    }
    //controller_.run();

    //// publish control and predicted path
    //publishControl();
    //publishReference();
    //publishPrediction();
    //RCLCPP_INFO(this->get_logger(), "DQ-NMPC");

}

void NMPCControlNodelet::run() {

    // Run controller
    int acados_status = controller_.run();

    // Check if NMPC found a solution
    switch (acados_status) {
        case 1:
            if (_aux_initial){
                RCLCPP_WARN(this->get_logger(), 
                    "[NMPC] ACADOS_FAILURE: could not find a solution!");
                publishSafeControl();
                _optimization_error = true;
                return;
            }
            else{
                //RCLCPP_WARN(this->get_logger(), 
                    //"[NMPC] ACADOS_FIRST_SOLUTION!");
                    break;;
            }
        case 2:
            RCLCPP_WARN(this->get_logger(), 
                "[NMPC] ACADOS_MAXITER: maximum number of iterations reached!");
            publishSafeControl();
            _optimization_error = true;
            return;
        case 3:
            RCLCPP_WARN(this->get_logger(), 
                "[NMPC] ACADOS_MINSTEP: minimum step size in QP solver reached!");
            publishSafeControl();
            _optimization_error = true;
            return;
        case 4:
            RCLCPP_WARN(this->get_logger(), 
                "[NMPC] ACADOS_QP_FAILURE: qp solver failed!");
            publishSafeControl();
            _optimization_error = true;
            return;
    }

    // Get solution
    Eigen::Matrix<double, kStateSize, 1> pred_state;
    Eigen::Matrix<double, kInputSize, 1> pred_input;
    pred_state = controller_.getPredictedState();
    pred_input = controller_.getPredictedInput();

    // Check if solution has no NaN values
    bool has_nan_in_state = pred_state.array().isNaN().any();
    bool has_nan_in_input = pred_input.array().isNaN().any();
    if (has_nan_in_state || has_nan_in_input) {
        RCLCPP_WARN(this->get_logger(), "[NMPC] NaN in current solution!");
        _optimization_error = true;
        _aux_initial = true;
        return;
    }

    // Publish solution
    publishControl(pred_state, pred_input);
    publishPrediction();
    publishReference();
}

void NMPCControlNodelet::odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    Eigen::Matrix<double, 13, 1> state;
    frame_id_ = odom_msg->header.frame_id;
    state(0) = odom_msg->pose.pose.position.x;
    state(1) = odom_msg->pose.pose.position.y;
    state(2) = odom_msg->pose.pose.position.z;

    state(3) = odom_msg->twist.twist.linear.x;
    state(4) = odom_msg->twist.twist.linear.y;
    state(5) = odom_msg->twist.twist.linear.z;

    state(10) = odom_msg->twist.twist.angular.x;
    state(11) = odom_msg->twist.twist.angular.y;
    state(12) = odom_msg->twist.twist.angular.z;

    Eigen::Vector4d current_odom_quat(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                                      odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);

    // set initial quaternion from odometry
    if (not set_pre_odom_quat_) {
        pre_odom_quat_ = current_odom_quat;
        set_pre_odom_quat_ = true;
    }

    if (current_odom_quat.dot(pre_odom_quat_) < 0) {
        state(6) = -odom_msg->pose.pose.orientation.w;
        state(7) = -odom_msg->pose.pose.orientation.x;
        state(8) = -odom_msg->pose.pose.orientation.y;
        state(9) = -odom_msg->pose.pose.orientation.z;
        pre_odom_quat_ = -current_odom_quat;
    }
    else {
        state(6) = odom_msg->pose.pose.orientation.w;
        state(7) = odom_msg->pose.pose.orientation.x;
        state(8) = odom_msg->pose.pose.orientation.y;
        state(9) = odom_msg->pose.pose.orientation.z;
        pre_odom_quat_ = current_odom_quat;
    }

    Eigen::Matrix<double, kStateSize, 1> dual;
    Eigen::Matrix<double, 4, 1> t;
    Eigen::Matrix<double, 4, 1> q;
    Eigen::Matrix<double, 3, 1> w;
    Eigen::Matrix<double, 3, 1> v;

            // Translation 
    t << 0.0, state(0), state(1), state(2);

    // Quaternion
    q << state(6), state(7), state(8), state(9);

    v << state(3), state(4), state(5);

    w << state(10), state(11), state(12);
    // Define the H_plus_q matrix
    Eigen::Matrix<double, 4, 4> H_plus_t;
    H_plus_t << t(0), -t(1), -t(2), -t(3),
                t(1),  t(0), -t(3),  t(2),
                t(2),  t(3),  t(0), -t(1),
                t(3), -t(2),  t(1),  t(0);

    Eigen::Matrix<double, 4, 1> aux_dual =(0.5) * H_plus_t * q;

    // Rotation Velocity Body frame
    // Convert the vector to a pure quaternion (0, vector)
    Eigen::Matrix<double, 4, 1> vector;
    vector << 0.0, v(0), v(1), v(2);
    // Compute the conjugate of the quaternion
    Eigen::Matrix<double, 4, 1> quat_c;
    Eigen::Matrix<double, 4, 1> quat;
    quat_c << q(0), -q(1), -q(2), -q(3);
    quat << q(0), q(1), q(2), q(3);

    // Define the H_plus_q_c matrix for the quaternion conjugate
    Eigen::Matrix<double, 4, 4> H_plus_q_c;
    H_plus_q_c << quat_c(0), -quat_c(1), -quat_c(2), -quat_c(3),
                    quat_c(1),  quat_c(0), -quat_c(3),  quat_c(2),
                    quat_c(2),  quat_c(3),  quat_c(0), -quat_c(1),
                    quat_c(3), -quat_c(2),  quat_c(1),  quat_c(0);

    // Perform the first multiplication
    Eigen::Matrix<double, 4, 1> aux_value = H_plus_q_c * vector;

    // Define the H_plus_aux matrix for the result of the first multiplication
    Eigen::Matrix<double, 4, 4> H_plus_aux;
    H_plus_aux << aux_value(0), -aux_value(1), -aux_value(2), -aux_value(3),
                    aux_value(1),  aux_value(0), -aux_value(3),  aux_value(2),
                    aux_value(2),  aux_value(3),  aux_value(0), -aux_value(1),
                    aux_value(3), -aux_value(2),  aux_value(1),  aux_value(0);

    // Perform the second multiplication
    Eigen::Matrix<double, 4, 1> vector_b = H_plus_aux * quat;

    // Final Dual quat and Twist
    dual(0) = q(0);
    dual(1) = q(1);
    dual(2) = q(2);
    dual(3) = q(3);
    dual(4) = aux_dual(0);
    dual(5) = aux_dual(1);
    dual(6) = aux_dual(2);
    dual(7) = aux_dual(3);
    dual(8) = state(10);
    dual(9) = state(11);
    dual(10) = state(12);
    dual(11) = vector_b(1);
    dual(12) = vector_b(2);
    dual(13) = vector_b(3);

    controller_.setState(dual, odom_msg->header.stamp.sec + odom_msg->header.stamp.nanosec * 1e-9);

    //auto msg = mujoco_msgs::msg::Dual();
    //msg.header.stamp = this->get_clock()->now();
    //msg.d_0 = dual(0);
    //msg.d_1 = dual(1);
    //msg.d_2 = dual(2);
    //msg.d_3 = dual(3);
    //msg.d_4 = dual(4);
    //msg.d_5 = dual(5);
    //msg.d_6 = dual(6);
    //msg.d_7 = dual(7);

    //msg.twist_0 = dual(8);
    //msg.twist_1 = dual(9);
    //msg.twist_2 = dual(10);
    //msg.twist_3 = dual(11);
    //msg.twist_4 = dual(12);
    //msg.twist_5 = dual(13);
    //pub_dual_->publish(msg);
}

void NMPCControlNodelet::imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    Eigen::Matrix<double, 3, 1> omega;
    omega(0) = imu_msg->angular_velocity.x;
    omega(1) = imu_msg->angular_velocity.y;
    omega(2) = imu_msg->angular_velocity.z;
    controller_.setOmega(omega);
}


void NMPCControlNodelet::motorsCallback(const std_msgs::msg::Bool::SharedPtr motors_msg) {
    if (motors_msg->data)
        RCLCPP_INFO(this->get_logger(), "Enabling Motors");
    else
        RCLCPP_INFO(this->get_logger(), "Disabling Motors");
    enable_motors_ = motors_msg->data;
}

void NMPCControlNodelet::publishControl(Eigen::Matrix<double, kStateSize, 1> pred_state,
                                        Eigen::Matrix<double, kInputSize, 1> pred_input) {
    //Eigen::Matrix<double, kStateSize, 1> pred_state = controller_.getPredictedState();
    //Eigen::Matrix<double, kInputSize, 1> pred_input = controller_.getPredictedInput();
    quadrotor_msgs::msg::TRPYCommand trpy_msg;
    Eigen::Quaternion<double> orientation(pred_state(0), pred_state(1), pred_state(2), pred_state(3));
    orientation = orientation.normalized();
    trpy_msg.header.stamp = clock_.now();
    trpy_msg.header.frame_id = frame_id_;
    trpy_msg.quaternion.x = pred_state(1);
    trpy_msg.quaternion.y = pred_state(2);
    trpy_msg.quaternion.z = pred_state(3);
    trpy_msg.quaternion.w = pred_state(0);
    trpy_msg.kom = kom_;
    trpy_msg.kr = kr_;
    trpy_msg.aux.enable_motors = enable_motors_;
    trpy_msg.thrust = pred_input(0);
    trpy_msg.angular_velocity.x = pred_state(8);
    trpy_msg.angular_velocity.y = pred_state(9);
    trpy_msg.angular_velocity.z = pred_state(10);
    pub_trpy_cmd_->publish(trpy_msg);
}

void NMPCControlNodelet::publishSafeControl() {
    quadrotor_msgs::msg::TRPYCommand trpy_msg;
    Eigen::Quaternion<double> orientation(1.0, 0.0, 0.0, 0.0);
    trpy_msg.header.stamp = clock_.now();
    trpy_msg.header.frame_id = frame_id_;
    trpy_msg.quaternion.w = orientation.w();
    trpy_msg.quaternion.x = orientation.x();
    trpy_msg.quaternion.y = orientation.y();
    trpy_msg.quaternion.z = orientation.z();
    trpy_msg.kom = kom_;
    trpy_msg.kr = kr_;
    trpy_msg.aux.enable_motors = enable_motors_;
    trpy_msg.thrust = 0.0;
    trpy_msg.angular_velocity.x = 0.0;
    trpy_msg.angular_velocity.y = 0.0;
    trpy_msg.angular_velocity.z = 0.0;
    pub_trpy_cmd_->publish(trpy_msg);
}

void NMPCControlNodelet::publishReference() {
    Eigen::Matrix<double, kStateSize, kSamples> reference_states = controller_.getReferenceStates();
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = clock_.now();
    path_msg.header.frame_id = frame_id_;
    path_msg.poses.reserve(kSamples);
    geometry_msgs::msg::PoseStamped pose;

    // Dual quaternion
    Eigen::Matrix<double, kStateSize, 1> dual_ref;
    Eigen::Matrix<double, 4, 1> quat_c;
    Eigen::Matrix<double, 4, 1> quat;
    Eigen::Matrix<double, 4, 1> dual_part;
    Eigen::Matrix<double, 4, 1> t_part;
    Eigen::Matrix<double, 4, 4> H_plus_dual_part;
    for (int i = 0; i < kSamples; i++) {
        // Set dual quaternions values
        dual_ref(0) = reference_states(0, i);
        dual_ref(1) = reference_states(1, i);
        dual_ref(2) = reference_states(2, i);
        dual_ref(3) = reference_states(3, i);
        dual_ref(4) = reference_states(4, i);
        dual_ref(5) = reference_states(5, i);
        dual_ref(6) = reference_states(6, i);
        dual_ref(7) = reference_states(7, i);

        // Get quat
        quat_c << dual_ref(0), -dual_ref(1), -dual_ref(2), -dual_ref(3);
        quat << dual_ref(0), dual_ref(1), dual_ref(2), dual_ref(3);

        // get dual part
        dual_part << dual_ref(4), dual_ref(5), dual_ref(6), dual_ref(7);

        H_plus_dual_part << dual_part(0), -dual_part(1), -dual_part(2), -dual_part(3),
                    dual_part(1),  dual_part(0), -dual_part(3),  dual_part(2),
                    dual_part(2),  dual_part(3),  dual_part(0), -dual_part(1),
                    dual_part(3), -dual_part(2),  dual_part(1),  dual_part(0);

        t_part = 2 * H_plus_dual_part * quat_c;

        pose.pose.position.x = t_part(1);
        pose.pose.position.y = t_part(2);
        pose.pose.position.z = t_part(3);
        pose.pose.orientation.w = quat(0);
        pose.pose.orientation.x = quat(1);
        pose.pose.orientation.y = quat(2);
        pose.pose.orientation.z = quat(3);

        path_msg.poses.push_back(pose);
    }
    pub_ref_traj_->publish(path_msg);
}

void NMPCControlNodelet::publishPrediction() {
    Eigen::Matrix<double, kStateSize, kSamples> predicted_states = controller_.getPredictedStates();
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = clock_.now();
    path_msg.header.frame_id = frame_id_;
    path_msg.poses.reserve(kSamples);
    geometry_msgs::msg::PoseStamped pose;

    // Dual quaternion
    Eigen::Matrix<double, kStateSize, 1> dual_ref;
    Eigen::Matrix<double, 4, 1> quat_c;
    Eigen::Matrix<double, 4, 1> quat;
    Eigen::Matrix<double, 4, 1> dual_part;
    Eigen::Matrix<double, 4, 1> t_part;
    Eigen::Matrix<double, 4, 4> H_plus_dual_part;
    for (int i = 0; i < kSamples; i++) {
        // Set dual quaternions values
        dual_ref(0) = predicted_states(0, i);
        dual_ref(1) = predicted_states(1, i);
        dual_ref(2) = predicted_states(2, i);
        dual_ref(3) = predicted_states(3, i);
        dual_ref(4) = predicted_states(4, i);
        dual_ref(5) = predicted_states(5, i);
        dual_ref(6) = predicted_states(6, i);
        dual_ref(7) = predicted_states(7, i);

        // Get quat
        quat_c << dual_ref(0), -dual_ref(1), -dual_ref(2), -dual_ref(3);
        quat << dual_ref(0), dual_ref(1), dual_ref(2), dual_ref(3);

        // get dual part
        dual_part << dual_ref(4), dual_ref(5), dual_ref(6), dual_ref(7);

        H_plus_dual_part << dual_part(0), -dual_part(1), -dual_part(2), -dual_part(3),
                    dual_part(1),  dual_part(0), -dual_part(3),  dual_part(2),
                    dual_part(2),  dual_part(3),  dual_part(0), -dual_part(1),
                    dual_part(3), -dual_part(2),  dual_part(1),  dual_part(0);

        t_part = 2 * H_plus_dual_part * quat_c;

        pose.pose.position.x = t_part(1);
        pose.pose.position.y = t_part(2);
        pose.pose.position.z = t_part(3);
        pose.pose.orientation.w = quat(0);
        pose.pose.orientation.x = quat(1);
        pose.pose.orientation.y = quat(2);
        pose.pose.orientation.z = quat(3);
        path_msg.poses.push_back(pose);
    }
    pub_pred_traj_->publish(path_msg);
}

}  // namespace nmpc_control_nodelet

RCLCPP_COMPONENTS_REGISTER_NODE(dq_nmpc_control_nodelet::NMPCControlNodelet)
