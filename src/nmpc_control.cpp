#include "dq_cpp/nmpc_control.h"

namespace dq_nmpc_control_nodelet {
NMPCControl::NMPCControl()
    : solve_from_scratch_(true),
      current_state_(Eigen::Matrix<double, kStateSize, 1>::Zero()),
      reference_states_(Eigen::Matrix<double, kStateSize, kSamples>::Zero()),
      reference_inputs_(Eigen::Matrix<double, kInputSize, kSamples>::Zero()),
      predicted_states_(Eigen::Matrix<double, kStateSize, kSamples>::Zero()),
      predicted_inputs_(Eigen::Matrix<double, kInputSize, kSamples>::Zero()) {
        
    // initialize quaternion w to 1
    current_state_(0) = 1.0;
    reference_states_.row(0).setOnes();
    predicted_states_.row(0).setOnes();
}

void NMPCControl::setState(const Eigen::Matrix<double, kStateSize, 1> &state, double stamp) {
    current_state_.block(0, 0, 14, 1) = state.block(0, 0, 14, 1);
    stamp_current_state_ = stamp;
}
void NMPCControl::setOmega(const Eigen::Matrix<double, 3, 1> &omega) { current_state_.block(8, 0, 3, 1) = omega; }
void NMPCControl::setReferenceStates(const Eigen::Matrix<double, kStateSize, kSamples> &reference_states) {
    reference_states_ = reference_states;
}
void NMPCControl::setReferenceInputs(const Eigen::Matrix<double, kInputSize, kSamples> &reference_inputs) {
    reference_inputs_ = reference_inputs;
}

void NMPCControl::setMass(double mass) { wrapper_.setMass(mass); }
void NMPCControl::setGravity(double gravity) { wrapper_.setGravity(gravity); }
void NMPCControl::setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e, std::vector<double> R) {
    wrapper_.setWeightMatrices(Q, Q_e, R);
}

double NMPCControl::getStampState() { return stamp_current_state_; }

Eigen::Matrix<double, kStateSize, 1> NMPCControl::getPredictedState() { return predicted_states_.col(1); }
Eigen::Matrix<double, kInputSize, 1> NMPCControl::getPredictedInput() { return predicted_inputs_.col(0); }
Eigen::Matrix<double, kStateSize, kSamples> NMPCControl::getPredictedStates() { return predicted_states_; }
Eigen::Matrix<double, kStateSize, kSamples> NMPCControl::getReferenceStates() { return reference_states_; }
Eigen::Matrix<double, kInputSize, kSamples> NMPCControl::getReferenceInputs() { return reference_inputs_; }

int NMPCControl::run() {

    wrapper_.setTrajectory(reference_states_, reference_inputs_);
    if (solve_from_scratch_) {
        std::cout << "Solving NMPC with hover as initial guess.\n";
        wrapper_.prepare(current_state_);
        solve_from_scratch_ = false;
    }

    int acados_status = wrapper_.update(current_state_);

    wrapper_.getStates(predicted_states_);
    wrapper_.getInputs(predicted_inputs_);

    return acados_status;
}

}  // namespace nmpc_control_nodelet