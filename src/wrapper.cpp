#include "dq_cpp/wrapper.h"

namespace dq_nmpc_control_nodelet {
// PLEASE DO NOT MOVE THESE ANYWHERE
// THEY BELONG HERE
// ELSE, EXPECT RANDOM PROBLEMS
solver_input acados_in;
solver_output acados_out;

NMPCWrapper::NMPCWrapper() {
    acados_ocp_capsule = quadrotor_acados_create_capsule();

    new_time_steps = NULL;
    status = quadrotor_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status) {
        printf("quadrotor_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    nlp_config = quadrotor_acados_get_nlp_config(acados_ocp_capsule);
    nlp_dims = quadrotor_acados_get_nlp_dims(acados_ocp_capsule);
    nlp_in = quadrotor_acados_get_nlp_in(acados_ocp_capsule);
    nlp_out = quadrotor_acados_get_nlp_out(acados_ocp_capsule);
    nlp_solver = quadrotor_acados_get_nlp_solver(acados_ocp_capsule);
    nlp_opts = quadrotor_acados_get_nlp_opts(acados_ocp_capsule);

    Eigen::Matrix<double, kStateSize, 1> hover_state(Eigen::Matrix<double, kStateSize, 1>::Zero());
    hover_state(0) = 1.0;

    // initialize states x and xN and input u.
    acados_initial_state_ = hover_state.template cast<double>();
    acados_states_ = hover_state.replicate(1, kSamples).template cast<double>();
    acados_inputs_ = kHoverInput_.replicate(1, kSamples).template cast<double>();

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", acados_in.x0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", acados_in.x0);

    // initialize references y and yN.
    acados_reference_states_.block(0, 0, kStateSize, kSamples) =
        hover_state.replicate(1, kSamples).template cast<double>();
    acados_reference_states_.block(kStateSize, 0, kInputSize, kSamples) = kHoverInput_.replicate(1, kSamples);
    acados_reference_end_state_.segment(0, kStateSize) = hover_state.template cast<double>();
}

void NMPCWrapper::initStates() {
    Eigen::Matrix<double, kStateSize, 1> hover_state(Eigen::Matrix<double, kStateSize, 1>::Zero());
    hover_state(0) = 1.0;
    kHoverInput_ = (Eigen::Matrix<real_t, kInputSize, 1>() << mass_ * gravity_, 0.0, 0.0, 0.0).finished();

    // initialize states x and xN and input u.
    acados_initial_state_ = hover_state.template cast<double>();
    acados_states_ = hover_state.replicate(1, kSamples).template cast<double>();
    acados_inputs_ = kHoverInput_.replicate(1, kSamples).template cast<double>();

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", acados_in.x0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", acados_in.x0);

    // initialize references y and yN.
    acados_reference_states_.block(0, 0, kStateSize, kSamples) =
        hover_state.replicate(1, kSamples).template cast<double>();
    acados_reference_states_.block(kStateSize, 0, kInputSize, kSamples) = kHoverInput_.replicate(1, kSamples);
    acados_reference_end_state_.segment(0, kStateSize) = hover_state.template cast<double>();
}

void NMPCWrapper::setTrajectory(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, kSamples>> states,
                                const Eigen::Ref<const Eigen::Matrix<double, kInputSize, kSamples>> inputs) {
    acados_reference_states_.block(0, 0, kStateSize, kSamples) =
        states.block(0, 0, kStateSize, kSamples).template cast<double>();
    acados_reference_states_.block(kStateSize, 0, kInputSize, kSamples) =
        inputs.block(0, 0, kInputSize, kSamples).template cast<double>();
    acados_reference_end_state_.segment(0, kStateSize) = states.col(kSamples - 1).template cast<double>();
}

bool NMPCWrapper::prepare(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state) {
    acados_states_ = state.replicate(1, kSamples).template cast<double>();
    acados_inputs_ = kHoverInput_.replicate(1, kSamples).template cast<double>();
    for (int i = 0; i <= N; i++) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", acados_out.x_out + i * NX);
    }
    for (int i = 0; i < N; i++) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", acados_out.u_out + i * NU);
    }
    return true;
}

bool NMPCWrapper::update(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state) {
    // this function provides as argument a NX by 1 vector which is the state
    // doing preparation step, this sets the solver to feedback and prepare phase
    int rti_phase = 0;  // zero sets rti_pahse to both prepare and feedback
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
    acados_is_prepared_ = true;

    // setting initial state
    acados_initial_state_ = state.template cast<double>();

    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, 0, "x", acados_in.x0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", acados_in.x0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", acados_in.x0);

    // loop over horizon and assign to each shooting node a segment of the acados_in.yref
    int y_indeces[yRefSize];
    std::iota(y_indeces, y_indeces + yRefSize, 0);
    for (int i = 0; i < N; i++) {
        quadrotor_acados_update_params_sparse(acados_ocp_capsule, i, y_indeces, acados_in.yref + i * yRefSize,
                                              yRefSize);
    }
    quadrotor_acados_update_params_sparse(acados_ocp_capsule, N, y_indeces, acados_in.yref_e, yRefSize);

    // solve NMPC optimization
    acados_status = quadrotor_acados_solve(acados_ocp_capsule);

    // getting solved states from acados
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &acados_out.x_out[ii * NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &acados_out.u_out[ii * NU]);

    return true;
}

void NMPCWrapper::getStates(Eigen::Matrix<double, kStateSize, kSamples> &return_states) {
    return_states = acados_states_.cast<double>();
}

void NMPCWrapper::getInputs(Eigen::Matrix<double, kInputSize, kSamples> &return_input) {
    return_input = acados_inputs_.cast<double>();
}

void NMPCWrapper::setMass(double mass) { mass_ = mass; }
void NMPCWrapper::setGravity(double gravity) { gravity_ = gravity; }

void NMPCWrapper::setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e, std::vector<double> R) {
    // concatenate parameters
    std::vector<double> params;
    params.reserve(Q.size() + Q_e.size() + R.size());
    for (const auto &vec : {Q, Q_e, R}) {
        params.insert(params.end(), vec.begin(), vec.end());
    }

    // send parameters to acados
    int params_size = params.size();
    int params_indeces[params_size];
    std::iota(params_indeces, params_indeces + params_size, yRefSize);
    for (int i = 0; i < N; i++) {
        quadrotor_acados_update_params_sparse(acados_ocp_capsule, i, params_indeces, params.data(), params_size);
    }
}

}  // namespace nmpc_control_nodelet