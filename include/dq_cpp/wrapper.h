#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>
#include <boost/array.hpp>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <numeric>

#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"
#include "acados/utils/math.h"
#include "acados/utils/print.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_quadrotor.h"
#include "quadrotor_model/quadrotor_model.h"


#define NX QUADROTOR_NX
#define NZ QUADROTOR_NZ
#define NU QUADROTOR_NU
#define NP QUADROTOR_NP
#define NBX QUADROTOR_NBX
#define NBX0 QUADROTOR_NBX0
#define NBU QUADROTOR_NBU
#define NSBX QUADROTOR_NSBX
#define NSBU QUADROTOR_NSBU
#define NSH QUADROTOR_NSH
#define NSG QUADROTOR_NSG
#define NSPHI QUADROTOR_NSPHI
#define NSHN QUADROTOR_NSHN
#define NSGN QUADROTOR_NSGN
#define NSPHIN QUADROTOR_NSPHIN
#define NSBXN QUADROTOR_NSBXN
#define NS QUADROTOR_NS
#define NSN QUADROTOR_NSN
#define NG QUADROTOR_NG
#define NBXN QUADROTOR_NBXN
#define NGN QUADROTOR_NGN
#define NY0 QUADROTOR_NY0
#define NY QUADROTOR_NY
#define NYN QUADROTOR_NYN
#define NH QUADROTOR_NH
#define NPHI QUADROTOR_NPHI
#define NHN QUADROTOR_NHN
#define NPHIN QUADROTOR_NPHIN
#define NR QUADROTOR_NR
const int N = QUADROTOR_N;

namespace dq_nmpc_control_nodelet {
static constexpr int kStateSize = QUADROTOR_NX;
static constexpr int kSamples = QUADROTOR_N;
static constexpr int kInputSize = QUADROTOR_NU;
static constexpr int yRefSize = QUADROTOR_NX + QUADROTOR_NU;

struct solver_output {
    // The Eigen Maps initialized in the class can directly change these values below
    // without worrying about transforming between matrices and arrays
    // the relevant sections of the arrays can then be passed to the solver
    double status, KKT_res, cpu_time;
    double u0[NU];
    double u1[NU];
    double x1[NX];
    double x2[NX];
    double x4[NX];
    double xi[NU];
    double ui[NU];
    double u_out[NU * (N)];
    double x_out[NX * (N)];
};

struct solver_input {
    double x0[NX];
    double x[NX * (N)];
    double u[NU * N];
    double yref[(NX + NU) * N];
    double yref_e[(NX + NU)];
    double W[NY * NY];
    double WN[NX * NX];
};

// PLEASE DO NOT MOVE THESE ANYWHERE
// THEY BELONG HERE
// ELSE, EXPECT RANDOM PROBLEMS
extern solver_input acados_in;
extern solver_output acados_out;

class NMPCWrapper {
public:
    NMPCWrapper();
    NMPCWrapper(const Eigen::VectorXd Q_, const Eigen::VectorXd R_, const Eigen::VectorXd lbu_,
                const Eigen::VectorXd ubu_);

    bool prepare(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state);
    bool update(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state);

    void getStates(Eigen::Matrix<double, kStateSize, kSamples> &return_state);
    void getInputs(Eigen::Matrix<double, kInputSize, kSamples> &return_input);

    void setTrajectory(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, kSamples>> states,
                       const Eigen::Ref<const Eigen::Matrix<double, kInputSize, kSamples>> inputs);
    void setMass(double mass);
    void setGravity(double gravity);
    void setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e, std::vector<double> R);

    void initStates();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    quadrotor_solver_capsule *acados_ocp_capsule;
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;
    double *new_time_steps;
    int status;
    double mass_;
    double gravity_;
    bool acados_is_prepared_{false};
    int acados_status;
    double *initial_state;
    Eigen::Map<Eigen::Matrix<double, yRefSize, kSamples, Eigen::ColMajor>> acados_reference_states_{
        acados_in.yref};
    Eigen::Map<Eigen::Matrix<double, kStateSize, 1, Eigen::ColMajor>> acados_initial_state_{acados_in.x0};
    Eigen::Map<Eigen::Matrix<double, yRefSize, 1, Eigen::ColMajor>> acados_reference_end_state_{acados_in.yref_e};
    Eigen::Map<Eigen::Matrix<double, kStateSize, kSamples, Eigen::ColMajor>> acados_states_in_{acados_in.x};
    Eigen::Map<Eigen::Matrix<double, kInputSize, kSamples, Eigen::ColMajor>> acados_inputs_in_{acados_in.u};
    Eigen::Map<Eigen::Matrix<double, kStateSize, kSamples, Eigen::ColMajor>> acados_states_{acados_out.x_out};
    Eigen::Map<Eigen::Matrix<double, kInputSize, kSamples, Eigen::ColMajor>> acados_inputs_{acados_out.u_out};
    Eigen::Matrix<real_t, kInputSize, 1> kHoverInput_ =
        // (Eigen::Matrix<real_t, kInputSize, 1>() << mass_*gravity_, 0.0, 0.0, 0.0).finished();
        (Eigen::Matrix<real_t, kInputSize, 1>() << mass_*gravity_, 0.0, 0.0, 0.0).finished();
};

}  // namespace nmpc_control_nodelet