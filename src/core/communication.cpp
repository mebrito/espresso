/*
 * Copyright (C) 2010-2019 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#ifdef OPEN_MPI
#include <dlfcn.h>
#endif
#include <cassert>

#include "communication.hpp"

#include "errorhandling.hpp"

#include "event.hpp"
#include "global.hpp"
#include "grid.hpp"
#include "grid_based_algorithms/lb.hpp"
#include "grid_based_algorithms/lb_interface.hpp"

#include <boost/mpi.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/string.hpp>
#include <utils/mpi/cart_comm.hpp>

using namespace std;

namespace Communication {
auto const &mpi_datatype_cache = boost::mpi::detail::mpi_datatype_cache();
std::shared_ptr<boost::mpi::environment> mpi_env;
} // namespace Communication

boost::mpi::communicator comm_cart;

namespace Communication {
std::unique_ptr<MpiCallbacks> m_callbacks;

/* We use a singleton callback class for now. */
MpiCallbacks &mpiCallbacks() {
  assert(m_callbacks && "Mpi not initialized!");

  return *m_callbacks;
}
} // namespace Communication

using Communication::mpiCallbacks;

int this_node = -1;
int n_nodes = -1;

// if you want to add a callback, add it here, and here only
#define CALLBACK_LIST                                                          \
  CB(mpi_gather_stats_slave)                                                   \

// create the forward declarations
#define CB(name) void name(int node, int param);
#ifndef DOXYGEN
/* this conditional on DOXYGEN prevents an interaction in Doxygen between
 * CALLBACK_LIST and whatever follows next, e.g. a function "int foo();"
 * would otherwise become "CALLBACK_LIST int foo();" */
CALLBACK_LIST
#endif

#undef CB

/**********************************************
 * procedures
 **********************************************/

#if defined(OPEN_MPI)
namespace {
/** Workaround for "Read -1, expected XXXXXXX, errno = 14" that sometimes
 *  appears when CUDA is used. This is a bug in OpenMPI 2.0-2.1.2 and 3.0.0
 *  according to
 *  https://www.mail-archive.com/users@lists.open-mpi.org/msg32357.html,
 *  so we set btl_vader_single_copy_mechanism = none.
 */
void openmpi_fix_vader() {
  if ((OMPI_MAJOR_VERSION == 2 && OMPI_MINOR_VERSION == 1 &&
       OMPI_RELEASE_VERSION < 3) or
      (OMPI_MAJOR_VERSION == 3 && OMPI_MINOR_VERSION == 0 &&
       OMPI_RELEASE_VERSION == 0)) {
    setenv("OMPI_MCA_btl_vader_single_copy_mechanism", "none", 0);
  }
}

/**
 * @brief Assure that openmpi is loaded to the global namespace.
 *
 * This was originally inspired by mpi4py
 * (https://github.com/mpi4py/mpi4py/blob/4e3f47b6691c8f5a038e73f84b8d43b03f16627f/src/lib-mpi/compat/openmpi.h).
 * It's needed because OpenMPI dlopens its submodules. These are unable to find
 * the top-level OpenMPI library if that was dlopened itself, i.e. when the
 * Python interpreter dlopens a module that is linked against OpenMPI. It's
 * about some weird two-level symbol namespace thing.
 */
void openmpi_global_namespace() {
  if (OMPI_MAJOR_VERSION >= 3)
    return;
#ifdef RTLD_NOLOAD
  const int mode = RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD;
#else
  const int mode = RTLD_NOW | RTLD_GLOBAL;
#endif

  const void *_openmpi_symbol = dlsym(RTLD_DEFAULT, "MPI_Init");
  if (!_openmpi_symbol) {
    fprintf(stderr, "Aborting because unable to find OpenMPI symbol.\n");
    errexit();
  }

  Dl_info _openmpi_info;
  dladdr(_openmpi_symbol, &_openmpi_info);

  const void *handle = dlopen(_openmpi_info.dli_fname, mode);

  if (!handle) {
    fprintf(stderr, "Aborting because unable to load libmpi into the "
                    "global symbol space.\n");
    errexit();
  }
}
} // namespace
#endif

namespace Communication {
void init(std::shared_ptr<boost::mpi::environment> mpi_env) {
  Communication::mpi_env = std::move(mpi_env);

  MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
  node_grid = Utils::Mpi::dims_create<3>(n_nodes);

  comm_cart =
      Utils::Mpi::cart_create(comm_cart, node_grid, /* reorder */ false);

  this_node = comm_cart.rank();

  Communication::m_callbacks =
      std::make_unique<Communication::MpiCallbacks>(comm_cart);

#define CB(name) Communication::m_callbacks->add(&(name));
  CALLBACK_LIST
#undef CB

  ErrorHandling::init_error_handling(mpiCallbacks());

  on_program_start();
}
} // namespace Communication

std::shared_ptr<boost::mpi::environment> mpi_init() {
#ifdef OPEN_MPI
  openmpi_fix_vader();
  openmpi_global_namespace();
#endif

  return std::make_shared<boost::mpi::environment>();
}

/*************** GATHER ************/
void mpi_gather_stats(GatherStats job, double *result) {
  auto job_slave = static_cast<int>(job);
  switch (job) {
  case GatherStats::lb_fluid_momentum:
    mpi_call(mpi_gather_stats_slave, -1, job_slave);
    lb_calc_fluid_momentum(result, lbpar, lbfields, lblattice);
    break;
#ifdef LB_BOUNDARIES
  case GatherStats::lb_boundary_forces:
    mpi_call(mpi_gather_stats_slave, -1, job_slave);
    lb_collect_boundary_forces(result);
    break;
#endif
  default:
    fprintf(
        stderr,
        "%d: INTERNAL ERROR: illegal request %d for mpi_gather_stats_slave\n",
        this_node, job_slave);
    errexit();
  }
}

void mpi_gather_stats_slave(int, int job_slave) {
  auto job = static_cast<GatherStats>(job_slave);
  switch (job) {
  case GatherStats::lb_fluid_momentum:
    lb_calc_fluid_momentum(nullptr, lbpar, lbfields, lblattice);
    break;
#ifdef LB_BOUNDARIES
  case GatherStats::lb_boundary_forces:
    lb_collect_boundary_forces(nullptr);
    break;
#endif
  default:
    fprintf(
        stderr,
        "%d: INTERNAL ERROR: illegal request %d for mpi_gather_stats_slave\n",
        this_node, job_slave);
    errexit();
  }
}

/*********************** MAIN LOOP for slaves ****************/

void mpi_loop() {
  if (this_node != 0)
    mpiCallbacks().loop();
}
