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
/** \file
 *
 *  Implementation of nsquare.hpp.
 */

#include "nsquare.hpp"
#include "cells.hpp"
#include "ghosts.hpp"

#include <boost/mpi/collectives/all_to_all.hpp>
#include <boost/serialization/vector.hpp>

#include <cassert>
#include <limits>

class AtomDecomposition {
  boost::mpi::communicator comm;
  std::vector<Cell> cells;

  std::vector<Cell *> m_local_cells;
  std::vector<Cell *> m_ghost_cells;

  GhostCommunicator m_exchange_ghosts_comm;
  GhostCommunicator m_collect_ghost_force_comm;

public:
  AtomDecomposition() = default;
  explicit AtomDecomposition(boost::mpi::communicator const &comm)
      : comm(comm), cells(comm.size()) {
    /* create communicators */
    configure_comms();
    /* configure neighbor relations */
    configure_neighbors();
    /* fill local and ghost cell lists */
    mark_cells();
  }

  void exchange_particles(bool global_flag, ParticleList &displaced_parts,
                          std::vector<Cell *> &modified_cells) {
    /* Local updates are a NoOp for this decomposition. */
    if (not global_flag) {
      assert(displaced_parts.empty());
      return;
    }

    /* Sort displaced particles by the node they belong to. */
    std::vector<std::vector<Particle>> send_buf(comm.size());
    for (auto &p : displaced_parts) {
      auto const target_node = id_to_rank(p.identity());
      send_buf.at(target_node).emplace_back(std::move(p));
    }
    displaced_parts.clear();

    /* Exchange particles */
    std::vector<std::vector<Particle>> recv_buf(comm.size());
    boost::mpi::all_to_all(comm, send_buf, recv_buf);

    if (std::any_of(recv_buf.begin(), recv_buf.end(),
                    [](auto const &buf) { return not buf.empty(); })) {
      modified_cells.push_back(std::addressof(local()));
    }

    /* Add new particles belonging to this node */
    for (auto &parts : recv_buf) {
      for (auto &p : parts) {
        local().particles().insert(std::move(p));
      }
    }
  }

  GhostCommunicator const &exchange_ghosts_comm() const {
    return m_exchange_ghosts_comm;
  }
  GhostCommunicator const &collect_ghost_force_comm() const {
    return m_collect_ghost_force_comm;
  };

  std::vector<Cell *> const &local_cells() const { return m_local_cells; }
  std::vector<Cell *> const &ghost_cells() const { return m_ghost_cells; }

  Cell *particle_to_cell(Particle const &p) { return id_to_cell(p.identity()); }

  Utils::Vector3d max_range() const {
    return Utils::Vector3d::broadcast(std::numeric_limits<double>::infinity());
  }

private:
  /**
   * @brief Determine which cell a particle id belongs to.
   *
   * Since there is only one cell this is trivial.
   *
   * @param id Particle id to find cell for.
   * @return Pointer to cell or nullptr if not local.
   */
  Cell *id_to_cell(int id) {
    return (id_to_rank(id) == comm.rank()) ? std::addressof(local()) : nullptr;
  }

  /**
   * @brief Get the local cell.
   */
  Cell &local() { return cells.at(comm.rank()); }

  void configure_neighbors() {
    std::vector<Cell *> red_neighbors;
    std::vector<Cell *> black_neighbors;

    /* distribute force calculation work  */
    for (int n = 0; n < comm.size(); n++) {
      auto const diff = n - comm.rank();
      /* simple load balancing formula. Basically diff % n, where n >=
         ad.comm.size(), n odd. The node itself is also left out, as it is
         treated differently */
      if (diff == 0) {
        continue;
      }

      if (((diff > 0 && (diff % 2) == 0) || (diff < 0 && ((-diff) % 2) == 1))) {
        red_neighbors.push_back(&cells.at(n));
      } else {
        black_neighbors.push_back(&cells.at(n));
      }
    }

    local().m_neighbors = Neighbors<Cell *>(red_neighbors, black_neighbors);
  }

  GhostCommunicator prepare_comm() {
    /* no need for comm for only 1 node */
    if (comm.size() == 1) {
      return GhostCommunicator{comm, 0};
    }

    auto ghost_comm = GhostCommunicator{comm, static_cast<size_t>(comm.size())};
    /* every node has its dedicated comm step */
    for (int n = 0; n < comm.size(); n++) {
      ghost_comm.communications[n].part_lists.resize(1);
      ghost_comm.communications[n].part_lists[0] = &(cells.at(n).particles());
      ghost_comm.communications[n].node = n;
    }

    return ghost_comm;
  }

  /**
   * @brief Setup ghost communicators.
   */
  void configure_comms() {
    m_exchange_ghosts_comm = prepare_comm();
    m_collect_ghost_force_comm = prepare_comm();

    if (comm.size() > 1) {
      for (int n = 0; n < comm.size(); n++) {
        /* use the prefetched send buffers. Node 0 transmits first and never
         * prefetches. */
        if (comm.rank() == 0 || comm.rank() != n) {
          m_exchange_ghosts_comm.communications[n].type = GHOST_BCST;
        } else {
          m_exchange_ghosts_comm.communications[n].type =
              GHOST_BCST | GHOST_PREFETCH;
        }
        m_collect_ghost_force_comm.communications[n].type = GHOST_RDCE;
      }
      /* first round: all nodes except the first one prefetch their send data */
      if (comm.rank() != 0) {
        m_exchange_ghosts_comm.communications[0].type |= GHOST_PREFETCH;
      }
    }
  }

  void mark_cells() {
    m_local_cells.resize(1, std::addressof(local()));
    m_ghost_cells.clear();
    for (int n = 0; n < comm.size(); n++) {
      if (n != comm.rank()) {
        m_ghost_cells.push_back(std::addressof(cells.at(n)));
      }
    }
  }

  /**
   * @brief Determine which rank owns a particle id.
   */
  int id_to_rank(int id) const { return id % comm.size(); }
};

AtomDecomposition ad;

void nsq_topology_init(const boost::mpi::communicator &comm) {
  ad = AtomDecomposition(comm);

  cell_structure.m_local_cells = ad.local_cells();
  cell_structure.m_ghost_cells = ad.ghost_cells();

  cell_structure.type = CELL_STRUCTURE_NSQUARE;
  cell_structure.particle_to_cell = [](const Particle &p) {
    return ad.particle_to_cell(p);
  };

  cell_structure.max_range = ad.max_range();
  cell_structure.exchange_ghosts_comm = ad.exchange_ghosts_comm();
  cell_structure.collect_ghost_force_comm = ad.collect_ghost_force_comm();
}

void nsq_exchange_particles(int global_flag, ParticleList *displaced_parts,
                            std::vector<Cell *> &modified_cells) {
  assert(displaced_parts);
  ad.exchange_particles(global_flag, *displaced_parts, modified_cells);
}
