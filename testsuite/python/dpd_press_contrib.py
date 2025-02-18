#
# Copyright (C) 2013-2025 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
import unittest as ut
import unittest_decorators as utx
import espressomd

@ut.skipIf(espressomd.has_features(["DPD"]), "check default dictionaries")
class NoDPDPressContrib(ut.TestCase):
    system = espressomd.System(box_l=3 * [10])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    n_part = 100

    np.random.seed(seed=42)
    system.part.add(pos=system.box_l * np.random.random((n_part, 3)))

    def test_nondpd_case(self):
        self.assertFalse(espressomd.has_features(["DPD"]))
        energy_dict = self.system.analysis.energy()
        self.assertNotIn("dpd", energy_dict, "'dpd' key should not be in the energy dictionary")

        pressure_dict = self.system.analysis.pressure()
        self.assertNotIn("dpd", pressure_dict, "'dpd' key should not be in the pressure dictionary")

        pressure_tensor_dict = self.system.analysis.pressure_tensor()
        self.assertNotIn("dpd", pressure_tensor_dict, "'dpd' key should not be in the pressure-tensor dictionary")


@utx.skipIfMissingFeatures("DPD")
class DPDPressContrib(ut.TestCase):
    system = espressomd.System(box_l=3 * [10])
    system.time_step = 0.01
    system.cell_system.skin = 0.4

    np.random.seed(seed=42)

    def test_dpd_case(self):
        self.assertTrue(espressomd.has_features(["DPD"]))
        
        n_part = 100
        kT = 1.
        gamma = 1.5
        r_cut = 1.
        # Repulsive parameter
        F_max = 1.

        # Add particles
        self.system.part.add(pos=self.system.box_l * np.random.random((n_part, 3)))
        
        # Activate the thermostat
        self.system.thermostat.set_dpd(kT=kT, seed=123)
        np.random.seed(seed=42)

        # Set up the DPD friction interaction
        self.system.non_bonded_inter[0, 0].dpd.set_params(
            weight_function=0, gamma=gamma, r_cut=r_cut,
            trans_weight_function=0, trans_gamma=gamma, trans_r_cut=r_cut)

        # Set up the repulsive interaction
        self.system.non_bonded_inter[0, 0].hat.set_params(F_max=F_max, cutoff=r_cut)

        # Integration
        self.system.integrator.run(1)

        energy_dict = self.system.analysis.energy()
        self.assertNotIn("dpd", energy_dict, "'dpd' key should not be in the energy dictionary")

        pressure_dict = self.system.analysis.pressure()
        self.assertIn("dpd", pressure_dict, "'dpd' key is not in the pressure dictionary")

        pressure_tensor_dict = self.system.analysis.pressure_tensor()
        self.assertIn("dpd", pressure_tensor_dict, "'dpd' key is not in the pressure-tensor dictionary")


if __name__ == "__main__":
    ut.main()
