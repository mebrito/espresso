#
# Copyright (C) 2013,2014,2015,2016 The ESPResSo project
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

"""Testmodule for the Reaction Ensemble.
"""
import os
import sys
import unittest as ut
import numpy as np
import espressomd  # pylint: disable=import-error
from espressomd import reaction_ensemble
from espressomd import grand_canonical


@ut.skipIf('REACTION_ENSEMBLE' not in espressomd.code_info.features(),
           "REACTION_ENSEMBLE not compiled in, can not check functionality.")
class ReactionEnsembleTest(ut.TestCase):
    """Test the core implementation of writing hdf5 files."""
    
    N0=40
    type_HA=0
    type_A=1
    type_H=2
    temperature=1.0
    standard_pressure_in_simulation_units=0.00108
    K_HA_diss=0.5; #could be in this test for example anywhere in the range 0.000001 ... 9
    system = espressomd.System()
    system.box_l = [35.0, 35.0, 35.0]
    system.cell_system.skin = 0.4
    system.time_step = 0.01
    RE=reaction_ensemble.ReactionEnsemble(standard_pressure=standard_pressure_in_simulation_units, temperature=1, exclusion_radius=1)        
    
    @classmethod
    def setUpClass(cls):
        """Prepare a testsystem."""
        for i in range(0,2*cls.N0,2):
            cls.system.part.add(id=i ,pos=np.random.random(3) * cls.system.box_l, type=cls.type_A)
            cls.system.part.add(id=i+1 ,pos=np.random.random(3) * cls.system.box_l, type=cls.type_H)
        
        cls.RE.add(equilibrium_constant=cls.K_HA_diss,reactant_types=[cls.type_HA],reactant_coefficients=[1], product_types=[cls.type_A,cls.type_H], product_coefficients=[1,1])
        cls.RE.default_charges(dictionary={"0":0,"1":-1, "2":+1})
        cls.RE.print_status()

    @classmethod
    def ideal_degree_of_association(cls,pK_a,pH):
        return 1-1.0/(1+10**(pK_a-pH))

    def test_ideal_titration_curve(self):
        N0=ReactionEnsembleTest.N0
        temperature=ReactionEnsembleTest.temperature
        type_A=ReactionEnsembleTest.type_A
        type_H=ReactionEnsembleTest.type_H
        type_HA=ReactionEnsembleTest.type_HA
        box_l=ReactionEnsembleTest.system.box_l
        standard_pressure_in_simulation_units=ReactionEnsembleTest.standard_pressure_in_simulation_units
        system=ReactionEnsembleTest.system
        K_HA_diss=ReactionEnsembleTest.K_HA_diss
        RE=ReactionEnsembleTest.RE
        """ chemical warmup in order to get to chemical equilibrium before starting to calculate the observable "degree of association" """
        for i in range(10*N0):
            RE.reaction()
            
        volume=np.prod(self.system.box_l) #cuboid box
        average_NH=0.0
        average_degree_of_association=0.0
        num_samples=2000
        for i in range(num_samples):
            RE.reaction()
            average_NH+=grand_canonical.number_of_particles(current_type=type_H)
            average_degree_of_association+=grand_canonical.number_of_particles(current_type=type_HA)/float(N0)
        average_NH/=num_samples
        average_degree_of_association/=num_samples
        pH=-np.log10(average_NH/volume)
        K_apparent_HA_diss=K_HA_diss*standard_pressure_in_simulation_units/temperature
        pK_a=-np.log10(K_apparent_HA_diss)
        real_error_in_degree_of_association=abs(average_degree_of_association-ReactionEnsembleTest.ideal_degree_of_association(pK_a,pH))/ReactionEnsembleTest.ideal_degree_of_association(pK_a,pH)
        self.assertTrue(real_error_in_degree_of_association<0.07, msg="Deviation to ideal titration curve for the given input parameters too large.")
    
if __name__ == "__main__":
    suite = ut.TestLoader().loadTestsFromTestCase(ReactionEnsembleTest)
    result=ut.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
