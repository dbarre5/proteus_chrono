import numpy as np
from proteus import Domain, Context
from proteus.mprans import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
from proteus import WaveTools as wt
import py2gmsh


# dependencies for FSI
from proteus.mbd import CouplingFSI as fsi
import pychrono
import pychrono.fea as fea

opts= Context.Options([
    ("final_time",100.0,"Final time for simulation"),
    ("dt_output",0.1,"Time interval to output solution"),
    ("cfl",0.5,"Desired CFL restriction"),
    ("he",0.7,"he relative to Length of domain in x"),
    ("he_aw_interface",1.5,"mesh size of a-w interface"),
    ("sections",2,"number of sections"),
    ("section_gap",.5,"spacing between sections"),
    ("section_length",24.0,"length of section"),
    ("trident", False, "Trident"),
    ("waves", False, "wave cond")
    ])

# general options
# sim time
T = opts.final_time
# initial step
dt_init = 0.001
# CFL value
cfl = 0.5
# mesh size
he = opts.he
# rate at which values are recorded
sampleRate = 0.05
# for ALE formulation
movingDomain = True
# for added mass stabilization
addedMass = True

# physical options
# water density
rho_0 = 998.2
# water kinematic viscosity
nu_0 = 1.004e-6
# air density
rho_1 = 1.205
# air kinematic viscosity
nu_1 = 1.5e-5
# gravitational acceleration
g = np.array([0., -9.81, 0.])

# body options
fixed = False

# wave options
water_level = 6.0
wave_period = 2.75
wave_height = 0.7
wave_direction = np.array([1., 0., 0.])
wave_type = 'Fenton'  #'Linear'
# number of Fourier coefficients
Nf = 15
wave = wt.MonochromaticWaves(period=wave_period,
                             waveHeight=wave_height,
                             mwl=water_level,
                             depth=water_level,
                             g=g,
                             waveDir=wave_direction,
                             waveType=wave_type,
                             Nf=15)
wavelength = wave.wavelength



#  ____                        _
# |  _ \  ___  _ __ ___   __ _(_)_ __
# | | | |/ _ \| '_ ` _ \ / _` | | '_ \
# | |_| | (_) | | | | | | (_| | | | | |
# |____/ \___/|_| |_| |_|\__,_|_|_| |_|
# Domain
# All geometrical options go here (but not mesh options)

domain = Domain.PlanarStraightLineGraphDomain()

# ----- SHAPES ----- #

if opts.trident:
    section_count = opts.sections+2
else:
    section_count = opts.sections

# TANK
tank = st.Tank2D(domain, dim=(3*wavelength+section_count*(opts.section_length+opts.section_gap), 2*water_level))

# SPONGE LAYERS
# generation zone: 1 wavelength
# absorption zone: 2 wavelengths
tank.setSponge(x_n=wavelength, x_p=2*wavelength)


# Set the domain dictionary
c_region_dictionary = {'water': 0, 'object': 1}


### Space the main stem sections within the domain ##
for i in range(0, section_count):
    # Open the STL file
    o_geom_section = st.Rectangle(domain, dim=(opts.section_length, 1.5), coords=(0., 0.))
    # Set the regions in the STL object
    o_geom_section.setRegions([[0., 0.]], [c_region_dictionary['object']])  # This is targeted to the initial COG
    o_geom_section.setHoles([[0.0, 0.0]])
    o_geom_section.holes_ind = [0]
    # Translate the section to be internal to the tank. If this is not done the rotation fails.
    o_geom_section.translate([3*wavelength+i*(opts.section_gap+opts.section_length),water_level+0.31])

    # Set the center of mass for the object 
    o_geom_section.setBarycenter([3*wavelength+i*(opts.section_gap+opts.section_length),water_level+0.31])
    # Set the section into the tank as a child object
    tank.setChildShape(o_geom_section)

    # set no-slip conditions on caisson
    for tag, bc in o_geom_section.BC.items():
        bc.setNoSlip()
### Space the main stem sections within the domain ##

## Open the STL file
#o_geom_section = st.Rectangle(domain, dim=(opts.section_length, 1.5), coords=(0., 0.))
## Set the regions in the STL object
#o_geom_section.setRegions([[0., 0.]], [c_region_dictionary['object']])  # This is targeted to the initial COG
#o_geom_section.setHoles([[0.0, 0.0]])
#o_geom_section.holes_ind = [0]
## Translate the section to be internal to the tank. If this is not done the rotation fails.
#o_geom_section.translate([3*wavelength+0*(opts.section_gap+opts.section_length),water_level+0.31])
#
## Set the center of mass for the object 
#o_geom_section.setBarycenter([3*wavelength+0*(opts.section_gap+opts.section_length),water_level+0.31])
## Set the section into the tank as a child object
#tank.setChildShape(o_geom_section)
#
## set no-slip conditions on caisson
#for tag, bc in o_geom_section.BC.items():
#    bc.setNoSlip()
#

#   ____ _
#  / ___| |__  _ __ ___  _ __   ___
# | |   | '_ \| '__/ _ \| '_ \ / _ \
# | |___| | | | | | (_) | | | | (_) |
#  \____|_| |_|_|  \___/|_| |_|\___/
# Chrono

# SYSTEM

# create system
system = fsi.ProtChSystem()
# access chrono object
chsystem = system.getChronoObject()
# communicate gravity to system
# can also be set with:
# system.ChSystem.Set_G_acc(pychrono.ChVectorD(g[0], g[1], g[2]))
system.setGravitationalAcceleration(g)
# set maximum time step for system
system.setTimeStep(1e-4)

solver = pychrono.ChSolverMINRES()
solver.SetVerbose(True)
#solver.EnableWarmStart(True)
chsystem.SetSolver(solver)
system.ChSystem.SetTimestepperType(pychrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
#system.ChSystem.SetSolverTolerance(0.00001)
system.ChSystem.SetSolverMaxIterations(2000)
system.ChSystem.SetMaxiter(2000)
# create floating bodies

for i in range(0, opts.sections):
    body = fsi.ProtChBody(system=system)
    # attach shape: this automatically adds a body at the barycenter of the caisson shape
    body.attachShape(domain.shape_list[i + 1])
    # set 2D width (for force calculation)
    body.setWidth2D(7.0)
    # access chrono object
    chbody = body.getChronoObject()
    # impose constraints
    chbody.SetBodyFixed(fixed)
    free_x = np.array([1., 1., 1.]) # translational
    free_r = np.array([1., 1., 1.]) # rotational
    body.setConstraints(free_x=free_x, free_r=free_r)
    # access pychrono ChBody
    # set mass
    # can also be set with:
    # body.ChBody.SetMass(14.5)
    body.setMass(65709.)
    # set inertia
    # can also be set with:
    # body.ChBody.setInertiaXX(pychrono.ChVectorD(1., 1., 0.35))
    body.setInertiaXX(np.array([302568.55, 3296380.13, 3026207.97]))
    # record values
    body.setRecordValues(all_values=True)

system.subcomponents[0].ChBody.SetBodyFixed(True) # Fix last section

cylindrical = False
lock = False
revolute = False

d_guillotine_mass = 0.005                  # [kg]
d_guillotine_Ixx = 0.000175                # [kg/m^2]
d_guillotine_Izz = 0.000155                # [kg/m^2]
d_guillotine_Iyy = 0.00010                # [kg/m^2]

d_pin_mass = 0.005                  # [kg]
d_pin_Ixx = 0.000175                # [kg/m^2]
d_pin_Izz = 0.000155                # [kg/m^2]
d_pin_Iyy = 0.00010                # [kg/m^2]

d_bar_mass = 0.005                  # [kg]
d_bar_Ixx = 0.000175                # [kg/m^2]
d_bar_Izz = 0.000155                # [kg/m^2]
d_bar_Iyy = 0.00010                # [kg/m^2]

o_inertia_guillotine = pychrono.ChVectorD(d_guillotine_Ixx, d_guillotine_Iyy, d_guillotine_Izz)
o_inertia_pin = pychrono.ChVectorD(d_pin_Ixx, d_pin_Iyy, d_pin_Izz)
o_inertia_bar = pychrono.ChVectorD(d_bar_Ixx, d_bar_Iyy, d_bar_Izz)

free_x = np.array([1., 1., 1.])  # Translational DOFs
free_r = np.array([1., 1., 1.])  # Rotational DOFs

#indice for components
system_indice = opts.sections
left_guillotines_ind = []
right_guillotines_ind = []
left_pin_ind = []
right_pin_ind = []
bar_ind = []
log_links = []



# Create a mesh, that is a container for groups
# of elements and their referenced nodes.
my_mesh = fea.ChMesh();
system.ChSystem.Add(my_mesh)
my_mesh.SetAutomaticGravity(True,2) # for max precision in gravity of FE, at least 2 integration points per element when using cubic IGA
#system.Set_G_acc(chrono.ChVectorD(0,-9.81, 0));

# widths of bar (m)
bar_width_y = 0.0635
bar_width_z = 0.0889

#Young modulus of the bar
bar_E = 205e9

# Shear modulus of the bar
bar_G = 80e9

# Density of the bar
bar_density = 7850.

# A simple specialization of ChBeamSectionEuler if you just need the simplest model for a rectangular centered beam, with uniform elasticity and uniform density.
#msection = fea.ChBeamSectionEasyRectangular(bar_width_y, bar_width_z, bar_E, bar_G, bar_density)
msection = fea.ChBeamSectionAdvanced()
msection.SetAsRectangularSection(bar_width_y,bar_width_z)
msection.SetDensity(bar_density)
msection.SetYoungModulus(bar_E)
msection.SetGshearModulus(bar_G)



builder = fea.ChBuilderBeamEuler()
builder.BuildBeam(my_mesh,      # the mesh to put the elements in
	msection,					# section of the beam
	20,							# number of sections (spans)
	pychrono.ChVectorD(3*wavelength+0*(opts.section_gap+opts.section_length)+opts.section_length/2., water_level+0.31, 0),		# start point
	pychrono.ChVectorD(3*wavelength+0*(opts.section_gap+opts.section_length)+opts.section_length/2.+opts.section_gap, water_level+0.31, 0),	# end point
	pychrono.VECT_Y)				# suggested Y direction of section


o_anchor = fsi.ProtChBody(system)
o_anchor_point = pychrono.ChVectorD((opts.section_gap+opts.section_length)+opts.section_length/2., water_level+0.31, 0)
o_anchor.ChBody.SetPos(o_anchor_point)
o_anchor.ChBody.SetBodyFixed(True)
mnodeA = builder.GetLastBeamNodes().front()
mnodeB = builder.GetLastBeamNodes().back()
constraintA = fea.ChLinkPointFrame()
hnode1 = fea.ChNodeFEAxyzrot(pychrono.ChFrameD(o_anchor_point))
for i in range(100):
    print(type(mnodeA))
    print(type(system.subcomponents[0].ChBody))
#constraintA.Initialize(hnode1, o_anchor, o_anchor_point)

constr_bc = pychrono.ChLinkMateGeneric()
constr_bc.Initialize(mnodeA, system.subcomponents[0].ChBody, False, mnodeA.Frame(), mnodeA.Frame())
system.ChSystem.Add(constr_bc)
constr_bc.SetConstrainedCoords(True, True, True,   # x, y, z
                               True, True, True)   # Rx, Ry, Rz
#mnodeB.SetForce(pychrono.ChVectorD(0, -1000, 0))
constr_bc = pychrono.ChLinkMateGeneric()
constr_bc.Initialize(mnodeB, system.subcomponents[1].ChBody, False, mnodeB.Frame(), mnodeB.Frame())
system.ChSystem.Add(constr_bc)
constr_bc.SetConstrainedCoords(True, True, True,   # x, y, z
                               True, True, True)   # Rx, Ry, Rz
nodeList = []
nodeList.append(builder.GetLastBeamNodes())

#    // Create also a truss
#    auto truss = chrono_types::make_shared<ChBody>();
#    truss->SetBodyFixed(true);
#    my_system.Add(truss);
# 
#    // Create a constraint between a node and the truss
#    auto constraintA = chrono_types::make_shared<ChLinkPointFrame>();
# 
#    constraintA->Initialize(mnodeA,  // node to connect
#                            truss);  // body to be connected to
#

#
#
#
#for i in range(0, section_count-1):
#    # Location of left box's guillotine
#    o_left_point = pychrono.ChVectorD(1*wavelength+i*(opts.section_gap+opts.section_length)+opts.section_length/2.,
#                                                                    water_level,
#                                                                    0.0)
#    # Location of right box's guillotine
#    o_right_point = pychrono.ChVectorD(1*wavelength+i*(opts.section_gap+opts.section_length)+opts.section_length/2.+opts.section_gap,
#                                                                    water_level,
#                                                                    0.0)   
#
#    o_mid_point = pychrono.ChVectorD(1*wavelength+i*(opts.section_gap+opts.section_length)+opts.section_length/2.+opts.section_gap/2.0,
#                                                                    water_level,
#                                                                    0.0)    
#
#
#                                                                                 # Shift y points to be edges of section, and z to be displaced from new top of section.
#    ## Create a section, i.e. thickness and material properties
#    ## for beams. This will be shared among some beams.
#    msection = fea.ChBeamSectionAdvanced()
#
#    beam_wy = 0.012
#    beam_wz = 0.025
#    msection.SetAsRectangularSection(beam_wy, beam_wz)
#    msection.SetYoungModulus(0.01e2)
#    msection.SetGshearModulus(0.01e2 * 0.3)
#    msection.SetBeamRaleyghDamping(0.000)
#    #msection.SetCentroid(0,0.02)
#    #msection.SetShearCenter(0,0.1)
#    #msection.SetSectionRotation(45*chrono.CH_C_RAD_TO_DEG)
#
#    # Add some EULER-BERNOULLI BEAMS:
#    beam_L = opts.section_gap
#
#    hnode1 = fea.ChNodeFEAxyzrot(pychrono.ChFrameD(o_left_point))
#    hnode2 = fea.ChNodeFEAxyzrot(pychrono.ChFrameD(o_right_point))
#
#    my_mesh.AddNode(hnode1)
#    my_mesh.AddNode(hnode2)
#
#    belement1 = fea.ChElementBeamEuler()
#
#    belement1.SetNodes(hnode1, hnode2)
#    belement1.SetSection(msection)
#
#    my_mesh.AddElement(belement1)
#
#    constr_bc = pychrono.ChLinkMateGeneric()
#    constr_bc.Initialize(hnode1, system.subcomponents[i].ChBody, False, hnode1.Frame(), hnode1.Frame())
#    system.ChSystem.Add(constr_bc)
#    constr_bc.SetConstrainedCoords(True, True, True,   # x, y, z
#                               True, True, True)   # Rx, Ry, Rz   
#
#    constr_bc = pychrono.ChLinkMateGeneric()
#    constr_bc.Initialize(hnode2, system.subcomponents[i+1].ChBody, False, hnode2.Frame(), hnode2.Frame())
#    system.ChSystem.Add(constr_bc)
#    constr_bc.SetConstrainedCoords(True, True, True,   # x, y, z
#                               True, True, True)   # Rx, Ry, Rz                              

#system.log_chrono_reactF = log_links
#system.subcomponents[section_count-1].ChBody.SetBodyFixed(True) # Fix last section


Revolute = True
if cylindrical == True: 
    for i in range(0, opts.sections-1):
            # Define the cylindrical link lock joint
            o_link_lock_section = pychrono.ChLinkLockCylindrical()

            # Define the two points give the rotation axis. This is at the pin location half way between the sections
            # Positioning is automatically calculated to the joint.
            o_first_point = pychrono.ChCoordsysD(pychrono.ChVectorD(1*wavelength+i*(opts.section_gap+opts.section_length)+opts.section_gap/2.+opts.section_length/2.,                                         # Shifting x axis placement to start where pier starts in domain.
                                                                    water_level+0.1,
                                                                    0.0))                                                                                        # Shift y points to be edges of section, and z to be displaced from new top of section.

            # Initialize the rotational constraint. This is intended to be around the
            o_link_lock_section.Initialize(system.subcomponents[i].ChBody,
                                        system.subcomponents[i+1].ChBody,
                                        o_first_point)

            # Add the joint to the system
            system.ChSystem.Add(o_link_lock_section)
if lock == True: 
    for i in range(0, opts.sections-1):
            # Define the cylindrical link lock joint
            o_fixed_lock = pychrono.ChLinkLockLock()

            # Define the two points give the rotation axis. This is at the pin location half way between the sections
            # Positioning is automatically calculated to the joint.
            o_first_point = pychrono.ChCoordsysD(pychrono.ChVectorD(1*wavelength+i*(opts.section_gap+opts.section_length)+opts.section_gap/2.+opts.section_length/2.,                                         # Shifting x axis placement to start where pier starts in domain.
                                                                    water_level+0.1,
                                                                    0.0))                                                                                        # Shift y points to be edges of section, and z to be displaced from new top of section.

            # Initialize the rotational constraint. This is intended to be around the
            o_fixed_lock.Initialize(system.subcomponents[i].ChBody,
                                    system.subcomponents[i+1].ChBody,
                                    o_first_point)

            # Add the joint to the system
            system.ChSystem.Add(o_fixed_lock)

if revolute == True: 
    for i in range(0, opts.sections-1):
            # Define the cylindrical link lock joint
            o_revolute_lock = pychrono.ChLinkLockRevolute()

            # Define the two points give the rotation axis. This is at the pin location half way between the sections
            # Positioning is automatically calculated to the joint.
            o_first_point = pychrono.ChCoordsysD(pychrono.ChVectorD(3*wavelength+i*(opts.section_gap+opts.section_length)+opts.section_gap/2.+opts.section_length/2.,water_level+0.25,0.0))                                                                                        # Shift y points to be edges of section, and z to be displaced from new top of section.

            # Initialize the rotational constraint. This is intended to be around the
            o_revolute_lock.Initialize(system.subcomponents[i].ChBody,
                                        system.subcomponents[i+1].ChBody,
                                        o_first_point)

            # Add the joint to the system
            system.ChSystem.Add(o_revolute_lock)
#  ____                        _                   ____                _ _ _   _
# | __ )  ___  _   _ _ __   __| | __ _ _ __ _   _ / ___|___  _ __   __| (_) |_(_) ___  _ __  ___
# |  _ \ / _ \| | | | '_ \ / _` |/ _` | '__| | | | |   / _ \| '_ \ / _` | | __| |/ _ \| '_ \/ __|
# | |_) | (_) | |_| | | | | (_| | (_| | |  | |_| | |__| (_) | | | | (_| | | |_| | (_) | | | \__ \
# |____/ \___/ \__,_|_| |_|\__,_|\__,_|_|   \__, |\____\___/|_| |_|\__,_|_|\__|_|\___/|_| |_|___/
#                                           |___/
# Boundary Conditions

# CAISSON






#for j in range(1000):
#    for i in range(len(nodeList[0])):
#        print(nodeList[0][i].GetPos())
# first need to initialize system
system.calculate_init()

nT = int(np.ceil(T/sampleRate))
for ii in range(nT):
    # calculate for one main step
    system.calculate(sampleRate)
    file1 = open("Beams.txt","a")
    for j in range(len(nodeList)):
        for i in range(len(nodeList[0])):
            file1.write(str(j) + " "+ str() + " " + str(nodeList[0][i].GetPos().x) + " " + str(nodeList[0][i].GetPos().y) + " " + str(nodeList[0][i].GetPos().z))
            file1.write("\n")
    file1.close()
    # print some info
    #print('n: {ii}, time: {time}'.format(ii=ii, time=system.ChSystem.GetChTime()))
    #print('tension in mooring1: {tens}'.format(tens=m1.getTensionBack()))