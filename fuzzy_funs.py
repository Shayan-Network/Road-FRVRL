import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
########################dirction##################
# Define the fuzzy system with input rules as indices (0, 1, 2, etc.)

def fuzzy_direction(s, R):
    # Define fuzzy variables
    density = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'density')
    velocity = ctrl.Antecedent(np.arange(0, 151, 1), 'velocity')
    direction = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'direction')

    # Define membership functions for 'density'
    density_mf = [
        fuzz.trapmf(density.universe, [0, 0, 0.1, 0.1]),  # vs
        fuzz.trimf(density.universe, [0.09, 0.25, 0.5]),  # s
        fuzz.trimf(density.universe, [0.25, 0.5, 0.75]),  # m
        fuzz.trimf(density.universe, [0.5, 0.75, 1]),     # l
        fuzz.trapmf(density.universe, [0.75, 1, 1.5, 2])  # vl
    ]

    # Assign membership functions to 'density'
    density['vs'], density['s'], density['m'], density['l'], density['vl'] = density_mf

    # Define membership functions for 'velocity'
    velocity_mf = [
        fuzz.trapmf(velocity.universe, [0, 0, 12, 22.5]),  # vs
        fuzz.trimf(velocity.universe, [0, 22.5, 45]),      # s
        fuzz.trimf(velocity.universe, [22.5, 45, 67.5]),   # m
        fuzz.trimf(velocity.universe, [45, 67.5, 90]),     # l
        fuzz.trapmf(velocity.universe, [67.5, 80, 90, 150])# vl
    ]

    # Assign membership functions to 'velocity'
    velocity['vs'], velocity['s'], velocity['m'], velocity['l'], velocity['vl'] = velocity_mf

    # Define membership functions for 'direction'
    direction_mf = [
        fuzz.trapmf(direction.universe, [0, 0, 0.5, 0.5]),  # yes
        fuzz.trapmf(direction.universe, [0.5, 0.5, 1, 1])   # no
    ]

    direction['yes'], direction['no'] = direction_mf

    # Create rule list based on input rules as indices
    rule_list = []

    # Map the indices (0, 1, 2, etc.) to their corresponding membership function labels
    density_terms = ['vs', 's', 'm', 'l', 'vl']
    velocity_terms = ['vs', 's', 'm', 'l', 'vl']
    direction_term =['yes','no']
    # Iterate through the rules matrix R
    for i in range(len(R)):
        rule = ctrl.Rule(density[density_terms[R[i][0]]] & velocity[velocity_terms[R[i][1]]], direction[direction_term[R[i][2]]])
        rule_list.append(rule)

    # Add rules to the system
    direction_ctrl = ctrl.ControlSystem(rule_list)
    direction_simulation = ctrl.ControlSystemSimulation(direction_ctrl)
    maxdencity=2
    if s[0]>maxdencity:
        s[0]=maxdencity

    # Set inputs to the fuzzy system
    direction_simulation.input['density'] = s[0]
    direction_simulation.input['velocity'] = s[1]

    # Compute the fuzzy output
    direction_simulation.compute()

    try:
        direction_simulation.compute()  # Try computing the output
        fuzzy_output = direction_simulation.output['direction']
    except KeyError:
        fuzzy_output = .5
        print("direction")
        
    return fuzzy_output
###########light###################################################################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Precompute membership functions
queu_mf = {
    'vs': fuzz.trapmf(np.arange(0, 201, 1), [0, 0, 10/3, 20/3]),
    's': fuzz.trimf(np.arange(0, 201, 1), [0, 30/3, 40/3]),
    'm': fuzz.trimf(np.arange(0, 201, 1), [30/3, 60/3, 90/3]),
    'l': fuzz.trimf(np.arange(0, 201, 1), [60/3, 90/3, 120/3]),
    'vl': fuzz.trapmf(np.arange(0, 201, 1), [90/3, 100/3, 120/3, 200])
}

arival_mf = {
    'vs': fuzz.trapmf(np.arange(0, 201, 1), [0, 0, 10/3, 20/3]),
    's': fuzz.trimf(np.arange(0, 201, 1), [0, 30/3, 40/3]),
    'm': fuzz.trimf(np.arange(0, 201, 1), [30/3, 60/3, 90/3]),
    'l': fuzz.trimf(np.arange(0, 201, 1), [60/3, 90/3, 120/3]),
    'vl': fuzz.trapmf(np.arange(0, 201, 1), [90/3, 100/3, 120/3, 200])
}

extend_mf = {
    'z': fuzz.trapmf(np.arange(0, 51, 1), [0, 0, 7, 15]),
    '0s': fuzz.trimf(np.arange(0, 51, 1), [0, 15, 20]),
    '0m': fuzz.trimf(np.arange(0, 51, 1), [15, 20, 30]),
    '0vl': fuzz.trapmf(np.arange(0, 51, 1), [20, 30, 50, 50])
}

# Define fuzzy variables
queu = ctrl.Antecedent(np.arange(0, 201, 1), 'queu')
arival = ctrl.Antecedent(np.arange(0, 201, 1), 'arival')
extend = ctrl.Consequent(np.arange(0, 51, 1), 'extend')

# Assign precomputed membership functions
for term, mf in queu_mf.items():
    queu[term] = mf
for term, mf in arival_mf.items():
    arival[term] = mf
for term, mf in extend_mf.items():
    extend[term] = mf

# Create control system and simulation only once
def fuzzy_light(s, R):
    # Create rule list based on input rules as indices
    rule_list = []

    # Map the indices to their corresponding membership function labels
    queu_terms = ['vs', 's', 'm', 'l', 'vl']
    arival_terms = ['vs', 's', 'm', 'l', 'vl']
    extend_terms = ['z', '0s', '0m', '0vl']

    # Iterate through the rules matrix R
    for i in range(len(R)):
        rule = ctrl.Rule(queu[queu_terms[R[i][0]]] & arival[arival_terms[R[i][1]]], extend[extend_terms[R[i][2]]])
        rule_list.append(rule)

    # Add rules to the system
    extend_ctrl = ctrl.ControlSystem(rule_list)
    extend_simulation = ctrl.ControlSystemSimulation(extend_ctrl)

    # Clip input values to the valid range
    s[0] = np.clip(s[0], 0, 200)  # Ensure queu is within [0, 200]
    s[1] = np.clip(s[1], 0, 200)  # Ensure arival is within [0, 200]

    # Set inputs to the fuzzy system
    extend_simulation.input['queu'] = s[0]
    extend_simulation.input['arival'] = s[1]

    # Compute the fuzzy output
    try:
        extend_simulation.compute()  # Try computing the output
        fuzzy_output = extend_simulation.output['extend']
    except KeyError:
        fuzzy_output = 5  # Default value if computation fails

    return fuzzy_output

##########################noe###############################################
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Precompute membership functions
density_mf = [
    fuzz.trapmf(np.arange(0, 2.01, 0.01), [0, 0, 0.1, 0.1]),   # vs
    fuzz.trimf(np.arange(0, 2.01, 0.01), [0.09, 0.25, 0.5]),   # s
    fuzz.trimf(np.arange(0, 2.01, 0.01), [0.25, 0.5, 0.75]),   # m
    fuzz.trimf(np.arange(0, 2.01, 0.01), [0.5, 0.75, 1]),      # l
    fuzz.trapmf(np.arange(0, 2.01, 0.01), [0.75, 1, 1.5, 2])   # vl
]
distance_mf = [
    fuzz.trapmf(np.arange(0, 1001, 1), [0, 0, 300, 450]),  # vs
    fuzz.trimf(np.arange(0, 1001, 1), [300, 450, 600]),    # l
    fuzz.trapmf(np.arange(0, 1001, 1), [600, 800, 1000, 1000])  # vl
]
avel_mf = [
    fuzz.trapmf(np.arange(0, 151, 1), [0, 0, 12, 22.5]),  # vs
    fuzz.trimf(np.arange(0, 151, 1), [0, 22.5, 45]),      # s
    fuzz.trimf(np.arange(0, 151, 1), [22.5, 45, 67.5]),   # m
    fuzz.trimf(np.arange(0, 151, 1), [45, 67.5, 90]),     # l
    fuzz.trapmf(np.arange(0, 151, 1), [67.5, 80, 90, 150]) # vl
]
noe_mf = [
    fuzz.trapmf(np.arange(0, 1.01, 0.01), [0, 0, 0.5, 0.5]),  # no
    fuzz.trapmf(np.arange(0, 1.01, 0.01), [0.5, 0.5, 1, 1])   # yes
]

# Define fuzzy variables
density = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'density')
distance = ctrl.Antecedent(np.arange(0, 1001, 1), 'distance')
avel = ctrl.Antecedent(np.arange(0, 151, 1), 'avel')
noe = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'noe')

# Assign precomputed membership functions
density['vs'], density['s'], density['m'], density['l'], density['vl'] = density_mf
distance['vs'], distance['l'], distance['vl'] = distance_mf
avel['vs'], avel['s'], avel['m'], avel['l'], avel['vl'] = avel_mf
noe['no'], noe['yes'] = noe_mf

# Create control system and simulation only once
def fuzzy_noe(s, R):
    # Create rule list based on input rules as indices
    rule_list = []

    # Map the indices (0, 1, 2, etc.) to their corresponding membership function labels
    density_terms = ['vs', 's', 'm', 'l', 'vl']
    distance_terms = ['vs', 'l', 'vl']
    avel_terms = ['vs', 's', 'm', 'l', 'vl']
    noe_terms = ['no', 'yes']

    # Iterate through the rules matrix R
    for i in range(len(R)):
        rule = ctrl.Rule(density[density_terms[int(R[i][0])]] &
                         distance[distance_terms[int(R[i][1])]] &
                         avel[avel_terms[int(R[i][2])]],
                         noe[noe_terms[int(R[i][3])]])
        rule_list.append(rule)

    # Add rules to the system
    noe_ctrl = ctrl.ControlSystem(rule_list)
    noe_simulation = ctrl.ControlSystemSimulation(noe_ctrl)

    # Clip input values to the valid range
    s[0] = np.clip(s[0], 0, 2)  # Ensure density is within [0, 2]
    s[1] = np.clip(s[1], 0, 1000)  # Ensure distance is within [0, 1000]
    s[2] = np.clip(s[2], 0, 150)  # Ensure avel is within [0, 150]

    # Set inputs to the fuzzy system
    noe_simulation.input['density'] = s[0]
    noe_simulation.input['distance'] = s[1]
    noe_simulation.input['avel'] = s[2]

    # Compute the fuzzy output
    try:
        noe_simulation.compute()  # Try computing the output
        fuzzy_output = noe_simulation.output['noe']
    except KeyError:
        fuzzy_output = 0.5  # Default value if computation fails

    return fuzzy_output

# Example usage and testing:
##################vel###########################################################################
def fuzzy_vel(s, R):
    # Define fuzzy variables
    denc = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'denc')
    avel = ctrl.Antecedent(np.arange(0, 151, 1), 'avel')
    vel = ctrl.Consequent(np.arange(0, 151, 1), 'vel')

    # Define membership functions for 'denc'
    denc_mf = [
        fuzz.trapmf(denc.universe, [0, 0, 0.1, 0.1]),  # vs
        fuzz.trimf(denc.universe, [0.09, 0.25, 0.5]),  # s
        fuzz.trimf(denc.universe, [0.25, 0.5, 0.75]),  # m
        fuzz.trimf(denc.universe, [0.5, 0.75, 1]),     # l
        fuzz.trapmf(denc.universe, [0.75, 1, 1.5, 2])  # vl
    ]
    denc['vs'], denc['s'], denc['m'], denc['l'], denc['vl'] = denc_mf

    # Define membership functions for 'avel'
    avel_mf = [
        fuzz.trapmf(avel.universe, [0, 0, 12, 22.5]),  # vs
        fuzz.trimf(avel.universe, [0, 22.5, 45]),      # s
        fuzz.trimf(avel.universe, [22.5, 45, 67.5]),   # m
        fuzz.trimf(avel.universe, [45, 67.5, 90]),     # l
        fuzz.trapmf(avel.universe, [67.5, 80, 90, 90]) # vl
    ]
    avel['vs'], avel['s'], avel['m'], avel['l'], avel['vl'] = avel_mf

    # Define membership functions for 'vel'
    vel_mf = [
        fuzz.trapmf(vel.universe, [0, 0, 12, 22.5]),   # vs
        fuzz.trimf(vel.universe, [0, 22.5, 45]),       # s
        fuzz.trimf(vel.universe, [22.5, 45, 67.5]),    # m
        fuzz.trimf(vel.universe, [45, 67.5, 90]),      # l
        fuzz.trapmf(vel.universe, [67.5, 80, 90, 150]) # vl
    ]
    vel['vs'], vel['s'], vel['m'], vel['l'], vel['vl'] = vel_mf

    # Create rule list based on input rules
    rule_list = []

    denc_terms = ['vs', 's', 'm', 'l', 'vl']
    avel_terms = ['vs', 's', 'm', 'l', 'vl']
    vel_terms = ['vs', 's', 'm', 'l', 'vl']

    for i in range(len(R)):
        rule = ctrl.Rule(denc[denc_terms[int(R[i][0])]] &
                         avel[avel_terms[int(R[i][1])]],
                         vel[vel_terms[int(R[i][2])]])
        rule_list.append(rule)

    # Add rules to the system
    vel_ctrl = ctrl.ControlSystem(rule_list)
    vel_simulation = ctrl.ControlSystemSimulation(vel_ctrl)
    maxdencity=2
    if s[0]>maxdencity:
        s[0]=maxdencity

    # Set inputs to the fuzzy system
    vel_simulation.input['denc'] = s[0]
    vel_simulation.input['avel'] = s[1]

    # Compute the fuzzy output
    try:
        vel_simulation.compute()  # Try computing the output
        fuzzy_output = vel_simulation.output['vel']
    except KeyError:
        fuzzy_output = 50


    return fuzzy_output

###############NON STOP###########################################
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Precompute membership functions
def create_membership_functions3():
    # Define fuzzy variables
    denc = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'denc')
    dis = ctrl.Antecedent(np.arange(0, 1001, 1), 'dis')
    vel = ctrl.Antecedent(np.arange(0, 151, 1), 'vel')
    noe = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'noe')

    # Define membership functions for 'denc'
    denc_mf = [
        fuzz.trapmf(denc.universe, [0, 0, 0.1, 0.1]),   # vs
        fuzz.trimf(denc.universe, [0.09, 0.25, 0.5]),   # s
        fuzz.trimf(denc.universe, [0.25, 0.5, 0.75]),   # m
        fuzz.trimf(denc.universe, [0.5, 0.75, 1]),      # l
        fuzz.trapmf(denc.universe, [0.75, 1, 1.5, 2])   # vl
    ]
    denc['vs'], denc['s'], denc['m'], denc['l'], denc['vl'] = denc_mf

    # Define membership functions for 'dis'
    dis_mf = [
        fuzz.trapmf(dis.universe, [0, 0, 300, 450]),    # vs
        fuzz.trimf(dis.universe, [300, 450, 600]),      # l
        fuzz.trapmf(dis.universe, [600, 800, 1000, 1000])  # vl
    ]
    dis['vs'], dis['l'], dis['vl'] = dis_mf

    # Define membership functions for 'vel'
    vel_mf = [
        fuzz.trapmf(vel.universe, [0, 0, 12, 22.5]),    # vs
        fuzz.trimf(vel.universe, [0, 22.5, 45]),        # s
        fuzz.trimf(vel.universe, [22.5, 45, 67.5]),     # m
        fuzz.trimf(vel.universe, [45, 67.5, 90]),       # l
        fuzz.trapmf(vel.universe, [67.5, 80, 90, 150])  # vl
    ]
    vel['vs'], vel['s'], vel['m'], vel['l'], vel['vl'] = vel_mf

    # Define membership functions for 'noe'
    noe_mf = [
        fuzz.trapmf(noe.universe, [0, 0, 0.5, 0.5]),    # yes
        fuzz.trapmf(noe.universe, [0.5, 0.5, 1, 1])     # no
    ]
    noe['yes'], noe['no'] = noe_mf

    return denc, dis, vel, noe

# Global control system and simulation (created once, reused)
denc, dis, vel, noe = create_membership_functions3()
noe_ctrl = None
noe_simulation = None

def fuzzy_non(s, R):
    global noe_ctrl, noe_simulation

    # Initialize the control system and simulation only once
    if noe_ctrl is None or noe_simulation is None:
        rule_list = []
        denc_terms = ['vs', 's', 'm', 'l', 'vl']
        dis_terms = ['vs', 'l', 'vl']
        vel_terms = ['vs', 's', 'm', 'l', 'vl']
        noe_terms = ['yes', 'no']

        # Create rule list based on input rules
        for i in range(len(R)):
            rule = ctrl.Rule(denc[denc_terms[int(R[i][0])]] &
                             dis[dis_terms[int(R[i][1])]] &
                             vel[vel_terms[int(R[i][2])]],
                             noe[noe_terms[int(R[i][3])]])
            rule_list.append(rule)

        # Add rules to the system
        noe_ctrl = ctrl.ControlSystem(rule_list)
        noe_simulation = ctrl.ControlSystemSimulation(noe_ctrl)

    # Set inputs to the fuzzy system
    maxdencity = 2
    s[0] = np.clip(s[0], 0, maxdencity)  # Use numpy clip to limit the values to maxdencity
    noe_simulation.input['denc'] = s[0]
    noe_simulation.input['dis'] = s[1]
    noe_simulation.input['vel'] = s[2]

    # Compute the fuzzy output
    try:
        noe_simulation.compute()  # Try computing the output
        fuzzy_output = noe_simulation.output['noe']
    except KeyError:
        fuzzy_output = 0.5  # Default value if computation fails
        print("Error in computing noe output")

    return fuzzy_output

##############evonodd########################################################
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Precompute membership functions
def create_membership_functions2():
    # Define fuzzy variables
    mor = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'mor')
    denc = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'denc')
    evonodd = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'evonodd')

    # Membership functions for 'mor'
    mor['yes'] = fuzz.trapmf(mor.universe, [0, 0, 0.5, 0.5])
    mor['no'] = fuzz.trapmf(mor.universe, [0.5, 0.5, 1, 1])

    # Membership functions for 'denc'
    denc['vs'] = fuzz.trapmf(denc.universe, [0, 0, 0.1, 0.1])
    denc['s'] = fuzz.trimf(denc.universe, [0.09, 0.25, 0.5])
    denc['m'] = fuzz.trimf(denc.universe, [0.25, 0.5, 0.75])
    denc['l'] = fuzz.trimf(denc.universe, [0.5, 0.75, 1])
    denc['vl'] = fuzz.trapmf(denc.universe, [0.75, 1, 1.5, 2])

    # Membership functions for 'evonodd'
    evonodd['evon'] = fuzz.trapmf(evonodd.universe, [0, 0, 0.3, 0.3])
    evonodd['no'] = fuzz.trapmf(evonodd.universe, [0.3, 0.3, 0.6, 0.6])
    evonodd['odd'] = fuzz.trapmf(evonodd.universe, [0.6, 0.6, 1, 1])

    return mor, denc, evonodd

# Global control system and simulation (created once, reused)
mor, denc, evonodd = create_membership_functions2()
evonodd_ctrl = None
evonodd_simulation = None

def fuzzy_evod(s, R):
    global evonodd_ctrl, evonodd_simulation

    # Initialize the control system and simulation only once
    if evonodd_ctrl is None or evonodd_simulation is None:
        rule_list = []
        mor_terms = ['yes', 'no']
        denc_terms = ['vs', 's', 'm', 'l', 'vl']
        evonodd_terms = ['evon', 'no', 'odd']

        # Create rule list based on input rules
        for i in range(len(R)):
            rule = ctrl.Rule(mor[mor_terms[int(R[i][0])]] &
                             denc[denc_terms[int(R[i][1])]],
                             evonodd[evonodd_terms[int(R[i][2])]])
            rule_list.append(rule)

        # Add rules to the system
        evonodd_ctrl = ctrl.ControlSystem(rule_list)
        evonodd_simulation = ctrl.ControlSystemSimulation(evonodd_ctrl)

    # Set inputs to the fuzzy system
    maxdencity = 2
    s[1] = np.clip(s[1], 0, maxdencity)  # Use numpy clip to limit the values to maxdencity
    evonodd_simulation.input['mor'] = s[0]
    evonodd_simulation.input['denc'] = s[1]

    # Compute the fuzzy output
    try:
        evonodd_simulation.compute()  # Try computing the output
        fuzzy_output = evonodd_simulation.output['evonodd']
    except KeyError:
        fuzzy_output = 0.5  # Default value if computation fails
        print("Error in computing evonodd output")

    return fuzzy_output

###############fuzzy_turn#############################################################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Precompute membership functions
def create_membership_functions1():
    # Define fuzzy variables for densities (denc1, denc2, denc3) and output (turn)
    denc1 = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'denc1')
    denc2 = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'denc2')
    denc3 = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'denc3')
    turn = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'turn')

    # Membership functions for 'denc1', 'denc2', 'denc3'
    for denc in [denc1, denc2, denc3]:
        denc['vs'] = fuzz.trapmf(denc.universe, [0, 0, 0.1, 0.1])
        denc['s'] = fuzz.trimf(denc.universe, [0.09, 0.25, 0.5])
        denc['m'] = fuzz.trimf(denc.universe, [0.25, 0.5, 0.75])
        denc['l'] = fuzz.trimf(denc.universe, [0.5, 0.75, 1])
        denc['vl'] = fuzz.trapmf(denc.universe, [0.75, 1, 1.5, 2])

    # Membership functions for 'turn'
    turn['0vs'] = fuzz.trapmf(turn.universe, [0, 0, 0.15, 0.15])
    turn['0s'] = fuzz.trapmf(turn.universe, [0.15, 0.15, 0.25, 0.25])
    turn['0m'] = fuzz.trapmf(turn.universe, [0.25, 0.25, 0.4, 0.4])
    turn['0l'] = fuzz.trapmf(turn.universe, [0.4, 0.4, 0.6, 0.6])
    turn['0vl'] = fuzz.trapmf(turn.universe, [0.6, 0.6, 0.8, 0.8])
    turn['0v1l'] = fuzz.trapmf(turn.universe, [0.8, 0.8, 1, 1])

    return denc1, denc2, denc3, turn

# Global control system and simulation (created once, reused)
denc1, denc2, denc3, turn = create_membership_functions1()
turn_ctrl = None
turn_simulation = None

def fuzzy_turn(s, R):
    global turn_ctrl, turn_simulation

    # Initialize the control system and simulation only once
    if turn_ctrl is None or turn_simulation is None:
        rule_list = []
        denc_terms = ['vs', 's', 'm', 'l', 'vl']
        turn_terms = ['0vs', '0s', '0m', '0l', '0vl', '0v1l']

        # Define rules based on R
        for i in range(len(R)):
            rule = ctrl.Rule(
                denc1[denc_terms[int(R[i][0])]] &
                denc2[denc_terms[int(R[i][1])]] &
                denc3[denc_terms[int(R[i][2])]],
                turn[turn_terms[int(R[i][3])]]
            )
            rule_list.append(rule)

        # Add rules to the control system
        turn_ctrl = ctrl.ControlSystem(rule_list)
        turn_simulation = ctrl.ControlSystemSimulation(turn_ctrl)

    # Set inputs to the fuzzy system
    maxdencity = 2
    s = np.clip(s, 0, maxdencity)  # Use numpy clip to limit the values to maxdencity
    turn_simulation.input['denc1'] = s[0]
    turn_simulation.input['denc2'] = s[1]
    turn_simulation.input['denc3'] = s[2]

    # Compute the fuzzy output
    try:
        turn_simulation.compute()  # Try computing the output
        fuzzy_output = turn_simulation.output['turn']
    except KeyError:
        fuzzy_output = 0.5  # Default value if computation fails
        print("Error in computing turn output")

    return fuzzy_output