import numpy as np

def trapmf(x, params):
    a, b, c, d = params
    if b == a or c == d:
        return 1
    if x <= a:
        return 0
    elif x <= b:
        return (x - a) / (b - a)
    elif x <= c:
        return 1
    elif x <= d:
        return (d - x) / (d - c)
    return 0

def trimf(x, params):
    a, b, c = params
    if x <= a:
        return 0
    elif x <= b:
        return (x - a) / (b - a)
    elif x <= c:
        return (c - x) / (c - b)
    return 0

def evalmf(mf_type, x):
    """Evaluate membership functions based on type."""
    if mf_type[0] == 'trapmf':
        return trapmf(x, mf_type[1])
    elif mf_type[0] == 'trimf':
        return trimf(x, mf_type[1])
    else:
        raise ValueError("Unsupported membership function type")

def fire_action_direction(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 12, 22.5]),
        ('trimf', [0, 22.5, 45]),
        ('trimf', [22.5, 45, 67.5]),
        ('trimf', [45, 67.5, 90]),
        ('trapmf', [67.5, 80, 90, 150]),
    ]

    # Use vectorization
    maxdencity = 2
    s[0] = min(s[0], maxdencity)
    
    x = np.zeros((25, 1))
    for i in range(25):
        A = evalmf(mf1[in_data[i][0]], s[0])
        B = evalmf(mf2[in_data[i][1]], s[1])
        x[i] = min(A, B)
    
    return x

def fire_action_light(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 10/3, 30/3]),
        ('trimf', [0.0, 30/3, 40/3]),
        ('trimf', [30/3, 60/3, 90/3]),
        ('trimf', [60/3, 90/3, 120/3]),
        ('trapmf', [90/3, 100/3, 120/3, 150/3]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 10/3, 30/3]),
        ('trimf', [0.0, 30/3, 40/3]),
        ('trimf', [30/3, 60/3, 90/3]),
        ('trimf', [60/3, 90/3, 120/3]),
        ('trapmf', [90/3, 100/3, 120/3, 150/3]),
    ]

    # Vectorize calculations
    maxdencity = 200
    s = np.minimum(s, maxdencity)
    
    x = np.zeros((25, 1))
    for i in range(25):
        A = evalmf(mf1[in_data[i][0]], s[0])
        B = evalmf(mf2[in_data[i][1]], s[1])
        x[i] = min(A, B)
    
    return x

def fire_action_noe(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 300, 450]),
        ('trimf', [300, 450, 600]),
        ('trapmf', [600, 800, 1000, 1000]),
    ]
    
    mf3 = [
        ('trapmf', [0, 0, 12, 22.5]),
        ('trimf', [0, 22.5, 45]),
        ('trimf', [22.5, 45, 67.5]),
        ('trimf', [45, 67.5, 90]),
        ('trapmf', [67.5, 80, 90, 150]),
    ]
    
    x = np.zeros((75, 1))
    for i in range(75):
        A = evalmf(mf1[int(in_data[i][0])], s[0])
        B = evalmf(mf2[int(in_data[i][1])], s[1])
        C = evalmf(mf3[int(in_data[i][2])], s[2])
        x[i] = min(A, B, C)
    
    return x

def fire_action_vel(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 12, 22.5]),
        ('trimf', [0, 22.5, 45]),
        ('trimf', [22.5, 45, 67.5]),
        ('trimf', [45, 67.5, 90]),
        ('trapmf', [67.5, 80, 90, 150]),
    ]
    
    x = np.zeros((25, 1))
    maxdencity = 2
    s[0] = min(s[0], maxdencity)
    
    for i in range(25):
        A = evalmf(mf1[int(in_data[i][0])], s[0])
        B = evalmf(mf2[int(in_data[i][1])], s[1])
        x[i] = min(A, B)
    
    return x
import numpy as np

def fire_action_non(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 300, 450]),
        ('trimf', [300, 450, 600]),
        ('trapmf', [600, 800, 1000, 1000]),
    ]
    
    mf3 = [
        ('trapmf', [0, 0, 12, 22.5]),
        ('trimf', [0, 22.5, 45]),
        ('trimf', [22.5, 45, 67.5]),
        ('trimf', [45, 67.5, 90]),
        ('trapmf', [67.5, 80, 90, 150]),
    ]
    
    x = np.zeros((75, 1))  # Initialize output array
    maxdencity = 2
    s[0] = min(s[0], maxdencity)
    
    for i in range(75):
        A = evalmf(mf1[int(in_data[i][0])], s[0])
        B = evalmf(mf2[int(in_data[i][1])], s[1])
        C = evalmf(mf3[int(in_data[i][2])], s[2])
        x[i, 0] = min(A, B, C)  # Store the minimum value
    
    return x

def fire_action_evod(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 0.5, 0.5]),
        ('trapmf', [0.5, 0.5, 1, 1]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    x = np.zeros((10, 1))  # Initialize output array
    maxdencity = 2
    s[1] = min(s[1], maxdencity)
    
    for i in range(10):
        A = evalmf(mf1[int(in_data[i][0])], s[0])
        B = evalmf(mf2[int(in_data[i][1])], s[1])
        x[i, 0] = min(A, B)  # Store the minimum value
    
    return x

def fire_action_turn(in_data, s):
    # Define membership functions
    mf1 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    mf2 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    mf3 = [
        ('trapmf', [0, 0, 0.1, 0.1]),
        ('trimf', [0.09, 0.25, 0.5]),
        ('trimf', [0.25, 0.5, 0.75]),
        ('trimf', [0.5, 0.75, 1]),
        ('trapmf', [0.75, 1, 1.5, 2]),
    ]
    
    x = np.zeros((125, 1))  # Initialize output array
    maxdencity = 2
    s[0] = min(s[0], maxdencity)
    s[1] = min(s[1], maxdencity)
    s[2] = min(s[2], maxdencity)
    
    for i in range(125):
        A = evalmf(mf1[int(in_data[i][0])], s[0])
        B = evalmf(mf2[int(in_data[i][1])], s[1])
        C = evalmf(mf3[int(in_data[i][2])], s[2])
        x[i, 0] = min(A, B, C)  # Store the minimum value
    
    return x
