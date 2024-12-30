import numpy as np
## idm acceleration determination
def idm_da(aa, p4, p1, bb, p0l, p0, vmax, s0, T, p4l):
    Rv = p4 - p4l
    term1 = 1 - (p4 / vmax[int(p1)]) ** 4
    A = np.sqrt(aa * bb)
    term2 = (s0 + p4 * T) + ((p4 * Rv) / (2 * (2 * A)))
    term3 = p0l - p0 - 1

    desired_acceleration = aa * (term1 - (term2 / term3))
    return desired_acceleration
import numpy as np
def idm_speed_update(ko, pos, i, j, red11, green11, red12, green12, trafic_light_matrix1, speed, dt, vmax, d, mm, pp):
    s0 = 1  # Minimum bumper-to-bumper distance (in meters)
    T = .5  # Safe time headway (in seconds)
    aa = 1  # Maximum acceleration (in m/s^2)
    bb = .1  # Comfortable deceleration (in m/s^2)

    A = ko[j]
    
    if pos[1, i] == j:  # This car is in lane `j`
        for k in range(len(A)):
            if pos[3, i] == k:
                lead_car = ko[j]

                # Check if the car is the first car (the lead car)
                if k == 0:  # First car, no car ahead
                    # Accelerate freely (no car in front, use max acceleration)
                    p4 = pos[4, i]  # Current speed
                    p1 = pos[1, i]  # Current lane
                    p0l = pos[0, i]  # Position of this car
                    p0 = pos[0, i]
                    p4l = pos[4, i]  # Speed of this car

                    # Calculate desired acceleration using just the max acceleration
                    desired_acceleration = aa * (1 - (p4 / vmax[int(p1)]) ** 2)  # Use max acceleration
        

                    if mm == 0:
                        pos[4, i] = speed[i] + desired_acceleration * dt

                    else:
                        for tc in range(len(trafic_light_matrix1[0])):
                            if pos[1, i] in trafic_light_matrix1[0][tc] and pos[0, i] >= d[int(pos[1, i])] - 30:
                                if red11[tc] == 1:  # Red light
                                    pos[4, i] = max(0, pos[4, i] - bb * dt)  # Gradual deceleration
                                elif green11[tc] == 1:  # Green light
                                    if pos[4, i] == 0:
                                        pos[4, i] = 50  # Start with a reasonable speed
                                    else:
                                        pos[4, i] += desired_acceleration * dt
                            elif pos[1, i] in trafic_light_matrix1[1][tc] and pos[0, i] >= d[int(pos[1, i])] - 30:
                                if red12[tc] == 1:  # Red light
                                    pos[4, i] = max(0, pos[4, i] - bb * dt)  # Gradual deceleration
                                elif green12[tc] == 1:  # Green light
                                    if pos[4, i] == 0:
                                        pos[4, i] = 50  # Start with a reasonable speed
                                    else:
                                        pos[4, i] += desired_acceleration * dt
                            else:
                                pos[4, i] = speed[i] + desired_acceleration * dt  # Normal acceleration 
                else:
                    # Normal behavior when there's a car in front
                    ll = lead_car[k - 1] if k > 0 else lead_car[0]
                    p4 = pos[4, i]
                    p1 = pos[1, i]
                    p0l = pos[0, ll]
                    p0 = pos[0, i]
                    p4l = pos[4, ll]

                    # Calculate desired acceleration based on the IDM model (lead car is ahead)
                    desired_acceleration = idm_da(aa, p4, p1, bb, p0l, p0, vmax, s0, T, p4l)


                    # Apply the desired acceleration based on the lead car's influence
                    if mm == 0:
                        pos[4, i] = speed[i] + desired_acceleration * dt

                    else:
                        for tc in range(len(trafic_light_matrix1[0])):
                            if pos[1, i] in trafic_light_matrix1[0][tc] and pos[0, i] >= d[int(pos[1, i])] - 30:
                                if red11[tc] == 1:  # Red light
                                    pos[4, i] = max(0, pos[4, i] - bb * dt)  # Gradual deceleration
                                elif green11[tc] == 1:  # Green light
                                    if pos[4, i] == 0:
                                        pos[4, i] = 50  # Start with a reasonable speed
                                    else:
                                        pos[4, i] += desired_acceleration * dt
                            elif pos[1, i] in trafic_light_matrix1[1][tc] and pos[0, i] >= d[int(pos[1, i])] - 30:
                                if red12[tc] == 1:  # Red light
                                    pos[4, i] = max(0, pos[4, i] - bb * dt)  # Gradual deceleration
                                elif green12[tc] == 1:  # Green light
                                    if pos[4, i] == 0:
                                        pos[4, i] = 50  # Start with a reasonable speed
                                    else:
                                        pos[4, i] += desired_acceleration * dt
                            else:
                                pos[4, i] = speed[i] + desired_acceleration * dt  # Normal acceleration
    # Ensure speed is within bounds
    p1 = pos[1, i]
    if np.isnan(pos[4, i]) or np.isinf(pos[4, i]) or pos[4, i] < 0 or pos[4, i] > 100:
        pos[4, i] = vmax[int(p1)]  # Reset to max speed if out of bounds

    return pos
import numpy as np
def IDM7(pos, num_cars, car_edge_path, car_edge_speed, d, numedge, car_stop_matrix, trafic_light_matrix1, numTl1, simulation_time, dt):
    if len(trafic_light_matrix1[0]) == 0:
        timer1 = []
    else:
        timer1 = np.array(numTl1)

    red11 = np.ones(len(trafic_light_matrix1[1]))
    red12 = np.zeros(len(trafic_light_matrix1[0]))
    green11 = np.zeros(len(trafic_light_matrix1[1]))
    green12 = np.ones(len(trafic_light_matrix1[0]))

    stop_time = 2 * np.ones(num_cars)
    num_steps = int(simulation_time / dt)

    pos1 = np.zeros((num_steps, 6, num_cars))
    distence_left = pos[2, :] - pos[0, :]
    pppp = 0*np.ones(num_cars) ## inital speed

    vmax = car_edge_speed  ## the max speed in sytem 
    ko = [None] * numedge

    for j in range(numedge):
        b3 = np.where(pos[1, :] == j)[0]
        FFF = pos[0, b3]
        I1 = np.argsort(-FFF)
        ko[j] = b3[I1]

    pos[4, :] = car_edge_speed[pos[1, :].astype(int)] ## giving the inital  speed of the rad 
    TEMe=np.zeros(num_cars)
    for t in range(num_steps):
        speed = pos[4, :] ## 

        if t % 5 == 0:
            print("idm", "time", "car", num_steps - t)

        for i in range(num_cars):
            if pos[5, i] <= 0:
                for j in range(numedge):
                    mm = 1 if len(timer1) else 0
                    pos = idm_speed_update(ko, pos, i, j, red11, green11, red12, green12, trafic_light_matrix1, speed, dt, vmax, d, mm,pppp[i])
                    
                    pos[0, i] += ((abs(pos[4, i] - pppp[i]) +(.001*pos[4, i]))) *dt
                    pppp[i] = pos[4, i] 
                    #print("speed dif",(abs(pos[4, i] - pppp[i]) ))
                    if pos[0, i] > pos[2, i]:
                        pos[0, i] = pos[2, i] - pos[0, i]
                        F = np.array(car_edge_path[i])
                        a = np.where(F == pos[1, i])
                        b = np.argwhere(F == pos[1, i])

                        if  len(b) == 0 or pos[1, i] == F[-1]:
                            pos[1, i] = F[0]
                        else:
                            pos[1, i] = F[b[0][0] + 1]
                            l5 = int(pos[1, i])
                        if pos[1, i] == F[-1]:
                            if  TEMe[i]==0:
                                TEMe[i]=t
                                #pos[1, i] = F[0]    
                        pos[2, i] = d[int(pos[1, i])]

                        for j in range(numedge):
                            kkk = ko[j]
                            pos[3, kkk] = np.arange(len(kkk), 0, -1)

                b3 = np.where(pos[1, :] == j)[0]
                FFF = pos[0, b3]
                I1 = np.argsort(-FFF)
                ko[j] = b3[I1]
                for tc in range(len(timer1)):
                    if timer1[tc] <= 0:
                        red11[tc], green11[tc] = green11[tc], red11[tc]
                        red12[tc], green12[tc] = green12[tc], red12[tc]
                        timer1[tc] = numTl1[tc]

                if len(car_stop_matrix) > 0:
                    if car_stop_matrix[0, i] == 1 and pos[1, i] == car_stop_matrix[1, i] and stop_time[i] > 0:
                        pos[4, i] = 0
                        stop_time[i] -= dt
                        if stop_time[i] <= 0:
                            stop_time[i] = 2

                if pos[0, i] > pos[2, i] or pos[0, i] < 0:
                    pos[0, i] = 0

            distence_left[i] = pos[2, i] - pos[0, i]
            

        pos[5, :] -= dt
        pos1[t, :, :] = pos[:, :]
        if len(timer1) > 0:
            timer1 -= dt

    return pos1,TEMe
