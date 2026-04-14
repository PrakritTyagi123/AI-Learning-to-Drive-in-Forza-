"""
ForzaTek AI — UDP Telemetry Parser
Parses the Forza "Dash" packet format used by FH4, FH5, and FM7+.
Packet = Sled (232 bytes) + Dash extension (92 bytes) = 324 bytes total.
FH4/FH5 use a slightly different layout with a 12-byte placeholder.
"""

import struct
import math

SLED_FORMAT = "<"
SLED_FORMAT += "i"    # IsRaceOn
SLED_FORMAT += "I"    # TimestampMs
SLED_FORMAT += "fff"  # EngineMaxRpm, EngineIdleRpm, CurrentEngineRpm
SLED_FORMAT += "fff"  # AccelerationX, Y, Z
SLED_FORMAT += "fff"  # VelocityX, Y, Z
SLED_FORMAT += "fff"  # AngularVelocityX, Y, Z
SLED_FORMAT += "fff"  # Yaw, Pitch, Roll
SLED_FORMAT += "ffff" # NormalizedSuspensionTravel FL, FR, RL, RR
SLED_FORMAT += "ffff" # TireSlipRatio FL, FR, RL, RR
SLED_FORMAT += "ffff" # WheelRotationSpeed FL, FR, RL, RR
SLED_FORMAT += "iiii" # WheelOnRumbleStrip FL, FR, RL, RR
SLED_FORMAT += "ffff" # WheelInPuddleDepth FL, FR, RL, RR
SLED_FORMAT += "ffff" # SurfaceRumble FL, FR, RL, RR
SLED_FORMAT += "ffff" # TireSlipAngle FL, FR, RL, RR
SLED_FORMAT += "ffff" # TireCombinedSlip FL, FR, RL, RR
SLED_FORMAT += "ffff" # SuspensionTravelMeters FL, FR, RL, RR
SLED_SIZE = struct.calcsize(SLED_FORMAT)

DASH_HORIZON_FORMAT = "<"
DASH_HORIZON_FORMAT += "iiiii"    # CarOrdinal, CarClass, CarPI, DrivetrainType, NumCylinders
DASH_HORIZON_FORMAT += "12x"      # HorizonPlaceholder
DASH_HORIZON_FORMAT += "fff"      # PositionX, Y, Z
DASH_HORIZON_FORMAT += "f"        # Speed (m/s)
DASH_HORIZON_FORMAT += "f"        # Power (watts)
DASH_HORIZON_FORMAT += "f"        # Torque (Nm)
DASH_HORIZON_FORMAT += "ffff"     # TireTemp FL, FR, RL, RR
DASH_HORIZON_FORMAT += "f"        # Boost
DASH_HORIZON_FORMAT += "f"        # Fuel
DASH_HORIZON_FORMAT += "f"        # DistanceTraveled
DASH_HORIZON_FORMAT += "f"        # BestLap
DASH_HORIZON_FORMAT += "f"        # LastLap
DASH_HORIZON_FORMAT += "f"        # CurrentLap
DASH_HORIZON_FORMAT += "f"        # CurrentRaceTime
DASH_HORIZON_FORMAT += "H"        # LapNumber
DASH_HORIZON_FORMAT += "B"        # RacePosition
DASH_HORIZON_FORMAT += "B"        # Accel
DASH_HORIZON_FORMAT += "B"        # Brake
DASH_HORIZON_FORMAT += "B"        # Clutch
DASH_HORIZON_FORMAT += "B"        # HandBrake
DASH_HORIZON_FORMAT += "B"        # Gear
DASH_HORIZON_FORMAT += "b"        # Steer
DASH_HORIZON_FORMAT += "b"        # NormalizedDrivingLine
DASH_HORIZON_FORMAT += "b"        # NormalizedAIBrakeDifference
DASH_HORIZON_SIZE = struct.calcsize(DASH_HORIZON_FORMAT)
TOTAL_HORIZON_SIZE = SLED_SIZE + DASH_HORIZON_SIZE

DASH_MOTORSPORT_FORMAT = DASH_HORIZON_FORMAT.replace("12x", "")
DASH_MOTORSPORT_SIZE = struct.calcsize(DASH_MOTORSPORT_FORMAT)
TOTAL_MOTORSPORT_SIZE = SLED_SIZE + DASH_MOTORSPORT_SIZE


def mps_to_mph(mps): return mps * 2.23694
def watts_to_kw(w): return w / 1000.0
def rad_to_deg(rad): return rad * 180.0 / math.pi

def format_lap_time(seconds):
    if seconds <= 0: return "--:--.---"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def parse_packet(data: bytes) -> dict | None:
    size = len(data)
    if size >= TOTAL_HORIZON_SIZE:
        is_horizon = True
        dash_format = DASH_HORIZON_FORMAT
    elif size >= TOTAL_MOTORSPORT_SIZE:
        is_horizon = False
        dash_format = DASH_MOTORSPORT_FORMAT
    else:
        return None

    try:
        sled = struct.unpack(SLED_FORMAT, data[:SLED_SIZE])
        dash = struct.unpack(dash_format, data[SLED_SIZE:SLED_SIZE + struct.calcsize(dash_format)])
    except struct.error:
        return None

    is_race_on = sled[0]
    engine_max_rpm, engine_idle_rpm, current_rpm = sled[2], sled[3], sled[4]
    accel_x, accel_y, accel_z = sled[5], sled[6], sled[7]
    yaw, pitch, roll = sled[14], sled[15], sled[16]
    susp_fl, susp_fr, susp_rl, susp_rr = sled[17], sled[18], sled[19], sled[20]
    slip_fl, slip_fr, slip_rl, slip_rr = sled[21], sled[22], sled[23], sled[24]
    wheel_speed_fl, wheel_speed_fr = sled[25], sled[26]
    wheel_speed_rl, wheel_speed_rr = sled[27], sled[28]
    slip_angle_fl, slip_angle_fr = sled[41], sled[42]
    slip_angle_rl, slip_angle_rr = sled[43], sled[44]

    idx = 0
    car_ordinal = dash[idx]; idx += 1
    car_class = dash[idx]; idx += 1
    car_pi = dash[idx]; idx += 1
    drivetrain = dash[idx]; idx += 1
    num_cylinders = dash[idx]; idx += 1
    pos_x, pos_y, pos_z = dash[idx], dash[idx+1], dash[idx+2]; idx += 3
    speed_mps = dash[idx]; idx += 1
    power_w = dash[idx]; idx += 1
    torque_nm = dash[idx]; idx += 1
    tire_temp_fl, tire_temp_fr = dash[idx], dash[idx+1]; idx += 2
    tire_temp_rl, tire_temp_rr = dash[idx], dash[idx+1]; idx += 2
    boost = dash[idx]; idx += 1
    fuel = dash[idx]; idx += 1
    distance = dash[idx]; idx += 1
    best_lap = dash[idx]; idx += 1
    last_lap = dash[idx]; idx += 1
    current_lap = dash[idx]; idx += 1
    current_race_time = dash[idx]; idx += 1
    lap_number = dash[idx]; idx += 1
    race_position = dash[idx]; idx += 1
    accel_input = dash[idx]; idx += 1
    brake_input = dash[idx]; idx += 1
    clutch_input = dash[idx]; idx += 1
    handbrake_input = dash[idx]; idx += 1
    gear = dash[idx]; idx += 1
    steer = dash[idx]; idx += 1
    norm_driving_line = dash[idx]; idx += 1
    norm_ai_brake_diff = dash[idx]; idx += 1

    g_force_lat = accel_x / 9.81
    g_force_lon = accel_z / 9.81
    g_force_total = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2) / 9.81
    speed_mph = mps_to_mph(speed_mps)

    return {
        "isRaceOn": is_race_on, "timestampMs": sled[1],
        "speed": round(speed_mph, 1), "rpm": round(current_rpm),
        "maxRpm": round(engine_max_rpm), "idleRpm": round(engine_idle_rpm), "gear": gear,
        "throttle": round(accel_input / 255.0, 3), "brake": round(brake_input / 255.0, 3),
        "clutch": round(clutch_input / 255.0, 3), "handbrake": round(handbrake_input / 255.0, 3),
        "steeringAngle": round(steer / 127.0 * 45, 1),
        "gForceX": round(g_force_lat, 3), "gForceY": round(g_force_lon, 3),
        "gForceTotal": round(g_force_total, 3),
        "pitch": round(rad_to_deg(pitch), 1), "yaw": round(rad_to_deg(yaw), 1),
        "roll": round(rad_to_deg(roll), 1),
        "suspFL": round(susp_fl, 3), "suspFR": round(susp_fr, 3),
        "suspRL": round(susp_rl, 3), "suspRR": round(susp_rr, 3),
        "tireSlipFL": round(abs(slip_fl), 3), "tireSlipFR": round(abs(slip_fr), 3),
        "tireSlipRL": round(abs(slip_rl), 3), "tireSlipRR": round(abs(slip_rr), 3),
        "tireAngleFL": round(rad_to_deg(slip_angle_fl), 1), "tireAngleFR": round(rad_to_deg(slip_angle_fr), 1),
        "tireAngleRL": round(rad_to_deg(slip_angle_rl), 1), "tireAngleRR": round(rad_to_deg(slip_angle_rr), 1),
        "tireTempFL": round(tire_temp_fl, 1), "tireTempFR": round(tire_temp_fr, 1),
        "tireTempRL": round(tire_temp_rl, 1), "tireTempRR": round(tire_temp_rr, 1),
        "wheelSpeedFL": round(mps_to_mph(abs(wheel_speed_fl)), 1),
        "wheelSpeedFR": round(mps_to_mph(abs(wheel_speed_fr)), 1),
        "wheelSpeedRL": round(mps_to_mph(abs(wheel_speed_rl)), 1),
        "wheelSpeedRR": round(mps_to_mph(abs(wheel_speed_rr)), 1),
        "power": round(watts_to_kw(power_w), 1), "torque": round(torque_nm, 1),
        "carClass": car_class, "carPI": car_pi, "drivetrainType": drivetrain,
        "numCylinders": num_cylinders, "distanceTraveled": round(distance, 2),
        "position": race_position, "lapNumber": lap_number,
        "currentLap": format_lap_time(current_lap), "currentLapRaw": current_lap,
        "bestLap": format_lap_time(best_lap), "bestLapRaw": best_lap,
        "lastLap": format_lap_time(last_lap), "lastLapRaw": last_lap,
        "currentRaceTime": round(current_race_time, 3),
        "normalizedDrivingLine": norm_driving_line,
        "normalizedAIBrakeDiff": norm_ai_brake_diff,
        "posX": round(pos_x, 2), "posY": round(pos_y, 2), "posZ": round(pos_z, 2),
        "isHorizon": is_horizon,
    }
