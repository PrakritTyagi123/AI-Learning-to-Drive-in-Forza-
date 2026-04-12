"""
ForzaTek AI — UDP Telemetry Parser
Parses the Forza "Dash" packet format used by FH4, FH5, and FM7+.
Packet = Sled (232 bytes) + Dash extension (92 bytes) = 324 bytes total.
FH4/FH5 use a slightly different layout with a 12-byte placeholder = 323 bytes.
We handle both.
"""

import struct
import math

# ─── Sled format (first 232 bytes, shared across all Forza titles) ───
SLED_FORMAT = "<"  # little-endian
SLED_FORMAT += "i"    # IsRaceOn (int32)
SLED_FORMAT += "I"    # TimestampMs (uint32)
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

SLED_SIZE = struct.calcsize(SLED_FORMAT)  # Should be 232

# ─── Dash extension for Horizon titles (after sled) ───
# CarOrdinal, CarClass, CarPI, DrivetrainType, NumCylinders = 5 int32s = 20 bytes
# HorizonPlaceholder = 12 bytes (unknown, skip)
# Then: PositionX/Y/Z, Speed, Power, Torque (6 floats)
# TireTemp FL/FR/RL/RR (4 floats)
# Boost, Fuel, DistanceTraveled, BestLap, LastLap, CurrentLap, CurrentRaceTime (7 floats)
# LapNumber (uint16), RacePosition (uint8)
# Accel, Brake, Clutch, HandBrake, Gear, Steer (uint8 x5 + int8)
# NormalizedDrivingLine, NormalizedAIBrakeDifference (int8, int8)

DASH_HORIZON_FORMAT = "<"
DASH_HORIZON_FORMAT += "iiiii"    # CarOrdinal, CarClass, CarPI, DrivetrainType, NumCylinders
DASH_HORIZON_FORMAT += "12x"      # HorizonPlaceholder (12 bytes, skip)
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
DASH_HORIZON_FORMAT += "H"        # LapNumber (uint16)
DASH_HORIZON_FORMAT += "B"        # RacePosition (uint8)
DASH_HORIZON_FORMAT += "B"        # Accel (uint8, 0-255)
DASH_HORIZON_FORMAT += "B"        # Brake (uint8, 0-255)
DASH_HORIZON_FORMAT += "B"        # Clutch (uint8, 0-255)
DASH_HORIZON_FORMAT += "B"        # HandBrake (uint8, 0-255)
DASH_HORIZON_FORMAT += "B"        # Gear (uint8)
DASH_HORIZON_FORMAT += "b"        # Steer (int8, -127 to 127)
DASH_HORIZON_FORMAT += "b"        # NormalizedDrivingLine (int8)
DASH_HORIZON_FORMAT += "b"        # NormalizedAIBrakeDifference (int8)

DASH_HORIZON_SIZE = struct.calcsize(DASH_HORIZON_FORMAT)
TOTAL_HORIZON_SIZE = SLED_SIZE + DASH_HORIZON_SIZE

# ─── Motorsport Dash (no 12-byte placeholder) ───
DASH_MOTORSPORT_FORMAT = DASH_HORIZON_FORMAT.replace("12x", "")
DASH_MOTORSPORT_SIZE = struct.calcsize(DASH_MOTORSPORT_FORMAT)
TOTAL_MOTORSPORT_SIZE = SLED_SIZE + DASH_MOTORSPORT_SIZE


def mps_to_mph(mps: float) -> float:
    return mps * 2.23694

def watts_to_kw(w: float) -> float:
    return w / 1000.0

def watts_to_hp(w: float) -> float:
    return w / 745.7

def format_lap_time(seconds: float) -> str:
    if seconds <= 0:
        return "--:--.---"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def parse_packet(data: bytes) -> dict | None:
    """
    Parse a Forza UDP packet. Returns a dict of telemetry values or None if invalid.
    Handles both Horizon (324 bytes) and Motorsport (312 bytes) formats.
    """
    size = len(data)

    if size >= TOTAL_HORIZON_SIZE:
        is_horizon = True
        dash_format = DASH_HORIZON_FORMAT
    elif size >= TOTAL_MOTORSPORT_SIZE:
        is_horizon = False
        dash_format = DASH_MOTORSPORT_FORMAT
    else:
        return None  # Unknown packet size

    try:
        sled = struct.unpack(SLED_FORMAT, data[:SLED_SIZE])
        dash = struct.unpack(dash_format, data[SLED_SIZE:SLED_SIZE + struct.calcsize(dash_format)])
    except struct.error:
        return None

    # ─── Unpack sled fields ───
    is_race_on = sled[0]
    timestamp_ms = sled[1]
    engine_max_rpm = sled[2]
    engine_idle_rpm = sled[3]
    current_rpm = sled[4]
    accel_x, accel_y, accel_z = sled[5], sled[6], sled[7]
    vel_x, vel_y, vel_z = sled[8], sled[9], sled[10]
    ang_vel_x, ang_vel_y, ang_vel_z = sled[11], sled[12], sled[13]
    yaw, pitch, roll = sled[14], sled[15], sled[16]
    susp_fl, susp_fr, susp_rl, susp_rr = sled[17], sled[18], sled[19], sled[20]
    slip_fl, slip_fr, slip_rl, slip_rr = sled[21], sled[22], sled[23], sled[24]
    wheel_speed_fl, wheel_speed_fr = sled[25], sled[26]
    wheel_speed_rl, wheel_speed_rr = sled[27], sled[28]
    rumble_fl, rumble_fr, rumble_rl, rumble_rr = sled[29], sled[30], sled[31], sled[32]
    puddle_fl, puddle_fr, puddle_rl, puddle_rr = sled[33], sled[34], sled[35], sled[36]
    surface_fl, surface_fr, surface_rl, surface_rr = sled[37], sled[38], sled[39], sled[40]
    slip_angle_fl, slip_angle_fr = sled[41], sled[42]
    slip_angle_rl, slip_angle_rr = sled[43], sled[44]
    combined_slip_fl, combined_slip_fr = sled[45], sled[46]
    combined_slip_rl, combined_slip_rr = sled[47], sled[48]
    susp_meters_fl, susp_meters_fr = sled[49], sled[50]
    susp_meters_rl, susp_meters_rr = sled[51], sled[52]

    # ─── Unpack dash fields ───
    idx = 0
    car_ordinal = dash[idx]; idx += 1
    car_class = dash[idx]; idx += 1
    car_pi = dash[idx]; idx += 1
    drivetrain = dash[idx]; idx += 1
    num_cylinders = dash[idx]; idx += 1
    # Horizon placeholder is skipped by struct format
    pos_x = dash[idx]; idx += 1
    pos_y = dash[idx]; idx += 1
    pos_z = dash[idx]; idx += 1
    speed_mps = dash[idx]; idx += 1
    power_w = dash[idx]; idx += 1
    torque_nm = dash[idx]; idx += 1
    tire_temp_fl = dash[idx]; idx += 1
    tire_temp_fr = dash[idx]; idx += 1
    tire_temp_rl = dash[idx]; idx += 1
    tire_temp_rr = dash[idx]; idx += 1
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

    # ─── Compute derived values ───
    g_force_lat = accel_x / 9.81
    g_force_lon = accel_z / 9.81
    g_force_total = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2) / 9.81
    speed_mph = mps_to_mph(speed_mps)

    return {
        "isRaceOn": is_race_on,
        "timestampMs": timestamp_ms,
        "speed": round(speed_mph, 1),
        "rpm": round(current_rpm),
        "maxRpm": round(engine_max_rpm),
        "idleRpm": round(engine_idle_rpm),
        "gear": gear,
        "throttle": round(accel_input / 255.0, 3),
        "brake": round(brake_input / 255.0, 3),
        "clutch": round(clutch_input / 255.0, 3),
        "handbrake": round(handbrake_input / 255.0, 3),
        "steeringAngle": round(steer / 127.0 * 45, 1),  # normalize to degrees
        "gForceX": round(g_force_lat, 3),
        "gForceY": round(g_force_lon, 3),
        "gForceTotal": round(g_force_total, 3),
        "pitch": round(rad_to_deg(pitch), 1),
        "yaw": round(rad_to_deg(yaw), 1),
        "roll": round(rad_to_deg(roll), 1),
        "suspFL": round(susp_fl, 3),
        "suspFR": round(susp_fr, 3),
        "suspRL": round(susp_rl, 3),
        "suspRR": round(susp_rr, 3),
        "tireSlipFL": round(abs(slip_fl), 3),
        "tireSlipFR": round(abs(slip_fr), 3),
        "tireSlipRL": round(abs(slip_rl), 3),
        "tireSlipRR": round(abs(slip_rr), 3),
        "tireAngleFL": round(rad_to_deg(slip_angle_fl), 1),
        "tireAngleFR": round(rad_to_deg(slip_angle_fr), 1),
        "tireAngleRL": round(rad_to_deg(slip_angle_rl), 1),
        "tireAngleRR": round(rad_to_deg(slip_angle_rr), 1),
        "tireTempFL": round(tire_temp_fl, 1),
        "tireTempFR": round(tire_temp_fr, 1),
        "tireTempRL": round(tire_temp_rl, 1),
        "tireTempRR": round(tire_temp_rr, 1),
        "wheelSpeedFL": round(mps_to_mph(abs(wheel_speed_fl)), 1),
        "wheelSpeedFR": round(mps_to_mph(abs(wheel_speed_fr)), 1),
        "wheelSpeedRL": round(mps_to_mph(abs(wheel_speed_rl)), 1),
        "wheelSpeedRR": round(mps_to_mph(abs(wheel_speed_rr)), 1),
        "power": round(watts_to_kw(power_w), 1),
        "torque": round(torque_nm, 1),
        "carClass": car_class,
        "carPI": car_pi,
        "drivetrainType": drivetrain,
        "numCylinders": num_cylinders,
        "distanceTraveled": round(distance, 2),
        "position": race_position,
        "lapNumber": lap_number,
        "currentLap": format_lap_time(current_lap),
        "currentLapRaw": current_lap,
        "bestLap": format_lap_time(best_lap),
        "bestLapRaw": best_lap,
        "lastLap": format_lap_time(last_lap),
        "lastLapRaw": last_lap,
        "currentRaceTime": round(current_race_time, 3),
        "normalizedDrivingLine": norm_driving_line,
        "normalizedAIBrakeDiff": norm_ai_brake_diff,
        "posX": round(pos_x, 2),
        "posY": round(pos_y, 2),
        "posZ": round(pos_z, 2),
        "isHorizon": is_horizon,
    }
