import csv
from socket import *
import json
import math
import io
import numpy as np

from step_length import *
from step_orientation import *
from utils import *

serverPort = 5000
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print(f"Server is ready to receive at {serverPort}")

defaultRes = {
    "isStepped": False,
    "x": 0.0,
    "y": 0.0,
    "radian": 0.0
}


current_x, current_y = (13, 52) # 초기 위치
current_angle = 0.0  # 초기 방향 (radians)

meter_per_coordinate_unit = 0.6

coordinateWidth = 153.0
coordinateHeight = 63.0

# 수신 중인 메시지 저장소
data_buffer = {}
expected_chunks = {}

def handle_localization(csv_data: io.StringIO):
    # NumPy로 CSV 로드 (첫 번째 행 건너뛰기)
    message_arr = np.genfromtxt(csv_data, delimiter=',', dtype=float)

    acc_arr = message_arr[:,0:3]
    gyro_arr = message_arr[:,3:6]
    # mag_arr = message_arr[:,6:9]
    
    if len(acc_arr) <= 9:
        return defaultRes
        
    step_lengths = compute_step_timestamp(
        acceleration_threshold=0.001,
        weinberg_gain=0.5,
        acce=acc_arr,
        frequency=100.0
    )

    global current_x, current_y, current_angle
    
    for step in step_lengths:
        step_length = step['step_length']
        start = step['start']
        end = step['end']

        # Turning estimation
        estimated_radian = estimate_turning_angle(
            gyro_data=gyro_arr[start:end, :],
            frequency=100.0
        )
        
        # 방향 업데이트
        current_angle -= estimated_radian
        
        # current_angle = round_angle(current_angle)
        
        # Coordinating
        step_length /= meter_per_coordinate_unit
        
        # 새로운 위치 계산
        dx = step_length * math.cos(current_angle) / coordinateWidth
        dy = step_length * math.sin(current_angle) / coordinateHeight
        
        # 방식 1. 장애물 고려x
        # current_x += dx
        # current_y += dy
        
        # 방식 2. 장애물 피해서 가기
        directions = [
            (dx, dy), # Move diagonally
            (0, dy),      # Move vertically only
            (dx, 0),      # Move horizontally only
            # (0, 0),            # Stay in place
        ]
        for dx_cand, dy_cand in directions:
            if map_constraints_check((current_x+dx_cand) * coordinateWidth, (current_y+dy_cand) * coordinateHeight):
                current_x += dx_cand
                current_y += dy_cand
                break
        
    res = {
        "isStepped": len(step_lengths) != 0,
        "x": current_x,
        "y": current_y,
        "radian": current_angle
    }

    print(f"Localization response: {json.dumps(res)}")

    return res


def handle_init(first_row: np.ndarray) -> bool:
    global current_x, current_y, current_angle

    current_x = float(first_row[0])
    current_y = float(first_row[1])
    current_angle = float(first_row[2])

    print(f"Initialization response: {True}")
    print(f"Initial State: ({current_x}, {current_y}, {current_angle})")

    return True


while True:
    message, clientAddress = serverSocket.recvfrom(1024)

    chunk_id, total_chunks, identifier, payload = message.decode('utf-8').split('/')
    # print(f"chunk_id, total_chunks, identifier, payload: {chunk_id}, {total_chunks}, {identifier}, {payload}")

    if identifier not in data_buffer:
        data_buffer[identifier] = {}
        expected_chunks[identifier] = int(total_chunks)

    data_buffer[identifier][int(chunk_id)] = payload
    print(f"len(data_buffer[identifier]: {len(data_buffer[identifier])}, expected_chunks[identifier]: {expected_chunks[identifier]}")

    if len(data_buffer[identifier]) == expected_chunks[identifier]:
        combined_data = ''.join(data_buffer[identifier][i] for i in range(1, int(total_chunks) + 1))
        csv_data = io.StringIO(combined_data)

        reader = csv.reader(csv_data)
        first_row = next(reader)
        print(f"first_row: {first_row}")

        if identifier == 'start':
            res = handle_init(first_row)
        elif identifier == 'update':
            pass
        elif identifier == 'locate':
            res = handle_localization(csv_data)

        serverSocket.sendto(json.dumps(res).encode(), clientAddress)

        csv_data.close()
        del data_buffer[identifier]
        del expected_chunks[identifier]