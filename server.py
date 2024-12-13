import csv
from socket import *
import json
import math
import io
import numpy as np
import os
from datetime import datetime

from step_length import *
from step_orientation import *
from utils import *

serverPort = 5003
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print(f"Server is ready to receive at {serverPort}")

defaultRes = {
    "isStepped": False,
    "x": 0.0,
    "y": 0.0,
    "radian": 0.0,
    "landmark": "None"
}

current_x, current_y = (13, 52) # 초기 위치
current_angle = 0.0  # 초기 방향 (radians)

trajectory = []

meter_per_coordinate_unit = 0.6

coordinateWidth = 153.0
coordinateHeight = 63.0

# 수신 중인 메시지 저장소
data_buffer = {}
expected_chunks = {}
last_received_time = {}

packet_loss_time = 0.0

init_ori = 0

def handle_localization(combined_data: str):
    # NumPy로 CSV 로드 (첫 번째 행 건너뛰기)
    csv_data = io.StringIO(combined_data)

    reader = csv.reader(csv_data)
    _ = next(reader)

    message_arr = np.genfromtxt(csv_data, delimiter=',', dtype=float)

    acc_arr = message_arr[:,0:3]
    gyro_arr = message_arr[:,3:6]
    # mag_arr = message_arr[:,6:9]
    ori_arr = message_arr[:,6:9]

    if len(acc_arr) <= 9:
        csv_data.close()
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"{current_time} [DR]: not enough acceleretometer data")
        return defaultRes
        
    step_lengths = compute_step_timestamp(
        acceleration_threshold=0.01,
        weinberg_gain=0.7,
        acce=acc_arr,
        frequency=100.0
    )

    global current_x, current_y, current_angle, trajectory, init_ori
    
    for step in step_lengths:
        step_length = step['step_length']
        start = step['start']
        end = step['end']

        # Turning estimation
        # estimated_radian = estimate_turning_angle(
        #     gyro_data=gyro_arr[start:end, :],
        #     frequency=100.0
        # )
        
        if init_ori == 0:
            init_ori = math.radians(ori_arr[end][0])
        print(ori_arr[end][0])
        estimated_radian = math.radians(ori_arr[end][0]) - init_ori
        
        # 방향 업데이트
        current_angle = estimated_radian
                
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
            PC_x = (current_x+dx_cand) * coordinateWidth
            PC_y = coordinateHeight - (current_y+dy_cand) * coordinateHeight
            if map_constraints_check(PC_x, PC_y):
                current_x += dx_cand
                current_y += dy_cand
                break
            else:
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                print(f"{current_time}: [DR]: {current_x+dx_cand}, {current_y+dy_cand} is in obstacle-contraints")

        # 방식 3.
        # PC_x = (current_x+dx) * coordinateWidth
        # PC_y = coordinateHeight - (current_y+dy) * coordinateHeight
        # if map_constraints_check(PC_x, PC_y):
        #     current_x += dx / 3.0
        #     current_y += dy / 3.0
        #     break
        # else:
        #     current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        #     print(f"{current_time}: [DR]: {current_x+dx}, {current_y+dy} is in obstacle-contraints")
    
    landmark = get_landmark(current_x, current_y)
    res = {
        "isStepped": len(step_lengths) != 0,
        "x": current_x,
        "y": current_y,
        "radian": current_angle,
        "landmark": landmark,
    }

    if len(step_lengths) != 0:
        trajectory.append((current_x, current_y, current_angle, landmark, False))

    csv_data.close()
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"{current_time}: [DR]: Localization response: {json.dumps(res)}")

    return res


def handle_init(combined_data: str) -> bool:
    global current_x, current_y, current_angle, trajectory

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    first_row = combined_data.split(',')

    current_x = float(first_row[0])
    current_y = float(first_row[1])
    current_angle = float(first_row[2])

    init_ori = 0

    trajectory.append((current_x, current_y, current_angle, "None", False))

    print(f"{current_time}: [Start]: Initialization response: {True}")
    print(f"{current_time}: [Start]: Initial State: ({current_x}, {current_y}, {current_angle})")

    return True


def handle_update(combined_data: str):
    global current_x, current_y, current_angle, trajectory

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    first_row = combined_data.split(',')

    estimate_x = float(first_row[0])
    estimate_y = float(first_row[1])
    WiFi_only = first_row[2] == 'true'

    est_x = estimate_x * coordinateHeight / coordinateWidth
    est_y = estimate_y

    current_x = (current_x + est_x) / 2.0
    current_y = (current_y + est_y) / 2.0
    print(f"{current_time}: [WiFi]: R + WiFi Calibrated")

    res = {
        "x": current_x,
        "y": current_y,
        "radian": current_angle,
    }
    
    trajectory.append((current_x, current_y, current_angle, get_landmark(current_x, current_y), True))

    return res


def handle_end():
    global trajectory

    # 파일 경로 설정
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = './trajectory'

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, f'trajectory_{current_time}.csv')

    # CSV 파일로 저장
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'orientation', 'landmark', 'WiFiCalibrated'])  # 헤더 작성
        writer.writerows(trajectory)  # 데이터 작성

    print(f'{current_time}: [Finish] The file has been saved: {filename}')
    
    trajectory = []
    del data_buffer[identifier]
    del expected_chunks[identifier]
    del last_received_time[identifier]

    return True

while True:
    try:
        message, clientAddress = serverSocket.recvfrom(1024)
        start_time = datetime.now()

        chunk_id, total_chunks, identifier, payload = message.decode('utf-8').split('/')

        if identifier not in data_buffer:
            data_buffer[identifier] = {}
            expected_chunks[identifier] = int(total_chunks)
            last_received_time[identifier] = start_time

        data_buffer[identifier][int(chunk_id)] = payload
        last_received_time[identifier] = start_time

        # Update elapsed time for packet loss detection
        elapsed_time = (datetime.now() - start_time).total_seconds()
        # print(f"Elapsed Time: {elapsed_time}")

        # Check if all chunks are received or timeout
        if (
            elapsed_time >= 3.0 or 
            len(data_buffer[identifier]) >= expected_chunks[identifier]
        ):
            # Check for missing chunks
            missing_chunks = [
                i for i in range(1, expected_chunks[identifier] + 1) 
                if i not in data_buffer[identifier]
            ]

            print(f"Missing chunks: {missing_chunks}")
            # Optionally: Request retransmission here if supported

            combined_data = ''.join(
                data_buffer[identifier][i] 
                for i in range(1, expected_chunks[identifier] + 1)
            )

            if identifier == 'start':
                res = handle_init(combined_data)
            elif identifier == 'update':
                res = handle_update(combined_data)
            elif identifier == 'locate':
                res = handle_localization(combined_data)
            elif identifier == 'end':
                res = handle_end()

            serverSocket.sendto(json.dumps(res).encode(), clientAddress)

            # Cleanup
            del data_buffer[identifier]
            del expected_chunks[identifier]
            del last_received_time[identifier]

            packet_loss_time = 0.0

    except Exception as e:
        print(f"Error occurred: {e}")
