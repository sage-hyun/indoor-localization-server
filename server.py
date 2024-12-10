from socket import *
import json
import math
import io

from step_length import *
from step_orientation import *
from utils import *

serverPort = 5000
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))
print(f"Server is ready to receive at {serverPort}")


current_x, current_y = (13, 52) # 초기 위치
current_angle = math.radians(0)  # 초기 방향 (radians)




while True:
    (message, clientAddress) = serverSocket.recvfrom(1024)
    
    # Bytes를 StringIO 객체로 변환
    csv_data = io.StringIO(message.decode('utf-8'))
    
    # NumPy로 CSV 로드 (첫 번째 행 건너뛰기)
    message_arr = np.genfromtxt(csv_data, delimiter=',', dtype=float, skip_header=1)
    
    acc_arr = message_arr[:,0:3]
    gyro_arr = message_arr[:,3:6]
    # mag_arr = message_arr[:,6:9]
    
        
    step_lengths = compute_step_timestamp(
        acceleration_threshold=0.001,
        weinberg_gain=0.5,
        acce=acc_arr,
        frequency=100.0
    )

    
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
        current_angle += estimated_radian 
        
        # current_angle = round_angle(current_angle)
        
        step_length /= 0.6
        # step_length /= 0.5
        
        # 새로운 위치 계산
        dx = step_length * math.cos(current_angle)
        dy = step_length * math.sin(current_angle)
        
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
            if map_constraints_check(current_x+dx_cand, current_y+dy_cand):
                current_x += dx_cand
                current_y += dy_cand
                break
        
    res = {
        "isStepped": len(step_lengths) != 0,
        "x": current_x,
        "y": current_y,
        "radian": current_angle
    }

    
    serverSocket.sendto(json.dumps(res).encode(), clientAddress)
