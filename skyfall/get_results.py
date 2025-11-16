#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import math
import json
import sys
import os
from collections import Counter

f = open("../config.json", "r", encoding='utf8')
table = json.load(f)
cons_name = table["Name"]
altitude = int(table["Altitude (km)"])
orbit_num = table["# of orbit"]
sat_per_cycle = table["# of satellites"]
inclination = table["Inclination"] 
average_gsl_num = 574 # average_gsl_num for starlink

ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
# 설치된(생성된) 데이터만 대상으로 동적으로 결정
topologies = [t for t in ['+grid', 'circle']
              if os.path.isdir('../' + cons_name + '/' + t + '_data')]
# grid가 기본 베이스이므로 최소한 +grid는 존재해야 함
bot_num = 0  # total number of malicious terminals
traffic_thre = 20  # upmost 20 malicious terminals accessed to a satellite
GSL_capacity = 4096
unit_traffic = 20   # 20Mbps per malicious terminal
RADIUS = 6371


if __name__ == "__main__":
    if not os.path.exists('../' + cons_name + '/results'): 
        subdirs = [
            "fig-10a", "fig-10b", "fig-10c", 
            "fig-11a", "fig-11b", "fig-12a", "fig-12b", 
            "fig-13a", "fig-13b", "fig-14a", "fig-14b"
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join('../' + cons_name + '/results', subdir), exist_ok=True)
    
    # fig-10, fig-11a, fig-11b
    legal_traffic = []
    base_path = '../' + cons_name + '/' + topologies[0] + "_data/link_traffic_data"
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
    for subdir in sorted(subdirs, key=int):
        subdir_path = os.path.join('../' + cons_name + '/' + topologies[0] + "_data/link_traffic_data", subdir)
        if os.path.isdir(subdir_path):
            # background traffic
            downlink_traffic_file_path = os.path.join(subdir_path, 'downlink_traffic.txt')
            if os.path.exists(downlink_traffic_file_path):
                with open(downlink_traffic_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    total = sum(values)
                    legal_traffic.append(total / 1024)
                    
    folder_path = '../' + cons_name + '/' + topologies[0] + "_data/attack_traffic_data_land_only_bot/0.5-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic)
    skyfall_traffic_after_aggregation = []
    traffic_ratios = []
    number_attacked_gsl =[]
    gsl_ratios = []
    
    time_slot = 0
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.isdigit()]
    for subdir in sorted(subdirs, key=int):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            # (Ratio of) reduced traffic by skyfall
            skyfall_bot_num_file_path = os.path.join(subdir_path, 'bot_num.txt')
            bot_num = 0
            if os.path.exists(skyfall_bot_num_file_path):
                with open(skyfall_bot_num_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    bot_num = sum(values)
            skyfall_traffic_file_path = os.path.join(subdir_path, 'cumu_affected_traffic_volume.txt')
            if os.path.exists(skyfall_traffic_file_path):
                with open(skyfall_traffic_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    total = sum(values) 
                    skyfall_traffic_after_aggregation.append((total * max(1, bot_num / 950)) / 1024 )
                    traffic_ratios.append(skyfall_traffic_after_aggregation[-1] / legal_traffic[time_slot])
            # (Ratio of) attacked GSLs by skyfall
            attack_gsl_file_path = os.path.join(subdir_path, 'attack_gsl.txt')
            if os.path.exists(attack_gsl_file_path):
                with open(attack_gsl_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    number_attacked_gsl.append(len(values))
                    gsl_ratios.append((len(values)) / average_gsl_num)
        time_slot += 1
    output_file = '../' + cons_name + '/results' + '/fig-10a/ratio_of_reduced_background_traffic_by_skyfall.txt'
    with open(output_file, 'w') as file:
        for value in traffic_ratios:
            file.write(str(value) + '\n')
    output_file = '../' + cons_name + '/results' + '/fig-10a/ratio_of_attacked_GSLs_by_skyfall.txt'
    with open(output_file, 'w') as file:
        for value in gsl_ratios:
            file.write(str(value) + '\n')
    output_file = '../' + cons_name + '/results' + '/fig-11a/number_attacked_GSLs_starlink_cdf.txt'
    with open(output_file, 'w') as file:
        sorted_number_attacked_gsl = np.sort(number_attacked_gsl)
        for value in sorted_number_attacked_gsl:
            file.write(str(value) + '\n')
    output_file = '../' + cons_name + '/results' + '/fig-11b/number_attacked_GSLs_starlink_box.txt'
    with open(output_file, 'w') as file:
        file.write("max: " + str(max(number_attacked_gsl)) + '\n')
        file.write("min: " + str(min(number_attacked_gsl)) + '\n')
        file.write("average: " + str(sum(number_attacked_gsl) / len(number_attacked_gsl)) + '\n')    
    output_file = '../' + cons_name + '/results' + '/fig-10c/background_traffic_without_attack.txt'
    with open(output_file, 'w') as file:
        for value in legal_traffic:
            file.write(str(value) + '\n')
            
    actual_throughput_skyfall_after_aggregation = [a - b for a, b in zip(legal_traffic, skyfall_traffic_after_aggregation)]
    output_file = '../' + cons_name + '/results' + '/fig-10c/actual_throughput_skyfall_after_aggregation.txt'
    with open(output_file, 'w') as file:
        for value in actual_throughput_skyfall_after_aggregation:
            file.write(str(value) + '\n')

    folder_path = '../' + cons_name + '/' + topologies[0] + "_data/attack_traffic_data_land_only_bot/0.5-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic) + "-comparison"
    icarus_traffic = []
    traffic_ratios = []
    gsl_ratios = []

    time_slot = 0
    if os.path.isdir(folder_path):
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            # (Ratio of) reduced traffic by icarus
            icarus_traffic_file_path = os.path.join(subdir_path, 'cumu_affected_traffic_volume_950.txt')
            if os.path.exists(icarus_traffic_file_path):
                with open(icarus_traffic_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    total = sum(values)
                    icarus_traffic.append(total / 1024)
                    traffic_ratios.append(icarus_traffic[-1] / legal_traffic[time_slot])
            # (Ratio of) attacked GSLs by icarus
            attack_gsl_file_path = os.path.join(subdir_path, 'attack_gsl_950.txt')
            if os.path.exists(attack_gsl_file_path):
                with open(attack_gsl_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    if values != []:
                        gsl_ratios.append((len(values)) / average_gsl_num)
                    else:
                        gsl_ratios.append(0)
        time_slot += 1
    
    output_file = '../' + cons_name + '/results' + '/fig-10b/ratio_of_reduced_background_traffic_by_icarus.txt'
    with open(output_file, 'w') as file:
        for value in traffic_ratios:
            file.write(str(value) + '\n')
    output_file = '../' + cons_name + '/results' + '/fig-10b/ratio_of_attacked_GSLs_by_icarus.txt'
    with open(output_file, 'w') as file:
        for value in gsl_ratios:
            file.write(str(value) + '\n')
    actual_throughput_icarus = [a - b for a, b in zip(legal_traffic, icarus_traffic)]
    output_file = '../' + cons_name + '/results' + '/fig-10c/actual_throughput_icarus.txt'
    with open(output_file, 'w') as file:
        for value in actual_throughput_icarus:
            file.write(str(value) + '\n')
                

    # fig-12a, fig-12b
    for topology in topologies:
        botnet_size_for_skyfall = []
        botnet_size_for_icarus = []
        for ratio in ratios:
            skyfall_folder_path = '../' + cons_name + '/' + topology + "_data/attack_traffic_data_land_only_bot/" + str(ratio) + "-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic)
            icarus_folder_path = '../' + cons_name + '/' + topology + "_data/attack_traffic_data_land_only_bot/" + str(ratio) + "-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic) + "-comparison"
            # skyfall bot_num 
            skyfall_botnum_file_path = os.path.join(skyfall_folder_path, 'bot_num.txt')
            if os.path.exists(skyfall_botnum_file_path):
                with open(skyfall_botnum_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    botnet_size_for_skyfall.append(values[0])
            # icarus bot_num 
            icarus_bot_size = []
            if os.path.isdir(icarus_folder_path):
                for subdir in os.listdir(icarus_folder_path):
                    subdir_path = os.path.join(icarus_folder_path, subdir)
                    if os.path.isdir(subdir_path):
                        icarus_botnum_file_path = os.path.join(subdir_path, 'bot_num.txt')
                        if os.path.exists(icarus_botnum_file_path):
                            with open(icarus_botnum_file_path, 'r') as file:
                                values = [float(line.strip()) for line in file]
                                if values[0] > 0:
                                    icarus_bot_size.append(values[0])
            botnet_size_for_icarus.append(int(sum(icarus_bot_size) / len(icarus_bot_size)) if icarus_bot_size else 0)
        sub_dir = 'fig-12a' if topology == topologies[0] else 'fig-12b'
            
        output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/botnet_size_for_skyfall.txt'
        with open(output_file, 'w') as file:
            for i in range (len(ratios)):
                file.write(str(round(1-ratios[i], 1)) + ": " + str(botnet_size_for_skyfall[i]) + '\n')
        output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/botnet_size_for_icarus.txt'
        with open(output_file, 'w') as file:
            for i in range (len(ratios)):
                file.write(str(round(1-ratios[i], 1)) + ": " + str(botnet_size_for_icarus[i]) + '\n')

            
    # fig-13a, fig-13b
    for topology in topologies:
        number_blocks_skyfall_after_trimming = []
        number_blocks_icarus = []
        for index, ratio in enumerate(ratios):
            skyfall_folder_path = '../' + cons_name + '/' + topology + "_data/attack_traffic_data_land_only_bot/" + str(ratio) + "-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic)
            icarus_folder_path = '../' + cons_name + '/' + topology + "_data/attack_traffic_data_land_only_bot/" + str(ratio) + "-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic) + "-comparison"
            # skyfall block_num 
            skyfall_block_num_file_path = os.path.join(skyfall_folder_path, 'bot_block_trim.txt')
            if os.path.exists(skyfall_block_num_file_path):
                with open(skyfall_block_num_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    number_blocks_skyfall_after_trimming.append(len(values))
            skyfall_block_num_file_path = os.path.join(skyfall_folder_path, 'bot_block.txt')
            # icarus block_num 
            icarus_block_num = []
            if os.path.isdir(icarus_folder_path):
                for subdir in os.listdir(icarus_folder_path):
                    subdir_path = os.path.join(icarus_folder_path, subdir)
                    if os.path.isdir(subdir_path):
                        icarus_block_num_file_path = os.path.join(subdir_path, 'block_num.txt')
                        if os.path.exists(icarus_block_num_file_path):
                            with open(icarus_block_num_file_path, 'r') as file:
                                values = [float(line.strip()) for line in file]
                                icarus_block_num.append(values[0])
            number_blocks_icarus.append(int(sum(icarus_block_num) / len(icarus_block_num)) if icarus_block_num else 0)

        sub_dir = 'fig-13a' if topology == topologies[0] else 'fig-13b'
            
        output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/botnet_size_for_skyfall.txt'

        output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/number_blocks_skyfall_' + topology + '.txt'
        with open(output_file, 'w') as file:
            for i in range (len(ratios)):
                file.write(str(round(1-ratios[i], 1)) + ": " + str(number_blocks_skyfall_after_trimming[i]) + '\n')
        output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/number_blocks_icarus_' + topology + '.txt'
        with open(output_file, 'w') as file:
            for i in range (len(ratios)):
                file.write(str(round(1-ratios[i], 1)) + ": " + str(number_blocks_icarus[i]) + '\n')

                
    # fig-14a, fig-14b
    for topology in topologies:
        # grid/circle 구분을 인덱스가 아니라 이름으로
        is_grid = (topology == '+grid')
        malicious_uplink_throughput_degradation = [[] for _ in ratios]  # throughput for various degradation
        background_traffic = []
        for index, ratio in enumerate(ratios):
            malicious_uplink_throughput = np.zeros(sat_per_cycle * orbit_num)
            if is_grid:
                for sat_id in range(-int(ratio*120)+96):
                    malicious_uplink_throughput[sat_id] = unit_traffic * (sat_id%4 +1)
            else:
                for sat_id in range(-int(ratio*100)+140):
                    malicious_uplink_throughput[sat_id] = unit_traffic * (sat_id%4 +1)
            skyfall_folder_path = '../' + cons_name + '/' + topology + "_data/attack_traffic_data_land_only_bot/" + str(ratio) + "-" +  str(traffic_thre) + "-" + str(sat_per_cycle) + "-" + str(GSL_capacity) + "-" + str(unit_traffic) + "/0/"
            # skyfall bot_num 
            bot_sat_file_path = os.path.join(skyfall_folder_path, 'bot_sat.txt')
            if os.path.exists(bot_sat_file_path):
                with open(bot_sat_file_path, 'r') as file:
                    values = [float(line.strip()) for line in file]
                    count = Counter(values)
                    for key, value in count.items():
                        malicious_uplink_throughput[int(key)] = int(value * unit_traffic) if value <= traffic_thre else unit_traffic * traffic_thre
                    malicious_uplink_throughput_degradation[index] = np.sort(malicious_uplink_throughput)
                # 현재 topology의 백그라운드 트래픽을 읽음
        downlink_traffic = []
        topo_link_dir = '../' + cons_name + '/' + topology + "_data/link_traffic_data/0"
        if os.path.isdir(topo_link_dir):
            downlink_traffic_file_path = os.path.join(topo_link_dir, 'downlink_traffic.txt')
            if os.path.exists(downlink_traffic_file_path):
                with open(downlink_traffic_file_path, 'r') as file:
                    downlink_traffic = [float(line.strip()) for line in file]
                    downlink_traffic = np.sort(downlink_traffic)

        # 결과 디렉토리 선택 (grid→fig-14a, circle→fig-14b)
        sub_dir = 'fig-14a' if is_grid else 'fig-14b'
        for i in range(len(ratios)):
            output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/malicious_uplink_throughput_degradation_' + str(int(100-100*ratios[i])) + '_percent.txt'
            with open(output_file, 'w') as file:
                for value in malicious_uplink_throughput_degradation[i]:
                    file.write(str(value) + '\n')
        output_file = '../' + cons_name + '/results' + '/' + sub_dir + '/background_traffic.txt'
        with open(output_file, 'w') as file:
            for value in downlink_traffic:
                file.write(str(value) + '\n')

    
