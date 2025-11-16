#!/usr/bin/env sh
TS="$1"
TH="$2"
LLA=$(($TS + 1))

bash build_lla.sh "$LLA"
bash build_traffic.sh "$TS" "$TH"
bash find_vital_GS.sh "$TS" "$TH"

bash time_slot_analysis.sh "$TS" "$TH" 0.9
bash time_slot_analysis.sh "$TS" "$TH" 0.8
bash time_slot_analysis.sh "$TS" "$TH" 0.7
bash time_slot_analysis.sh "$TS" "$TH" 0.6
bash time_slot_analysis.sh "$TS" "$TH" 0.5

bash aggregated_deployment.sh "$TS" 0.9
bash aggregated_deployment.sh "$TS" 0.8
bash aggregated_deployment.sh "$TS" 0.7
bash aggregated_deployment.sh "$TS" 0.6
bash aggregated_deployment.sh "$TS" 0.5

bash get_results.sh
