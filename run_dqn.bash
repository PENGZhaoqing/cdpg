# dqn
# ----2 agent----
# ---com---

# --basic signal 4--
#/usr/bin/python2.7 dqn/com/dqn_2a_com.py --eps-start 1.0 --eps-min 0.05 --replay-start-size 64 --decay-rate 50000 --replay-memory-size 1000000 --toxin 0 --signal-num 4  --epoch-num 40   --testing 0 --testing-epoch 39 --continue-training 0 --start-epoch 40 --a1-Qnet-folder basic_signal_4/a1_Qnet --a2-Qnet-folder basic_signal_4/a2_Qnet  --a1-CDPG-folder basic_signal_4/a1_CDPG  --a2-CDPG-folder basic_signal_4/a2_CDPG --save-log  basic_signal_4/log

# ---uncom---
# --basic--
#/usr/bin/python2.7 dqn/uncom/dqn_2a_uncom.py --eps-start 1.0 --eps-min 0.05 --replay-start-size 64 --decay-rate 50000 --replay-memory-size 1000000 --toxin 0 --signal-num 4  --epoch-num 40   --testing 0 --testing-epoch 39 --continue-training 0 --start-epoch 40 --a1-Qnet-folder basic/a1_Qnet --a2-Qnet-folder basic/a2_Qnet --save-log  basic/log

# ---central---
# --basic--
/usr/bin/python2.7 dqn/central/dqn_2a_central.py --eps-start 1.0  --eps-min 0.05 --replay-start-size 64 --decay-rate 50000 --replay-memory-size 1000000 --toxin 0 --signal-num 4  --epoch-num 40   --testing 0 --testing-epoch 39 --continue-training 0 --start-epoch 40 --a1-Qnet-folder basic/a1_Qnet  --save-log  basic/log

