set -e
export PYTHONIOENCODING=utf-8
echo "########## Phase B 시작 $(date +%H:%M:%S) ##########"
python train.py --preset B
echo "########## Phase C 시작 $(date +%H:%M:%S) ##########"
python train.py --preset C --eval_every 250
echo "########## 전부 완료 $(date +%H:%M:%S) ##########"
python gen.py --list
