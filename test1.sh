retrain_type='mix'
config='configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml'
num=10
update=1
seed=666




for alpha in {0.1,0.2,0.5,1}
do

python retrain.py  --gpu 2 --sim_type gc --retrain_type $retrain_type --num $num --update 5 --alpha $alpha --task_type +- --train_type mix --seed $seed --config=$config&
python retrain.py  --gpu 3 --sim_type gc --retrain_type $retrain_type --num $num --update 10 --alpha $alpha --task_type + --train_type mix --seed $seed --config=$config&

done


