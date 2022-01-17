seed=666
retrain_type='mix'
config='configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml'

for num in {20,40,60,80,100}
do


#sim_cos
python retrain.py --seed $seed --config=$config --gpu $1 --sim_type sim_cos --task_type +- --retrain_type $retrain_type --num $num --train_type mix & 
python retrain.py --seed $seed --config=$config --gpu $2 --sim_type sim_cos --task_type + --retrain_type $retrain_type --num $num --train_type mix &
python retrain.py --seed $seed --config=$config --gpu $3 --sim_type sim_cos --task_type - --retrain_type $retrain_type --num $num --train_type mix

#gc
update=1
python retrain.py  --gpu $1 --sim_type gc --retrain_type $retrain_type --num $num --update $update  --task_type +- --train_type mix --seed $seed --config=$config&
python retrain.py  --gpu $2 --sim_type gc --retrain_type $retrain_type --num $num --update $update  --task_type + --train_type mix --seed $seed --config=$config&
python retrain.py  --gpu $3 --sim_type gc --retrain_type $retrain_type --num $num --update $update  --task_type - --train_type mix --seed $seed --config=$config
#gc update 5,10

for alpha in {0.1,0.2,0.5,1}
do
for update in {5,10}
do
python retrain.py  --gpu $1 --sim_type gc --retrain_type $retrain_type --num $num --update $update --alpha $alpha --task_type +- --train_type mix --seed $seed --config=$config&
python retrain.py  --gpu $2 --sim_type gc --retrain_type $retrain_type --num $num --update $update --alpha $alpha --task_type + --train_type mix --seed $seed --config=$config&
python retrain.py  --gpu $3 --sim_type gc --retrain_type $retrain_type --num $num --update $update --alpha $alpha --task_type - --train_type mix --seed $seed --config=$config
done
done

#random
python retrain.py --seed $seed --config=$config --gpu $1 --sim_type random --retrain_type $retrain_type --num $num --train_type mix 

done

