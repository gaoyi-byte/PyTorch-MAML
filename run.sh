
retrain_type='mix'
config='configs/convnet4/mini-imagenet/5_way_5_shot/retrain1k.yaml'
task_type='+'
num=20
alpha=0.1
seed=666

python retrain.py --gpu 0 --sim_type sim_cos --train_type mix  --seed $seed --config=$config&
python retrain.py  --gpu 0 --sim_type gc --train_type mix --update 1  --config=$config --seed $seed&

for alpha in {0.1,0.2,0.5,1}
do
python retrain.py  --gpu 0 --sim_type gc --retrain_type $retrain_type --num $num --update 5 --alpha $alpha --task_type +- --train_type mix --seed $seed --config=$config&
done

for alpha in {0.1,0.2,0.5,1}
do
python retrain.py  --gpu 2 --sim_type gc --retrain_type $retrain_type --num $num --update 10 --alpha $alpha --task_type +- --train_type mix --seed $seed --config=$config&
done