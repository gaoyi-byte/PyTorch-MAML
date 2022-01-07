
retrain_type='mix'
train_type='mix'
for num in {20,40,50,80,100}
do


for alpha in {1,0.5,0.1}
do
for task_type in {'+','+-','-'}
do
python retrain.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml --gpu $1 --sim_type gc --task_type $task_type --retrain_type $retrain_type --num $num --update 10 --alpha $alpha --train_type $train_type&
python retrain.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml --gpu $1 --sim_type gc --task_type $task_type --retrain_type $retrain_type --num $num --update 5 --alpha $alpha --train_type $train_type
done
done


#sim_cos
for task_type in {'+','+-','-'}
do
python retrain.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml --gpu $1 --sim_type gc --task_type $task_type --retrain_type $retrain_type --num $num --update 1 --train_type $train_type&
python retrain.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml --gpu $1 --sim_type sim_cos --task_type $task_type --retrain_type $retrain_type --num $num --update 1 --train_type $train_type
done

#random
#python retrain.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml --sim_type random --retrain_type mix --num 20 --update 1 --train_type mix &
#python retrain.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/retrain.yaml --gpu 0 --sim_type random  --retrain_type com --num $num --update 1 --train_type mix

done


