seed=666
#retrain_type='mix'
config='configs/convnet4/mini-imagenet/5_way_5_shot/retrain1k.yaml'

for num in {20,40,50,60,80,100}
do
for retrain_type in {'mix','com'}
do
echo 'sim_cos',$num,$retrain_type


for alpha in {0.1,0.2,0.5,1}
do
for update in {5,10}
do

echo 'gc',$num,$retrain_type,$alpha,$update
done
done

#random
echo 'random'

done
done
