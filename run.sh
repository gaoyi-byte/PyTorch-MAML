def convnet4()
{
python test.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/test.yaml --seed $1 --test_type com
python test.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/test.yaml --seed $1 --test_type mix
python test.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/test.yaml --seed $1 --test_type hmix
python test.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/test.yaml --seed $1 --test_type cls
python test.py --config=configs/convnet4/mini-imagenet/5_way_5_shot/test.yaml --seed $1 --test_type re
}

def resnet12(){
python test.py --seed 666 --gpu 2 --test_type com --config=configs/resnet12/mini-imagenet/5_way_5_shot/test.yaml  
python test.py --seed 666 --gpu 2 --test_type mix --config=configs/resnet12/mini-imagenet/5_way_5_shot/test.yaml 
#python test.py --seed 666 --gpu 2 --test_type hmix --config=configs/resnet12/mini-imagenet/5_way_5_shot/test.yaml


#python test.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/test.yaml --seed 666 --test_type hmix
#python test.py --config=configs/resnet12/mini-imagenet/5_way_5_shot/test.yaml --seed 666 --test_type cls
}

resnet12