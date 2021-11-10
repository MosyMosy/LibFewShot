#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=Proto_tierd_freeitter
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=2-12:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load python/3.7
source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/LibFewShot .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/dataset/LibFewShot/* .

echo "Extract to dataset folder"
date +"%T"
cd LibFewShot/dataset

# tar -xf $SLURM_TMPDIR/CIFAR100.tar.gz
# tar -xf $SLURM_TMPDIR/CUB_200_2011_FewShot.tar.gz
# tar -xf $SLURM_TMPDIR/CUB_birds_2010.tar.gz
# tar -xf $SLURM_TMPDIR/StanfordCar.tar.gz
# tar -xf $SLURM_TMPDIR/StanfordDog.tar.gz

tar -xf $SLURM_TMPDIR/miniImageNet--ravi.tar.gz
cat $SLURM_TMPDIR/tieredImageNet.tar.gz* | tar -zxf -

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"

cd ..

for shot in 1 5; do
    python run_trainer.py --shot_num $shot --train_episode 10000 --training_test_episode 1000 --epoch 100 --test_epoch 5 --tag freeitter --conf_file ./config/proto.yaml --data_root ./dataset/tiered_imagenet
    mv $SLURM_TMPDIR/LibFewShot/temp/* ~/scratch/LibFewShot/results/

done

cp -r $SLURM_TMPDIR/LibFewShot/temp/* ~/scratch/LibFewShot/results/
