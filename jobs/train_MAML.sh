#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=train_MAML
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=2-00:00
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
python run_trainer.py --shot_num 1 --data_root ./dataset/miniImageNet--ravi --conf_file ./config/Reproduce/MAML-miniImageNet-Conv64F-5-1.yaml
python run_trainer.py --shot_num 1 --data_root ./dataset/miniImageNet--ravi --conf_file ./config/Reproduce/MAML-miniImageNet-Conv64F-5-5.yaml

python run_trainer.py --shot_num 5 --data_root ./dataset/tiered_imagenet --conf_file ./config/Reproduce/MAML-miniImageNet-Conv64F-5-1.yaml
python run_trainer.py --shot_num 5 --data_root ./dataset/tiered_imagenet --conf_file ./config/Reproduce/MAML-miniImageNet-Conv64F-5-5.yaml

wait

cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/LibFewShot/results/MAML* ~/scratch/LibFewShot/results/