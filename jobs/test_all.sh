#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=test_all
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-06:00
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

declare -a proto_list=(
    "ProtoNet-miniImageNet--ravi-Conv64F-5-1-Seed0-Nov-06-2021-17-11-08"
    "ProtoNet-tiered_imagenet-Conv64F-5-1-Seed0-Nov-06-2021-17-21-05"
    "ProtoNet-miniImageNet--ravi-Conv64F-5-5-Seed0-Nov-06-2021-17-28-29"
    "ProtoNet-tiered_imagenet-Conv64F-5-5-Seed0-Nov-06-2021-17-39-29")

declare -a maml_list=(
    "MAML-tiered_imagenet-Conv64F-5-1-Seed0-Nov-06-2021-17-45-18"
    "MAML-tiered_imagenet-Conv64F-5-5-Seed0-Nov-06-2021-19-42-31"
    "MAML-miniImageNet--ravi-Conv64F-5-1-Seed0-Nov-06-2021-16-56-02"
    "MAML-miniImageNet--ravi-Conv64F-5-5-Seed0-Nov-06-2021-18-33-14")

declare -a baseline_list=(
    "Baseline-miniImageNet--ravi-Conv64F-5-1-Seed0-Nov-06-2021-16-33-24"
    "Baseline-miniImageNet--ravi-Conv64F-5-5-Seed0-Nov-06-2021-17-33-37"
    ""
    "")


method_list=("${proto_list[@]}" "${maml_list[@]}" "${baseline_list[@]}")
for result_method in ${method_list[@]}; do
    echo $result_method
    python run_test.py --result_method $result_method
    cp -r $SLURM_TMPDIR/LibFewShot/results/$result_method/log_files ~/scratch/LibFewShot/results/$result_method/
done
