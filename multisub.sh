sbatch -J CNO job_a100_extra.sh configs/snellius/cno_1.yaml
sbatch -J ATT job_a100_extra.sh configs/snellius/att_1.yaml
sbatch -J CNN job_a100.sh configs/snellius/cnn_1.yaml
