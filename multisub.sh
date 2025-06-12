#sbatch -J CNO job_a100_extra.sh configs/snellius/cno_1.yaml
sbatch -J ATT job_a100_extra.sh configs/snellius/att_1.yaml
sbatch -J CNN job_a100.sh configs/snellius/cnn_1.yaml

sbatch -J backsol job_a100.sh configs/snellius/cnn_backsol.yaml
sbatch -J interp job_a100.sh configs/snellius/cnn_interp.yaml
sbatch -J rk4 job_a100.sh configs/snellius/cnn_rk4.yaml

sbatch -J CNN_proj job_a100.sh configs/snellius/cnn_proj.yaml
sbatch -J CNN_nopr job_a100.sh configs/snellius/cnn_noproj.yaml
