for stmodel in 0 2
do
for sed in 1 2 3 4 5
do
    python main_opennmt.py -train_student -manager nips_0514 -stModel $stmodel -stud_act prelu -seed $sed -data wmt
done
done
