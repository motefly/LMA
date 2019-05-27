for sed in 1 2 3 4 5
do
for stmodel in 0 1
do
    python main_wrn.py -train_student -manager nips_0430 -stModel $stmodel -stud_act relu -seed $sed
    python main_wrn.py -train_student -manager nips_0430 -stModel $stmodel -stud_act prelu -seed $sed
    python main_wrn.py -train_student -manager nips_0430 -stModel $stmodel -stud_act lma -seed $sed
    python main_wrn.py -train_student -manager nips_0430 -stModel $stmodel -stud_act swish -seed $sed
    python main_wrn.py -train_student -manager nips_0430 -stModel $stmodel -stud_act aplu -seed $sed
done
done
