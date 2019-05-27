for sed in 1 2 3 4 5
do
for stmodel in 0 1 2
do
    python main_opennmt.py -train_student -manager nips_0512 -stModel $stmodel -stud_act relu -seed $sed
    python main_opennmt.py -train_student -manager nips_0512 -stModel $stmodel -stud_act prelu -seed $sed
    python main_opennmt.py -train_student -manager nips_0512 -stModel $stmodel -stud_act siwsh -seed $sed
    python main_opennmt.py -train_student -manager nips_0512 -stModel $stmodel -stud_act aplu -seed $sed
    python main_opennmt.py -train_student -manager nips_0512 -stModel $stmodel -stud_act lma -seed $sed
done
done
