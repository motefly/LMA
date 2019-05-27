for sed in 1 2 3 4 5
do
for stmodel in 0 1 2
do
    python main_joint.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act relu -plot_title 0514 -seed $sed -numBits 4,8
    python main_joint.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act lma -plot_title 0514 -seed $sed -numBits 4,8
done
done
