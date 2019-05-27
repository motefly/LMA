for sed in 1 2 3 4 5
do
for stmodel in 0 1 2
do
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act relu -plot_title 0523 -seed $sed
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act lma -plot_title 0523 -seed $sed
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act swish -plot_title 0523 -seed $sed
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act aplu -plot_title 0523 -seed $sed
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act prelu -plot_title 0523 -seed $sed
done
done
