for stmodel in 0 1 2
do
for sed in 1 2 3 4 5
do
for nb in 4 6 8 10 12
do
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act lma -plot_title 0514 -seed $sed -num_bins $nb
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $stmodel -stud_act aplu -plot_title 0514 -seed $sed -num_bins $nb
done
done
done
