for model in 0 1 2
do
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $model -stud_act aplu -test_memory
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $model -stud_act prelu -test_memory
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $model -stud_act relu -test_memory
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $model -stud_act swish -test_memory
    python main.py -train_student -manager nips_0430 -init_lr 1e-2 -stModel $model -stud_act lma -test_memory
done
