for model in 0 1
do
    python main_wrn.py -train_student -manager nips_0430 -stModel $model -test_memory -stud_act lma
    python main_wrn.py -train_student -manager nips_0430 -stModel $model -test_memory -stud_act relu
    python main_wrn.py -train_student -manager nips_0430 -stModel $model -test_memory -stud_act prelu
    python main_wrn.py -train_student -manager nips_0430 -stModel $model -test_memory -stud_act aplu
    python main_wrn.py -train_student -manager nips_0430 -stModel $model -test_memory -stud_act swish
    python main_wrn.py -train_teacher -manager nips_0430 -stModel $model -test_memory 
    python main.py -train_teacher -manager nips_0430 -stModel $model -test_memory
done
