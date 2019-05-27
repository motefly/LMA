for dat in wmt integ
do
python main_opennmt.py -train_teacher -test_memory -data $dat
for stmodel in 0 1 2
do
    python main_opennmt.py -stModel $stmodel -stud_act relu -test_memory -data $dat
    python main_opennmt.py -stModel $stmodel -stud_act lma -test_memory -data $dat
    python main_opennmt.py -stModel $stmodel -stud_act swish -test_memory -data $dat
    python main_opennmt.py -stModel $stmodel -stud_act naplu -test_memory -data $dat
    python main_opennmt.py -stModel $stmodel -stud_act nprelu -test_memory -data $dat
done
done
done